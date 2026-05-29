package com.prasanna.capture360.stitching

import android.util.Log
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.DMatch
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc

class PanoramaStitcher {

    companion object {
        private const val TAG = "PanoramaStitcher"
        private const val MIN_MATCH_COUNT = 10
        private const val RANSAC_THRESHOLD = 4.0
        private const val RATIO_THRESHOLD = 0.75f
        private const val MAX_FEATURES = 5000
        private const val MAX_IMAGE_DIM = 1200
    }

    sealed class StitchResult {
        data class Success(val panorama: Mat) : StitchResult()
        data class Error(val message: String) : StitchResult()
    }

    fun stitch(images: List<Mat>): StitchResult {
        if (images.size < 2) {
            return StitchResult.Error("Need at least 2 images")
        }

        Log.d(TAG, "Starting stitch of ${images.size} images")

        val resized = images.map { resizeIfNeeded(it) }
        var stitchedCount = 0

        try {
            var result = resized[0].clone()
            Log.d(TAG, "Base image: ${result.cols()}x${result.rows()}")

            for (i in 1 until resized.size) {
                Log.d(TAG, "Stitching image ${i + 1}/${resized.size} (${resized[i].cols()}x${resized[i].rows()})")
                val stitched = stitchPair(result, resized[i])
                if (stitched != null) {
                    result.release()
                    result = stitched
                    stitchedCount++
                    Log.d(TAG, "Stitch ${i + 1} SUCCESS -> panorama now ${result.cols()}x${result.rows()}")
                } else {
                    Log.w(TAG, "Stitch ${i + 1} FAILED (not enough matching features), skipping")
                }
            }

            if (stitchedCount == 0) {
                result.release()
                return StitchResult.Error("Could not match any image pairs. Try capturing with more overlap between frames.")
            }

            Log.d(TAG, "Stitching complete: $stitchedCount pairs merged, final size ${result.cols()}x${result.rows()}")
            return StitchResult.Success(result)
        } catch (e: Exception) {
            Log.e(TAG, "Stitching failed", e)
            return StitchResult.Error("Stitching failed: ${e.message}")
        }
    }

    private fun resizeIfNeeded(img: Mat): Mat {
        val maxDim = maxOf(img.cols(), img.rows())
        if (maxDim <= MAX_IMAGE_DIM) return img.clone()

        val scale = MAX_IMAGE_DIM.toDouble() / maxDim
        val resized = Mat()
        Imgproc.resize(img, resized, Size(img.cols() * scale, img.rows() * scale))
        return resized
    }

    private fun stitchPair(base: Mat, newImage: Mat): Mat? {
        val orb = ORB.create(MAX_FEATURES)

        val kp1 = MatOfKeyPoint()
        val desc1 = Mat()
        val kp2 = MatOfKeyPoint()
        val desc2 = Mat()

        orb.detectAndCompute(base, Mat(), kp1, desc1)
        orb.detectAndCompute(newImage, Mat(), kp2, desc2)

        Log.d(TAG, "Features: base=${kp1.toList().size}, new=${kp2.toList().size}")

        if (desc1.empty() || desc2.empty() || kp1.toList().size < 4 || kp2.toList().size < 4) {
            Log.w(TAG, "Insufficient features detected")
            cleanup(kp1, desc1, kp2, desc2)
            return null
        }

        val goodMatches = findGoodMatches(desc1, desc2)
        Log.d(TAG, "Good matches after ratio test: ${goodMatches.size}")

        if (goodMatches.size < MIN_MATCH_COUNT) {
            Log.w(TAG, "Not enough good matches: ${goodMatches.size} < $MIN_MATCH_COUNT")
            cleanup(kp1, desc1, kp2, desc2)
            return null
        }

        val kpList1 = kp1.toList()
        val kpList2 = kp2.toList()

        // Map new image points -> base image points
        val srcPoints = MatOfPoint2f(*goodMatches.map {
            kpList2[it.trainIdx].pt
        }.toTypedArray())

        val dstPoints = MatOfPoint2f(*goodMatches.map {
            kpList1[it.queryIdx].pt
        }.toTypedArray())

        val inlierMask = MatOfByte()
        val homography = Calib3d.findHomography(
            srcPoints, dstPoints, Calib3d.RANSAC, RANSAC_THRESHOLD, inlierMask
        )

        // Count inliers
        val inlierCount = inlierMask.toList().count { it.toInt() == 1 }
        Log.d(TAG, "Homography inliers: $inlierCount / ${goodMatches.size}")

        srcPoints.release()
        dstPoints.release()
        inlierMask.release()
        cleanup(kp1, desc1, kp2, desc2)

        if (homography.empty() || inlierCount < MIN_MATCH_COUNT) {
            Log.w(TAG, "Homography failed or too few inliers")
            if (!homography.empty()) homography.release()
            return null
        }

        val result = warpAndBlend(base, newImage, homography)
        homography.release()
        return result
    }

    private fun findGoodMatches(desc1: Mat, desc2: Mat): List<DMatch> {
        // Use kNN matching with Lowe's ratio test for better quality
        val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
        val knnMatches = mutableListOf<MatOfDMatch>()
        matcher.knnMatch(desc1, desc2, knnMatches, 2)

        val goodMatches = mutableListOf<DMatch>()
        for (matchPair in knnMatches) {
            val matchList = matchPair.toList()
            if (matchList.size >= 2) {
                val best = matchList[0]
                val secondBest = matchList[1]
                if (best.distance < RATIO_THRESHOLD * secondBest.distance) {
                    goodMatches.add(best)
                }
            }
            matchPair.release()
        }

        return goodMatches.sortedBy { it.distance }
    }

    private fun warpAndBlend(base: Mat, newImage: Mat, homography: Mat): Mat {
        // Transform corners of newImage to find bounding box in base coordinate space
        val corners = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(newImage.cols().toDouble(), 0.0),
            Point(newImage.cols().toDouble(), newImage.rows().toDouble()),
            Point(0.0, newImage.rows().toDouble())
        )

        val transformedCorners = MatOfPoint2f()
        Core.perspectiveTransform(corners, transformedCorners, homography)
        val pts = transformedCorners.toArray()
        corners.release()
        transformedCorners.release()

        // Compute output canvas bounds (union of base image and warped newImage)
        var minX = 0.0
        var minY = 0.0
        var maxX = base.cols().toDouble()
        var maxY = base.rows().toDouble()

        for (pt in pts) {
            if (pt.x < minX) minX = pt.x
            if (pt.y < minY) minY = pt.y
            if (pt.x > maxX) maxX = pt.x
            if (pt.y > maxY) maxY = pt.y
        }

        val width = (maxX - minX).toInt()
        val height = (maxY - minY).toInt()

        if (width <= 0 || height <= 0 || width > 15000 || height > 8000) {
            Log.w(TAG, "Invalid panorama dimensions: ${width}x${height}, returning base")
            return base.clone()
        }

        // Translation matrix to shift everything so minX,minY becomes 0,0
        val translation = Mat.eye(3, 3, CvType.CV_64F)
        translation.put(0, 2, -minX)
        translation.put(1, 2, -minY)

        val adjustedH = Mat()
        Core.gemm(translation, homography, 1.0, Mat(), 0.0, adjustedH)

        val panoSize = Size(width.toDouble(), height.toDouble())

        // Warp the new image into the output canvas
        val warpedNew = Mat()
        Imgproc.warpPerspective(newImage, warpedNew, adjustedH, panoSize)

        // Place the base image into the output canvas (just translation, no warp)
        val warpedBase = Mat()
        Imgproc.warpPerspective(base, warpedBase, translation, panoSize)

        translation.release()
        adjustedH.release()

        // Create masks for blending
        val baseMask = Mat()
        Imgproc.cvtColor(warpedBase, baseMask, Imgproc.COLOR_BGR2GRAY)
        Imgproc.threshold(baseMask, baseMask, 1.0, 255.0, Imgproc.THRESH_BINARY)

        val newMask = Mat()
        Imgproc.cvtColor(warpedNew, newMask, Imgproc.COLOR_BGR2GRAY)
        Imgproc.threshold(newMask, newMask, 1.0, 255.0, Imgproc.THRESH_BINARY)

        // Start with warped new image as base canvas, then overlay base image on top
        // Base image gets priority where it exists (preserves accumulated panorama)
        val result = warpedNew.clone()
        warpedBase.copyTo(result, baseMask)

        baseMask.release()
        newMask.release()
        warpedBase.release()
        warpedNew.release()

        return cropBlackBorders(result)
    }

    private fun cropBlackBorders(img: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY)
        val thresh = Mat()
        Imgproc.threshold(gray, thresh, 1.0, 255.0, Imgproc.THRESH_BINARY)
        gray.release()

        val points = Mat()
        Core.findNonZero(thresh, points)
        thresh.release()

        if (points.empty()) {
            points.release()
            return img
        }

        val rect = Imgproc.boundingRect(points)
        points.release()

        if (rect.width <= 0 || rect.height <= 0) return img

        val cropped = Mat(img, rect)
        val result = cropped.clone()
        cropped.release()
        return result
    }

    private fun cleanup(vararg mats: Mat) {
        mats.forEach { it.release() }
    }
}
