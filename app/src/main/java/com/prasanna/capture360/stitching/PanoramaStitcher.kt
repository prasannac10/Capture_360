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
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.features2d.ORB
import org.opencv.features2d.BFMatcher
import org.opencv.imgproc.Imgproc

class PanoramaStitcher {

    companion object {
        private const val TAG = "PanoramaStitcher"
        private const val MIN_MATCH_COUNT = 15
        private const val RANSAC_THRESHOLD = 5.0
        private const val MAX_IMAGE_DIM = 800
    }

    sealed class StitchResult {
        data class Success(val panorama: Mat) : StitchResult()
        data class Error(val message: String) : StitchResult()
    }

    fun stitch(images: List<Mat>): StitchResult {
        if (images.size < 2) {
            return StitchResult.Error("Need at least 2 images")
        }

        Log.d(TAG, "Stitching ${images.size} images")

        val resized = images.map { resizeIfNeeded(it) }

        try {
            var result = resized[0].clone()

            for (i in 1 until resized.size) {
                Log.d(TAG, "Stitching image ${i + 1}/${resized.size}")
                val next = resized[i]
                val stitched = stitchPair(result, next)
                if (stitched != null) {
                    result.release()
                    result = stitched
                } else {
                    Log.w(TAG, "Could not stitch image ${i + 1}, skipping")
                }
            }

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

    private fun stitchPair(left: Mat, right: Mat): Mat? {
        val orb = ORB.create(2000)
        val matcher = BFMatcher.create(Core.NORM_HAMMING, true)

        val kp1 = MatOfKeyPoint()
        val desc1 = Mat()
        val kp2 = MatOfKeyPoint()
        val desc2 = Mat()

        orb.detectAndCompute(left, Mat(), kp1, desc1)
        orb.detectAndCompute(right, Mat(), kp2, desc2)

        if (desc1.empty() || desc2.empty()) {
            Log.w(TAG, "No features detected in one of the images")
            cleanup(kp1, desc1, kp2, desc2)
            return null
        }

        val matches = MatOfDMatch()
        matcher.match(desc1, desc2, matches)

        val matchList = matches.toList().sortedBy { it.distance }
        matches.release()

        if (matchList.size < MIN_MATCH_COUNT) {
            Log.w(TAG, "Not enough matches: ${matchList.size}")
            cleanup(kp1, desc1, kp2, desc2)
            return null
        }

        val goodMatches = filterGoodMatches(matchList)
        Log.d(TAG, "Good matches: ${goodMatches.size}")

        if (goodMatches.size < MIN_MATCH_COUNT) {
            Log.w(TAG, "Not enough good matches: ${goodMatches.size}")
            cleanup(kp1, desc1, kp2, desc2)
            return null
        }

        val kpList1 = kp1.toList()
        val kpList2 = kp2.toList()

        val srcPoints = MatOfPoint2f(*goodMatches.map {
            kpList2[it.trainIdx].pt
        }.toTypedArray())

        val dstPoints = MatOfPoint2f(*goodMatches.map {
            kpList1[it.queryIdx].pt
        }.toTypedArray())

        val mask = MatOfByte()
        val homography = Calib3d.findHomography(
            srcPoints, dstPoints, Calib3d.RANSAC, RANSAC_THRESHOLD, mask
        )

        srcPoints.release()
        dstPoints.release()
        mask.release()
        cleanup(kp1, desc1, kp2, desc2)

        if (homography.empty()) {
            Log.w(TAG, "Could not find homography")
            return null
        }

        val result = warpAndBlend(left, right, homography)
        homography.release()
        return result
    }

    private fun filterGoodMatches(matches: List<DMatch>): List<DMatch> {
        if (matches.isEmpty()) return emptyList()
        val minDist = matches.first().distance
        val threshold = maxOf(2.0f * minDist, 30.0f)
        return matches.filter { it.distance < threshold }.take(200)
    }

    private fun warpAndBlend(left: Mat, right: Mat, homography: Mat): Mat {
        val corners = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(right.cols().toDouble(), 0.0),
            Point(right.cols().toDouble(), right.rows().toDouble()),
            Point(0.0, right.rows().toDouble())
        )

        val transformedCorners = MatOfPoint2f()
        Core.perspectiveTransform(corners, transformedCorners, homography)

        val pts = transformedCorners.toArray()
        corners.release()
        transformedCorners.release()

        var minX = 0.0
        var minY = 0.0
        var maxX = left.cols().toDouble()
        var maxY = left.rows().toDouble()

        for (pt in pts) {
            if (pt.x < minX) minX = pt.x
            if (pt.y < minY) minY = pt.y
            if (pt.x > maxX) maxX = pt.x
            if (pt.y > maxY) maxY = pt.y
        }

        val width = (maxX - minX).toInt()
        val height = (maxY - minY).toInt()

        if (width <= 0 || height <= 0 || width > 10000 || height > 10000) {
            Log.w(TAG, "Invalid panorama dimensions: ${width}x${height}")
            return left.clone()
        }

        val translation = Mat.eye(3, 3, CvType.CV_64F)
        translation.put(0, 2, -minX)
        translation.put(1, 2, -minY)

        val adjustedH = Mat()
        Core.gemm(translation, homography, 1.0, Mat(), 0.0, adjustedH)

        val panoSize = Size(width.toDouble(), height.toDouble())
        val warpedRight = Mat()
        Imgproc.warpPerspective(right, warpedRight, adjustedH, panoSize)

        val warpedLeft = Mat()
        Imgproc.warpPerspective(left, warpedLeft, translation, panoSize)

        translation.release()
        adjustedH.release()

        // Blend: place left on top of warped right where left has content
        val result = warpedRight.clone()
        val leftMask = Mat()
        Imgproc.cvtColor(warpedLeft, leftMask, Imgproc.COLOR_BGR2GRAY)
        val leftBinary = Mat()
        Imgproc.threshold(leftMask, leftBinary, 1.0, 255.0, Imgproc.THRESH_BINARY)
        leftMask.release()

        warpedLeft.copyTo(result, leftBinary)

        leftBinary.release()
        warpedLeft.release()
        warpedRight.release()

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
