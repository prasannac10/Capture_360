package com.prasanna.capture360.vision

import org.opencv.core.*
import org.opencv.features2d.ORB
import org.opencv.features2d.BFMatcher
import org.opencv.imgproc.Imgproc

class FeatureTracker {
    private val orb = ORB.create()
    private val matcher = BFMatcher.create(Core.NORM_HAMMING, true)

    private var lastDescriptors = Mat()
    private var lastKeypoints = MatOfKeyPoint()

    fun estimateYawCorrection(currentFrame: Mat): Float {
        val gray = Mat()
        Imgproc.cvtColor(currentFrame, gray, Imgproc.COLOR_BGR2GRAY)

        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()

        // Detect ORB features
        orb.detectAndCompute(gray, Mat(), keypoints, descriptors)
        gray.release()

        // If it's our first frame, just store and wait for the next
        if (lastDescriptors.empty() || descriptors.empty()) {
            descriptors.copyTo(lastDescriptors)
            keypoints.copyTo(lastKeypoints)
            // Cleanup local Mats
            keypoints.release()
            descriptors.release()
            return 0f
        }

        val matches = MatOfDMatch()
        matcher.match(lastDescriptors, descriptors, matches)

        // Sort matches by distance (lower is better) and take top 50
        val sortedMatches = matches.toList().sortedBy { it.distance }.take(50)

        var yawShift = 0f
        if (sortedMatches.size >= 10) {
            val prevPts = lastKeypoints.toList()
            val currPts = keypoints.toList()

            val horizontalShifts = sortedMatches.map { match ->
                currPts[match.trainIdx].pt.x - prevPts[match.queryIdx].pt.x
            }
            yawShift = horizontalShifts.average().toFloat()
        }

        // Clean up old "last" data
        lastDescriptors.release()
        lastKeypoints.release()

        // Move current data to "last" data for next frame
        // We use copyTo to ensure the native memory is preserved correctly
        lastDescriptors = Mat()
        descriptors.copyTo(lastDescriptors)
        lastKeypoints = MatOfKeyPoint()
        keypoints.copyTo(lastKeypoints)

        // Cleanup current frame local Mats
        matches.release()
        keypoints.release()
        descriptors.release()

        return yawShift * 0.05f
    }
}

