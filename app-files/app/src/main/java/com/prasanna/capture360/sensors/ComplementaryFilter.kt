package com.prasanna.capture360.sensors

import kotlin.math.PI
import kotlin.math.abs

class ComplementaryFilter(
    private val baseAlpha: Float = 0.98f   // 98% gyro, 2% absolute correction (base value)
) {

    private var fusedYaw = 0f
    private var fusedPitch = 0f
    private var fusedRoll = 0f
    private var isInitialized = false
    private var adaptiveAlpha = baseAlpha

    // Track sensor anomalies for adaptive filtering
    private var largeErrorCount = 0
    private var stabilityFrames = 0
    private val ANOMALY_THRESHOLD = 0.5f // radians; if error exceeds this, sensor is unreliable
    private val STABILIZE_FRAMES = 10    // frames needed to recover confidence

    fun update(
        gyroOrientation: Orientation,
        absoluteOrientation: Orientation
    ): Orientation {

        // On the very first frame, we don't filter; we snap to the absolute position
        if (!isInitialized) {
            fusedYaw = absoluteOrientation.yaw
            fusedPitch = absoluteOrientation.pitch
            fusedRoll = absoluteOrientation.roll
            isInitialized = true
            return absoluteOrientation
        }

        // Apply the filter with Shortest-Path logic for each axis
        fusedYaw = applyFilter(fusedYaw, gyroOrientation.yaw, absoluteOrientation.yaw)
        fusedPitch = applyFilter(fusedPitch, gyroOrientation.pitch, absoluteOrientation.pitch)
        fusedRoll = applyFilter(fusedRoll, gyroOrientation.roll, absoluteOrientation.roll)

        return Orientation(fusedYaw, fusedPitch, fusedRoll)
    }

    private fun applyFilter(currentFused: Float, gyroVal: Float, absVal: Float): Float {
        // 1. Gyro integration part (the "prediction")
        val predicted = gyroVal

        // 2. Calculate the difference between prediction and absolute sensor
        var diff = absVal - predicted

        // 3. Normalize the difference to [-PI, PI] to take the shortest path
        while (diff < -PI) diff += (2 * PI).toFloat()
        while (diff > PI) diff -= (2 * PI).toFloat()

        // 4. Adaptive weighting: if the error is unusually large, reduce trust in gyro
        val correctionWeight = if (abs(diff) > ANOMALY_THRESHOLD) {
            // Large error: sensor anomaly detected, use less gyro (more correction)
            largeErrorCount++
            stabilityFrames = 0
            0.5f // Use 50% gyro, 50% absolute for recovery
        } else {
            // Normal operation: slowly recover confidence in gyro
            if (largeErrorCount > 0) stabilityFrames++
            if (stabilityFrames > STABILIZE_FRAMES) {
                largeErrorCount = 0
                stabilityFrames = 0
            }
            1 - adaptiveAlpha
        }

        // 5. The core Complementary Filter equation with adaptive weight:
        // NewValue = Prediction + (AdaptiveCorrectionWeight * Error)
        return predicted + correctionWeight * diff
    }
}

