package com.prasanna.capture360.tracking

import com.prasanna.capture360.sensors.Orientation
import kotlin.math.abs
import kotlin.math.sqrt

class FrameGate {
    // We store the orientation of the last photo we actually TOOK
    private var lastCapturedOrientation: Orientation? = null

    // Sensitivity settings (11-15 degrees is standard for 360 apps)
    private val motionThreshold = 0.25f // ~14 degrees

    // Motion stability detection to reduce jitter
    private val motionHistory = mutableListOf<Float>()
    private val HISTORY_SIZE = 5
    private val STABILITY_THRESHOLD = 0.08f // radians per frame; lower = steadier required

    fun shouldCapture(current: Orientation): Boolean {
        // If it's the first time, take a photo immediately to start the process
        val last = lastCapturedOrientation ?: run {
            lastCapturedOrientation = current
            return true
        }

        // Calculate movement speed: how fast we're moving from previous orientation
        val deltaYaw = abs(current.yaw - last.yaw)
        val deltaPitch = abs(current.pitch - last.pitch)
        val speed = sqrt(deltaYaw * deltaYaw + deltaPitch * deltaPitch)

        // Track motion smoothness by recording speeds
        motionHistory.add(speed)
        if (motionHistory.size > HISTORY_SIZE) {
            motionHistory.removeAt(0)
        }

        // Check if device motion is stable (low jitter)
        val isStable = isMotionStable()

        // Did we move enough to justify a new frame AND is motion stable?
        val movedEnough = deltaYaw > motionThreshold || deltaPitch > motionThreshold

        if (movedEnough && isStable) {
            // Update the baseline so we don't capture again until we move another threshold
            lastCapturedOrientation = current
            return true
        }

        return false
    }

    private fun isMotionStable(): Boolean {
        // Need enough history to evaluate stability
        if (motionHistory.size < HISTORY_SIZE / 2) {
            return true // Allow early captures
        }

        // Calculate average motion / variance
        val avg = motionHistory.average()

        // If average motion is very low, it's stable
        if (avg < STABILITY_THRESHOLD) {
            return true
        }

        // If we have jitter (high variance), reject the frame
        val variance = motionHistory.map { (it - avg) * (it - avg) }.average()
        return variance < 0.01f // Low variance = smooth motion
    }

    // Call this if the user wants to restart
    fun reset() {
        lastCapturedOrientation = null
        motionHistory.clear()
    }
}

