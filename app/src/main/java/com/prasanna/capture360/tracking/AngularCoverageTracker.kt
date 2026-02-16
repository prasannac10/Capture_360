package com.prasanna.capture360.tracking

import android.graphics.PointF
import com.prasanna.capture360.sensors.Orientation
import kotlin.math.*

class AngularCoverageTracker(
    private val yawBins: Int = 10,
    private val pitchBins: Int = 3
) {
    private val covered = Array(yawBins) { BooleanArray(pitchBins) }
    private var currentTarget: Pair<Int, Int>? = null // Lock onto one bin

    private val totalTargets = yawBins * pitchBins

    // Cache commonly used values to reduce repeated calculations
    private val yawPerBin = 360f / yawBins
    private val pitchPerBin = 180f / pitchBins

    // Helper: convert orientation to bin indices
    private fun getBinIndices(orientation: Orientation): Pair<Int, Int> {
        val yawDeg = Math.toDegrees(orientation.yaw.toDouble()).toFloat()
        val pitchDeg = Math.toDegrees(orientation.pitch.toDouble()).toFloat()

        val yIdx = (((yawDeg + 180) / 360) * yawBins).toInt().coerceIn(0, yawBins - 1)
        val pIdx = (((pitchDeg + 90) / 180) * pitchBins).toInt().coerceIn(0, pitchBins - 1)

        return yIdx to pIdx
    }

    fun update(orientation: Orientation) {
        val (yIdx, pIdx) = getBinIndices(orientation)
        covered[yIdx][pIdx] = true

        // If we just captured our locked target, clear it so we can find a new one
        if (currentTarget == Pair(yIdx, pIdx)) {
            currentTarget = null
        }
    }

    fun getRemainingCount(): Int {
        val done = covered.sumOf { row -> row.count { it } }
        return totalTargets - done
    }

    // Get the center point of the current bin
    fun getTargetCenter(orientation: Orientation): PointF {
        val (yIdx, pIdx) = getBinIndices(orientation)

        return PointF(
            (yIdx * yawPerBin) - 180f + (yawPerBin / 2),
            (pIdx * pitchPerBin) - 90f + (pitchPerBin / 2)
        )
    }

    fun getTargetDirection(curYaw: Float, curPitch: Float, rotation: Int): String {
        val target = currentTarget ?: findNearestUncaptured(curYaw, curPitch) ?: return "ALL COVERED"

        val tYaw = (target.first * yawPerBin) - 180f + (yawPerBin / 2)
        val tPitch = (target.second * pitchPerBin) - 90f + (pitchPerBin / 2)

        val dy = calculateYawDiff(curYaw, tYaw)
        val dp = tPitch - curPitch

        // Adjust instructions based on screen orientation
        return when (rotation) {
            android.view.Surface.ROTATION_90, android.view.Surface.ROTATION_270 -> {
                // Landscape
                when {
                    dy > 15 -> "↑ TILT UP"
                    dy < -15 -> "↓ TILT DOWN"
                    dp > 10 -> "→ ROTATE RIGHT"
                    dp < -10 -> "← ROTATE LEFT"
                    else -> "HOLD STEADY"
                }
            }
            else -> {
                // Portrait (Standard)
                when {
                    dp > 10 -> "↑ TILT UP"
                    dp < -10 -> "↓ TILT DOWN"
                    dy > 15 -> "→ ROTATE RIGHT"
                    dy < -15 -> "← ROTATE LEFT"
                    else -> "HOLD STEADY"
                }
            }
        }
    }

    private fun findNearestUncaptured(curYaw: Float, curPitch: Float): Pair<Int, Int>? {
        var bestBin: Pair<Int, Int>? = null
        var minDistance = Float.MAX_VALUE

        for (p in 0 until pitchBins) {
            for (y in 0 until yawBins) {
                if (!covered[y][p]) {
                    val tYaw = (y * yawPerBin) - 180f + (yawPerBin / 2)
                    val tPitch = (p * pitchPerBin) - 90f + (pitchPerBin / 2)
                    val dist = sqrt(calculateYawDiff(curYaw, tYaw).pow(2) + (tPitch - curPitch).pow(2))
                    if (dist < minDistance) {
                        minDistance = dist
                        bestBin = y to p
                    }
                }
            }
        }
        return bestBin
    }

    private fun calculateYawDiff(current: Float, target: Float): Float {
        var diff = target - current
        while (diff <= -180) diff += 360
        while (diff > 180) diff -= 360
        return diff
    }

    fun completion(): Float = (covered.sumOf { row -> row.count { it } }.toFloat() / totalTargets) * 100f
    fun getGrid() = covered

    fun isCurrentAreaCovered(orientation: Orientation): Boolean {
        val (yIdx, pIdx) = getBinIndices(orientation)
        return covered[yIdx][pIdx]
    }
}

