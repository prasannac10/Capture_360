package com.yourcompany.capture360.tracking

import com.yourcompany.capture360.sensors.Orientation
import com.yourcompany.capture360.sensors.SensorUtils

class AngularCoverageTracker(
    private val yawBins: Int = 36,
    private val pitchBins: Int = 18
) {

    private val covered = Array(yawBins) {
        BooleanArray(pitchBins)
    }

    fun update(orientation: Orientation) {

        val yawDeg = SensorUtils.radToDeg(orientation.yaw)
        val pitchDeg = SensorUtils.radToDeg(orientation.pitch)

        val yawIndex = ((yawDeg + 180) / 360 * yawBins).toInt()
        val pitchIndex = ((pitchDeg + 90) / 180 * pitchBins).toInt()

        if (yawIndex in 0 until yawBins &&
            pitchIndex in 0 until pitchBins) {
            covered[yawIndex][pitchIndex] = true
        }
    }

    fun completion(): Float {
        val total = yawBins * pitchBins
        val done = covered.sumOf { row -> row.count { it } }
        return done.toFloat() / total
    }

    fun getGrid(): Array<BooleanArray> = covered
}
