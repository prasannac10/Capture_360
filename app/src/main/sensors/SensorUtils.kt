package com.yourcompany.capture360.sensors

object SensorUtils {

    fun normalizeAngle(angle: Float): Float {
        var a = angle
        while (a > Math.PI) a -= (2 * Math.PI).toFloat()
        while (a < -Math.PI) a += (2 * Math.PI).toFloat()
        return a
    }

    fun radToDeg(rad: Float): Float {
        return Math.toDegrees(rad.toDouble()).toFloat()
    }

    fun degToRad(deg: Float): Float {
        return Math.toRadians(deg.toDouble()).toFloat()
    }
}
