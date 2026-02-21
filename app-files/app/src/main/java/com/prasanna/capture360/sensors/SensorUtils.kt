package com.prasanna.capture360.sensors

object SensorUtils {

    fun normalizeAngle(angle: Float): Float {
        val tau = (2 * Math.PI).toFloat()
        var a = angle % tau
        if (a > Math.PI) a -= tau
        if (a < -Math.PI) a += tau
        return a
    }

    fun radToDeg(rad: Float): Float {
        return Math.toDegrees(rad.toDouble()).toFloat()
    }

    fun degToRad(deg: Float): Float {
        return Math.toRadians(deg.toDouble()).toFloat()
    }
}

