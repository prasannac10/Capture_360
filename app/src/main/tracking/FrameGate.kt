class FrameGate {

    private var lastOrientation: Orientation? = null

    fun shouldCapture(current: Orientation): Boolean {

        val prev = lastOrientation ?: run {
            lastOrientation = current
            return false
        }

        val deltaYaw = abs(current.yaw - prev.yaw)
        val deltaPitch = abs(current.pitch - prev.pitch)

        val stable = deltaYaw < 0.015f && deltaPitch < 0.015f
        val movedEnough = deltaYaw > 0.2f || deltaPitch > 0.2f

        lastOrientation = current
        return stable && movedEnough
    }
}
