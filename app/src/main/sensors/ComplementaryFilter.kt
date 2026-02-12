class ComplementaryFilter(
    private val alpha: Float = 0.98f   // 98% gyro, 2% absolute correction
) {

    private var fusedYaw = 0f
    private var fusedPitch = 0f
    private var fusedRoll = 0f

    fun update(
        gyroOrientation: Orientation,
        absoluteOrientation: Orientation
    ): Orientation {

        fusedYaw = alpha * gyroOrientation.yaw +
                (1 - alpha) * absoluteOrientation.yaw

        fusedPitch = alpha * gyroOrientation.pitch +
                (1 - alpha) * absoluteOrientation.pitch

        fusedRoll = alpha * gyroOrientation.roll +
                (1 - alpha) * absoluteOrientation.roll

        return Orientation(fusedYaw, fusedPitch, fusedRoll)
    }
}
