class SensorFusionManager(context: Context) : SensorEventListener {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val magnet = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

    private val filter = ComplementaryFilter()

    private var accelData = FloatArray(3)
    private var magnetData = FloatArray(3)

    private var gyroYaw = 0f
    private var gyroPitch = 0f
    private var gyroRoll = 0f

    private var lastTimestamp: Long = 0L

    private var fusedOrientation = Orientation(0f, 0f, 0f)

    fun start() {
        sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_GAME)
        sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_GAME)
        sensorManager.registerListener(this, magnet, SensorManager.SENSOR_DELAY_GAME)
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {

        when (event.sensor.type) {

            Sensor.TYPE_GYROSCOPE -> {
                if (lastTimestamp != 0L) {
                    val dt = (event.timestamp - lastTimestamp) * 1e-9f
                    gyroYaw += event.values[2] * dt
                    gyroPitch += event.values[1] * dt
                    gyroRoll += event.values[0] * dt
                }
                lastTimestamp = event.timestamp
            }

            Sensor.TYPE_ACCELEROMETER -> {
                accelData = event.values.clone()
            }

            Sensor.TYPE_MAGNETIC_FIELD -> {
                magnetData = event.values.clone()
            }
        }

        computeFusedOrientation()
    }

    private fun computeFusedOrientation() {

        val R = FloatArray(9)
        val I = FloatArray(9)

        if (SensorManager.getRotationMatrix(R, I, accelData, magnetData)) {

            val orientation = FloatArray(3)
            SensorManager.getOrientation(R, orientation)

            val absolute = Orientation(
                yaw = orientation[0],
                pitch = orientation[1],
                roll = orientation[2]
            )

            val gyro = Orientation(gyroYaw, gyroPitch, gyroRoll)

            fusedOrientation = filter.update(gyro, absolute)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    fun getOrientation(): Orientation = fusedOrientation
}
