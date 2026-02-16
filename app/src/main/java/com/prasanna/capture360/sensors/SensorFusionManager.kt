package com.prasanna.capture360.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import kotlin.math.sqrt

class SensorFusionManager(context: Context) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val magnet = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

    private val filter = ComplementaryFilter(0.98f)

    private var accelData = FloatArray(3)
    private var magnetData = FloatArray(3)
    private var hasAccel = false
    private var hasMagnet = false

    // Accumulated Gyro values
    private var gyroYaw = 0f
    private var gyroPitch = 0f
    private var gyroRoll = 0f

    private var lastTimestamp: Long = 0L
    private var fusedOrientation = Orientation(0f, 0f, 0f)

    // Sensor data validation
    private val ACCEL_GRAVITY = 9.81f
    private val ACCEL_TOLERANCE = 2.0f // Allow 2 m/s² deviation from gravity
    private val MAGNET_VALID_RANGE = 50f // Earth's magnetic field is ~25-65 μT

    fun start() {
        sensorManager.registerListener(this, gyro, SensorManager.SENSOR_DELAY_GAME)
        sensorManager.registerListener(this, accel, SensorManager.SENSOR_DELAY_GAME)
        sensorManager.registerListener(this, magnet, SensorManager.SENSOR_DELAY_GAME)
    }

    fun stop() {
        sensorManager.unregisterListener(this)
        lastTimestamp = 0L // Reset for next start
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                if (lastTimestamp != 0L) {
                    val dt = (event.timestamp - lastTimestamp) * 1e-9f
                    // Only integrate if dt is reasonable (avoid huge jumps on resumption)
                    if (dt > 0 && dt < 0.1f) {
                        gyroYaw += event.values[2] * dt
                        gyroPitch += event.values[1] * dt
                        gyroRoll += event.values[0] * dt
                    }
                }
                lastTimestamp = event.timestamp
            }
            Sensor.TYPE_ACCELEROMETER -> {
                // Validate accelerometer data: magnitude should be close to gravity
                val magnitude = sqrt(event.values[0] * event.values[0] +
                        event.values[1] * event.values[1] +
                        event.values[2] * event.values[2])

                if (isValidAccelData(magnitude)) {
                    System.arraycopy(event.values, 0, accelData, 0, 3)
                    hasAccel = true
                }
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                // Validate magnetometer data: magnitude should be within Earth's field range
                val magnitude = sqrt(event.values[0] * event.values[0] +
                        event.values[1] * event.values[1] +
                        event.values[2] * event.values[2])

                if (isValidMagnetData(magnitude)) {
                    System.arraycopy(event.values, 0, magnetData, 0, 3)
                    hasMagnet = true
                }
            }
        }

        // Only compute if we have data from all sensors AND they are valid
        if (hasAccel && hasMagnet) {
            computeFusedOrientation()
        }
    }

    private fun isValidAccelData(magnitude: Float): Boolean {
        // Check if magnitude is within expected range (gravity ± tolerance)
        val withinRange = magnitude in (ACCEL_GRAVITY - ACCEL_TOLERANCE)..(ACCEL_GRAVITY + ACCEL_TOLERANCE)
        if (!withinRange) {
            Log.w("SensorFusion", "Invalid accel data: magnitude=$magnitude")
        }
        return withinRange
    }

    private fun isValidMagnetData(magnitude: Float): Boolean {
        // Earth's magnetic field is ~25-65 μT
        val withinRange = magnitude in 20f..70f
        if (!withinRange) {
            Log.w("SensorFusion", "Invalid magnet data: magnitude=$magnitude (possible interference)")
        }
        return withinRange
    }

    private fun computeFusedOrientation() {
        val R = FloatArray(9)
        val I = FloatArray(9)

        if (SensorManager.getRotationMatrix(R, I, accelData, magnetData)) {

            // Remap coordinates for Portrait Mode (Vertical)
            // World-X -> Device-X, World-Y -> Device-Z
            val outR = FloatArray(9)
            SensorManager.remapCoordinateSystem(R, SensorManager.AXIS_X, SensorManager.AXIS_Z, outR)

            val orientationValues = FloatArray(3)
            SensorManager.getOrientation(outR, orientationValues)

            val absolute = Orientation(
                yaw = orientationValues[0],
                pitch = orientationValues[1],
                roll = orientationValues[2]
            )

            // Current integrated gyro state
            val gyroState = Orientation(gyroYaw, gyroPitch, gyroRoll)

            // Fusion: Absolute corrects the Gyro drift
            fusedOrientation = filter.update(gyroState, absolute)

            // IMPORTANT: Sync the gyro accumulation back to the fused value
            // to prevent the gyro from "fighting" the correction
            gyroYaw = fusedOrientation.yaw
            gyroPitch = fusedOrientation.pitch
            gyroRoll = fusedOrientation.roll
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    fun getOrientation(): Orientation = fusedOrientation
}

