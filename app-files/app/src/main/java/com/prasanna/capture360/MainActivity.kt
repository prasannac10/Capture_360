package com.prasanna.capture360

import kotlin.math.sqrt
import kotlin.math.pow
import kotlin.math.abs
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.prasanna.capture360.camera.CameraController
import com.prasanna.capture360.sensors.SensorFusionManager
import com.prasanna.capture360.tracking.AngularCoverageTracker
import com.prasanna.capture360.tracking.FrameGate
import com.prasanna.capture360.ui.GuidanceOverlayView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import com.prasanna.capture360.sensors.Orientation

import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File

class MainActivity : AppCompatActivity() {

    private var lastCaptureYaw = 0f
    private var lastCapturePitch = 0f
    private var isCameraBusy = false // Prevents overlapping hardware calls
    private lateinit var camera: CameraController
    private lateinit var fusion: SensorFusionManager
    private lateinit var tracker: AngularCoverageTracker
    private val gate = FrameGate()

    private lateinit var preview: PreviewView
    private lateinit var overlay: GuidanceOverlayView
    private lateinit var progressBar: ProgressBar
    private lateinit var statusText: TextView
    private lateinit var btnFinish: Button
    private lateinit var btnCancel: Button
    private lateinit var btnRetry: Button
    private lateinit var errorContainer: LinearLayout
    private lateinit var errorText: TextView
    private lateinit var captureCountText: TextView
    private lateinit var rotationAngleText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initOpenCV()
        setContentView(R.layout.activity_main)

        // Initialize UI elements
        preview = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        progressBar = findViewById(R.id.progressBar)
        statusText = findViewById(R.id.statusText)
        btnFinish = findViewById(R.id.btnFinish)
        btnCancel = findViewById(R.id.btnCancel)
        btnRetry = findViewById(R.id.btnRetry)
        errorContainer = findViewById(R.id.errorContainer)
        errorText = findViewById(R.id.errorText)
        captureCountText = findViewById(R.id.captureCount)
        rotationAngleText = findViewById(R.id.rotationAngle)

        // Button listeners
        btnFinish.setOnClickListener { initiateStitching() }
        btnCancel.setOnClickListener { resetCapture() }
        btnRetry.setOnClickListener { retryCamera() }

        if (allPermissionsGranted()) {
            startCaptureLogic()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(
                    Manifest.permission.CAMERA,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE,
                    Manifest.permission.READ_EXTERNAL_STORAGE
                ),
                1001
            )
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1001) {
            if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                Log.d("Permissions", "All permissions granted, starting capture logic")
                startCaptureLogic()
            } else {
                Log.w("Permissions", "Permissions denied: ${grantResults.joinToString()}")
                statusText.text = "Permissions required: Camera + Storage"
            }
        }
    }

    private fun startCaptureLogic() {
        Log.d("MainActivity", "startCaptureLogic called")
        
        try {
            // 1. Initialize Components
            camera = CameraController(this, this)
            fusion = SensorFusionManager(this)
            tracker = AngularCoverageTracker(10, 3)
            Log.d("MainActivity", "Components initialized successfully")

            // 2. Clear old images
            try {
                getExternalFilesDir(null)?.listFiles()?.forEach {
                    if (it.name.startsWith("capture_")) it.delete()
                }
            } catch (e: Exception) {
                Log.w("MainActivity", "Could not delete old images", e)
            }

            // 3. Start Camera (this should show preview immediately)
            Log.d("MainActivity", "Starting camera...")
            camera.startCamera(preview)
            Log.d("MainActivity", "Camera started")

            // Hide error container and show preview
            errorContainer.visibility = View.GONE
            preview.visibility = View.VISIBLE

            // 4. Start Sensors
            Log.d("MainActivity", "Starting sensors...")
            fusion.start()
            Log.d("MainActivity", "Sensors started")

            statusText.text = "Ready - Rotate to capture"

            // 5. Main control loop - SIMPLIFIED
            lifecycleScope.launch {
                Log.d("MainActivity", "Control loop started")
                var frameCount = 0
                while (isActive) {
                    try {
                        frameCount++
                        val orientation = fusion.getOrientation()

                        val curYaw = Math.toDegrees(orientation.yaw.toDouble()).toFloat()
                        val curPitch = Math.toDegrees(orientation.pitch.toDouble()).toFloat()
                        val rotation = getScreenRotation()

                        val remaining = tracker.getRemainingCount()
                        val capturedCount = 30 - remaining
                        val guidance = tracker.getTargetDirection(curYaw, curPitch, rotation)

                        runOnUiThread {
                            overlay.updateDisplay(tracker.getGrid(), "$guidance\n($capturedCount/30)")
                            progressBar.progress = ((capturedCount.toFloat() / 30f) * 100).toInt()
                            captureCountText.text = "$capturedCount/30"
                            rotationAngleText.text = "Yaw: ${curYaw.toInt()}° Pitch: ${curPitch.toInt()}°"
                            if (capturedCount >= 20) btnFinish.visibility = View.VISIBLE
                        }

                        // Simple capture: just check if we've moved and area is uncovered
                        val yawDiff = calculateYawDiff(curYaw, lastCaptureYaw)
                        val pitchDiff = abs(curPitch - lastCapturePitch)

                        if (!isCameraBusy && remaining > 0 && (abs(yawDiff) > 15f || pitchDiff > 10f) && !tracker.isCurrentAreaCovered(orientation)) {
                            triggerImageCapture(curYaw, curPitch, orientation)
                            lastCaptureYaw = curYaw
                            lastCapturePitch = curPitch
                        }

                        delay(100)
                    } catch (e: Exception) {
                        Log.e("ControlLoop", "Error in frame $frameCount", e)
                        if (frameCount % 100 == 0) {
                            Log.d("ControlLoop", "Still running, frame $frameCount")
                        }
                    }
                }
            }

        } catch (e: Exception) {
            Log.e("MainActivity", "startCaptureLogic failed", e)
            showError("Error: ${e.message}")
            if (e.message?.contains("Camera") == true) {
                btnRetry.visibility = View.VISIBLE
            }
        }
    }

    private fun triggerImageCapture(yaw: Float, pitch: Float, orientation: Orientation) {
        if (isCameraBusy) return
        
        isCameraBusy = true
        val dir = getExternalFilesDir(null)
        
        if (dir == null) {
            Log.e("Capture", "External files directory is null")
            isCameraBusy = false
            return
        }

        val file = File(dir, "capture_${System.currentTimeMillis()}.jpg")
        
        camera.captureFrame(file) {
            isCameraBusy = false
            
            if (file.exists() && file.length() > 0) {
                tracker.update(orientation)
                runOnUiThread {
                    val count = 30 - tracker.getRemainingCount()
                    statusText.text = "Captured: $count/30"
                    Log.d("Capture", "SUCCESS: ${file.name}")
                }
            } else {
                Log.w("Capture", "Failed to capture")
            }
        }
    }

private fun initiateStitching() {
    val directory = getExternalFilesDir(null)
    val files = directory?.listFiles { f -> f.extension == "jpg" && f.name.startsWith("capture_") } ?: emptyArray()
    Log.d("Stitching", "Found ${files.size} images to stitch: ${files.joinToString(", ") { it.name }}")

    // 1. Validation
    if (files.size < 2) {
        statusText.text = "Need at least 2 images to stitch! (Found ${files.size})"
        return
    }

    // 2. UI Preparation
    btnFinish.isEnabled = false
    statusText.text = "Stitching ${files.size} images... Please wait."
    progressBar.isIndeterminate = true // Show active processing

    // 3. Offload to Background Thread
    lifecycleScope.launch(Dispatchers.Default) {
        val listToStitch = mutableListOf<Mat>()
        val panorama = Mat()
        
        try {
            // Load images with reduced resolution to prevent OutOfMemory crashes
            files.forEach { file ->
                val src = Imgcodecs.imread(file.absolutePath, Imgcodecs.IMREAD_REDUCED_COLOR_4)
                if (!src.empty()) {
                    listToStitch.add(src)
                }
            }

            // Initialize OpenCV Stitcher
            val stitcher = org.opencv.photo.Stitcher.create(org.opencv.photo.Stitcher.PANORAMA)
            
            
            
            // Perform the actual stitching
            val status = stitcher.stitch(listToStitch, panorama)

            // 4. Handle Result on Main Thread
            withContext(Dispatchers.Main) {
                if (status == 0) {
                    val resultFile = File(directory, "panorama_result.jpg")
                    Imgcodecs.imwrite(resultFile.absolutePath, panorama)
                    statusText.text = "Success! Saved as panorama_result.jpg"
                    
                    // Optional: Show the result (implementation depends on your UI)
                    // showResultDialog(resultFile) 
                } else {
                    statusText.text = "Stitching failed (OpenCV Error: $status)"
                }
                
                // Cleanup UI
                btnFinish.isEnabled = true
                progressBar.isIndeterminate = false
                progressBar.progress = 100
            }

        } catch (e: Exception) {
            withContext(Dispatchers.Main) {
                statusText.text = "Error: ${e.localizedMessage}"
                btnFinish.isEnabled = true
            }
        } finally {
            // 5. CRITICAL: Manual memory management for OpenCV Mats
            listToStitch.forEach { it.release() }
            panorama.release()
        }
    }
}

    private fun allPermissionsGranted() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED &&
        ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED &&
        ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED

    private fun initOpenCV() {
        // Try the older but more common initialization
        if (!org.opencv.android.OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization")
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!")
        }
    }

    private fun getScreenRotation(): Int {
    return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
        // 'display' property is available from API 28 (Pie) onwards
        display?.rotation ?: 0
    } else {
        @Suppress("DEPRECATION")
        windowManager.defaultDisplay.rotation
    }
    }

    private fun calculateYawDiff(current: Float, target: Float): Float {
    var diff = target - current
    while (diff <= -180) diff += 360
    while (diff > 180) diff -= 360
    return diff
}

    private fun showError(message: String) {
        runOnUiThread {
            errorContainer.visibility = View.VISIBLE
            errorText.text = message
            preview.visibility = View.GONE
        }
    }

    private fun retryCamera() {
        errorContainer.visibility = View.GONE
        btnRetry.visibility = View.GONE
        startCaptureLogic()
    }

    private fun resetCapture() {
        tracker.reset()
        lastCaptureYaw = 0f
        lastCapturePitch = 0f
        progressBar.progress = 0
        btnFinish.visibility = View.GONE
        statusText.text = "Capture reset - Rotate to start again"
        captureCountText.text = "0/30"
    }
}

