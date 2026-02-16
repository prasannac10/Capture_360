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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initOpenCV()
        setContentView(R.layout.activity_main)

        preview = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        progressBar = findViewById(R.id.progressBar)
        statusText = findViewById(R.id.statusText)
        btnFinish = findViewById(R.id.btnFinish)

        btnFinish.setOnClickListener { initiateStitching() }

        if (allPermissionsGranted()) {
            startCaptureLogic()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1001)
        }
    }
private fun startCaptureLogic() {
    // 1. Initialize Components
    camera = CameraController(this, this)
    fusion = SensorFusionManager(this)
    tracker = AngularCoverageTracker(10, 3) // Explicitly set 10x3 grid for 30 points

    // 2. Clear out old images so the stitcher doesn't use old data
    getExternalFilesDir(null)?.listFiles()?.forEach { 
        if (it.name.startsWith("pano_")) it.delete() 
    }

    // 3. Start Hardware
    camera.startCamera(preview)
    fusion.start()

    // 4. The Main Control Loop
    lifecycleScope.launch {
        while (isActive) {
            val orientation = fusion.getOrientation()
            
            // Convert radians to degrees for easier math
            val curYaw = Math.toDegrees(orientation.yaw.toDouble()).toFloat()
            val curPitch = Math.toDegrees(orientation.pitch.toDouble()).toFloat()
            val rotation = getScreenRotation()

            // Calculate distance from the last capture point
            val yawDiff = calculateYawDiff(curYaw, lastCaptureYaw)
            val pitchDiff = curPitch - lastCapturePitch
            val angularDistance = sqrt(yawDiff.toDouble().pow(2.0) + pitchDiff.toDouble().pow(2.0)).toFloat()

            // Update UI Guidance
            val remaining = tracker.getRemainingCount()
            val capturedCount = 30 - remaining
            val guidance = tracker.getTargetDirection(curYaw, curPitch, rotation)

            runOnUiThread {
                overlay.updateDisplay(tracker.getGrid(), "$guidance\n($capturedCount/30 Captured)")
                progressBar.progress = ((capturedCount.toFloat() / 30f) * 100).toInt()
                
                // Show finish button if we have enough coverage (e.g., 20+ images)
                if (capturedCount >= 20) btnFinish.visibility = View.VISIBLE
            }

            // 5. Intelligent Capture Trigger
            if (!isCameraBusy && 
                remaining > 0 && 
                gate.shouldCapture(orientation) && 
                !tracker.isCurrentAreaCovered(orientation) && 
                angularDistance > 25f) {

                triggerImageCapture(curYaw, curPitch, orientation)
            }

            delay(100) // 10Hz update rate is plenty for UI and sensors
        }
    }
}

private fun triggerImageCapture(yaw: Float, pitch: Float, orientation: Orientation) {
    isCameraBusy = true
    val file = File(getExternalFilesDir(null), "pano_${System.currentTimeMillis()}.jpg")

    camera.captureFrame(file) {
        if (file.exists() && file.length() > 0) {
            tracker.update(orientation)
            lastCaptureYaw = yaw
            lastCapturePitch = pitch
            
            runOnUiThread {
                statusText.text = "Point captured!"
            }
        }
        isCameraBusy = false 
    }
}

   private fun initiateStitching() {
    val directory = getExternalFilesDir(null)
    val files = directory?.listFiles { f -> f.extension == "jpg" && f.name.startsWith("img_") } ?: emptyArray()

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

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

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
}

