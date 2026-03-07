package com.prasanna.capture360

import kotlin.math.sqrt
import kotlin.math.pow
import kotlin.math.abs
import android.Manifest
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
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
            requestRequiredPermissions()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1001) {
            if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                Log.d("Permissions", "All permissions granted in onRequestPermissionsResult")
                Toast.makeText(this, "Permissions granted!", Toast.LENGTH_SHORT).show()
                startCaptureLogic()
            } else {
                Log.w("Permissions", "Some permissions denied")
                Toast.makeText(this, "Permissions denied - app needs Camera + Storage", Toast.LENGTH_LONG).show()
                statusText.text = "Permissions required: Camera + Storage"
            }
        }
    }
    private fun getPublicPicturesDir(): File? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            // Android 11+: Use app-specific Pictures directory (no READ_EXTERNAL required)
            File(getExternalFilesDir(null), "Pictures").apply { mkdirs() }
        } else {
            // Android 10 and below: Use public Pictures directory
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).apply { mkdirs() }
        }
    }

    /**
     * Saves image to gallery-accessible location and notifies MediaStore
     */
    private fun saveImageToGallery(sourceFile: File, displayName: String): Boolean {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                // Android 10+: Use MediaStore API
                val contentValues = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Capture360")
                }

                val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
                if (uri != null) {
                    contentResolver.openOutputStream(uri)?.use { outputStream ->
                        sourceFile.inputStream().use { inputStream ->
                            inputStream.copyTo(outputStream)
                        }
                    }
                    Log.d("Gallery", "Image saved to gallery via MediaStore: $displayName")
                    true
                } else {
                    Log.w("Gallery", "Failed to create MediaStore entry")
                    false
                }
            } else {
                // Android 9 and below: Direct file copy + MediaStore scan
                val destFile = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "Capture360/$displayName")
                destFile.parentFile?.mkdirs()
                sourceFile.copyTo(destFile, overwrite = true)
                
                // Notify MediaStore to scan the file
                MediaStore.Images.Media.insertImage(contentResolver, destFile.absolutePath, destFile.name, null)
                Log.d("Gallery", "Image saved to gallery (Android 9): $displayName")
                true
            }
        } catch (e: Exception) {
            Log.e("Gallery", "Failed to save image to gallery", e)
            false
        }
    }

    private fun requestRequiredPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.CAMERA,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        )
        
        // Android 13+ requires READ_MEDIA_IMAGES
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.READ_MEDIA_IMAGES)
        } else {
            permissions.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        
        // Android 13+ requires POST_NOTIFICATIONS
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        
        ActivityCompat.requestPermissions(this, permissions.toTypedArray(), 1001)
    }

    private fun allPermissionsGranted(): Boolean {
        val cameraGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        val writeStorageGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
        
        val readStorageGranted = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED
        } else {
            ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
        }
        
        return cameraGranted && writeStorageGranted && readStorageGranted
    }

private fun startCaptureLogic() {
    Log.d("MainActivity", "startCaptureLogic called")
    Toast.makeText(this, "Starting camera...", Toast.LENGTH_SHORT).show()
    
    try {
        // 1. Initialize Components
        Log.d("MainActivity", "Initializing components...")
        camera = CameraController(this, this)
        fusion = SensorFusionManager(this)
        tracker = AngularCoverageTracker(10, 3)
        Toast.makeText(this, "Components initialized", Toast.LENGTH_SHORT).show()

        // 2. Clear out old images
        val cacheDir = getExternalFilesDir(null)
        Log.d("MainActivity", "Cache dir: ${cacheDir?.absolutePath}")
        cacheDir?.listFiles()?.forEach { 
            if (it.name.startsWith("capture_")) {
                it.delete()
            }
        }

        // 3. Start Camera and bind to preview
        Log.d("MainActivity", "Starting camera preview... preview=$preview")
        Toast.makeText(this, "Starting camera preview...", Toast.LENGTH_SHORT).show()
        camera.startCamera(preview)
        Log.d("MainActivity", "Camera started successfully")
        Toast.makeText(this, "Camera preview active!", Toast.LENGTH_SHORT).show()
        
        // 4. Start Sensors
        Log.d("MainActivity", "Starting sensors...")
        Toast.makeText(this, "Starting sensors...", Toast.LENGTH_SHORT).show()
        fusion.start()
        Log.d("MainActivity", "Sensors started")
        
        Toast.makeText(this, "Ready to capture!", Toast.LENGTH_SHORT).show()
        statusText.text = "Ready - Rotate to capture"

        // 5. The Main Control Loop
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
                    angularDistance > 8f) {  // Reduced from 25° to 8° - much more practical

                    triggerImageCapture(curYaw, curPitch, orientation)
                }

                delay(100) // 10Hz update rate is plenty for UI and sensors
            }
        }
        
    } catch (e: Exception) {
        Log.e("MainActivity", "startCaptureLogic FAILED with exception", e)
        Toast.makeText(this, "Error: ${e.message}", Toast.LENGTH_LONG).show()
        statusText.text = "Error: ${e.localizedMessage}"
    }
}
        Log.e("Capture", "External files directory is null")
        isCameraBusy = false
        return
    }
    
    val tempFile = File(dir, "capture_${System.currentTimeMillis()}.jpg")
    Log.d("Capture", "Capturing to: ${tempFile.absolutePath}")

    camera.captureFrame(tempFile) {
        if (tempFile.exists() && tempFile.length() > 0) {
            tracker.update(orientation)
            lastCaptureYaw = yaw
            lastCapturePitch = pitch
            
            runOnUiThread {
                val count = 30 - tracker.getRemainingCount()
                statusText.text = "Captured: $count/30"
                Log.d("Capture", "SUCCESS: ${tempFile.name}")
            }
        } else {
            Log.w("Capture", "Failed to capture file: ${tempFile.absolutePath}")
        }
        isCameraBusy = false 
    }
}

private fun initiateStitching() {
    val directory = getExternalFilesDir(null)
    
    if (directory == null) {
        statusText.text = "Error: Cannot access cache directory"
        Log.e("Stitching", "Cache directory is null")
        return
    }
    
    if (!directory.exists()) {
        directory.mkdirs()
        Log.d("Stitching", "Created cache directory: ${directory.absolutePath}")
    }
    
    val files = directory.listFiles { f -> f.extension == "jpg" && f.name.startsWith("capture_") }?.sortedBy { it.lastModified() }?.toTypedArray() ?: emptyArray()
    Log.d("Stitching", "Cache directory: ${directory.absolutePath}")
    Log.d("Stitching", "Found ${files.size} images to stitch: ${files.joinToString(", ") { it.name }}")

    // 1. Validation
    if (files.size < 2) {
        statusText.text = "Need at least 2 images to stitch! (Found ${files.size})"
        Log.w("Stitching", "Not enough images: ${files.size}")
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
                Log.d("Stitching", "Loading image: ${file.absolutePath}")
                val src = Imgcodecs.imread(file.absolutePath, Imgcodecs.IMREAD_REDUCED_COLOR_4)
                if (!src.empty()) {
                    Log.d("Stitching", "Successfully loaded: ${file.name}")
                    listToStitch.add(src)
                } else {
                    Log.w("Stitching", "Failed to load: ${file.name}")
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
                    
                    // Save result to gallery
                    val gallerySuccess = saveImageToGallery(resultFile, "panorama_${System.currentTimeMillis()}.jpg")
                    
                    statusText.text = "Success! Panorama saved to gallery" + if (gallerySuccess) " ✓" else ""
                    Log.d("Stitching", "Panorama saved: ${resultFile.absolutePath}") 
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

