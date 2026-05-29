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
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
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
import com.prasanna.capture360.sensors.Orientation
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.prasanna.capture360.stitching.PanoramaStitcher
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import java.io.File

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val PERMISSION_REQUEST_CODE = 1001
        private const val TOTAL_BINS = 30
        private const val MIN_IMAGES_FOR_STITCH = 2
        private const val MIN_IMAGES_FOR_FINISH = 8
        private const val CAPTURE_ANGULAR_THRESHOLD = 8f
        private const val CAPTURE_LOOP_INTERVAL_MS = 100L
    }

    private var lastCaptureYaw = 0f
    private var lastCapturePitch = 0f
    private var isCameraBusy = false
    private var capturedImageCount = 0
    private lateinit var camera: CameraController
    private lateinit var fusion: SensorFusionManager
    private lateinit var tracker: AngularCoverageTracker
    private val gate = FrameGate()

    private lateinit var preview: PreviewView
    private lateinit var overlay: GuidanceOverlayView
    private lateinit var progressBar: ProgressBar
    private lateinit var statusText: TextView
    private lateinit var btnFinish: Button
    private lateinit var captureFlashView: View

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initOpenCV()
        setContentView(R.layout.activity_main)

        preview = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        progressBar = findViewById(R.id.progressBar)
        statusText = findViewById(R.id.statusText)
        btnFinish = findViewById(R.id.btnFinish)
        captureFlashView = findViewById(R.id.captureFlash)

        btnFinish.setOnClickListener { initiateStitching() }

        if (allPermissionsGranted()) {
            startCaptureLogic()
        } else {
            requestRequiredPermissions()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            // Log each permission result for debugging
            permissions.forEachIndexed { index, perm ->
                val result = if (index < grantResults.size) grantResults[index] else -1
                Log.d(TAG, "Permission: $perm -> ${if (result == PackageManager.PERMISSION_GRANTED) "GRANTED" else "DENIED"}")
            }

            // Only require camera permission to proceed; storage is checked separately
            val cameraIndex = permissions.indexOf(Manifest.permission.CAMERA)
            val cameraGranted = cameraIndex >= 0 &&
                    cameraIndex < grantResults.size &&
                    grantResults[cameraIndex] == PackageManager.PERMISSION_GRANTED

            if (cameraGranted) {
                Log.d(TAG, "Camera permission granted, starting capture")
                startCaptureLogic()
            } else {
                Log.w(TAG, "Camera permission denied")
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
                statusText.text = "Camera permission required"
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::fusion.isInitialized) fusion.stop()
        if (::camera.isInitialized) camera.shutdown()
    }

    private fun requestRequiredPermissions() {
        val permissions = mutableListOf(Manifest.permission.CAMERA)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.READ_MEDIA_IMAGES)
        } else if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
            permissions.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            permissions.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
        Log.d(TAG, "Requesting permissions: $permissions")
        ActivityCompat.requestPermissions(this, permissions.toTypedArray(), PERMISSION_REQUEST_CODE)
    }

    private fun allPermissionsGranted(): Boolean {
        val cameraGranted = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
        Log.d(TAG, "allPermissionsGranted: camera=$cameraGranted, SDK=${Build.VERSION.SDK_INT}")
        // Camera is the only hard requirement; storage uses scoped access on modern Android
        return cameraGranted
    }

    // ──────────────────────────────────────────────────────────────
    // Capture Logic
    // ──────────────────────────────────────────────────────────────

    private fun startCaptureLogic() {
        Log.d(TAG, "startCaptureLogic called")

        try {
            camera = CameraController(this, this)
            fusion = SensorFusionManager(this)
            tracker = AngularCoverageTracker(10, 3)
            capturedImageCount = 0

            clearOldCaptures()
            camera.startCamera(preview)
            fusion.start()

            statusText.text = "Ready - Rotate slowly to capture"

            lifecycleScope.launch {
                while (isActive) {
                    val orientation = fusion.getOrientation()
                    val curYaw = Math.toDegrees(orientation.yaw.toDouble()).toFloat()
                    val curPitch = Math.toDegrees(orientation.pitch.toDouble()).toFloat()
                    val rotation = getScreenRotation()

                    val yawDiff = calculateYawDiff(curYaw, lastCaptureYaw)
                    val pitchDiff = curPitch - lastCapturePitch
                    val angularDistance = sqrt(
                        yawDiff.toDouble().pow(2.0) + pitchDiff.toDouble().pow(2.0)
                    ).toFloat()

                    val remaining = tracker.getRemainingCount()
                    val capturedCount = TOTAL_BINS - remaining
                    val guidance = tracker.getTargetDirection(curYaw, curPitch, rotation)

                    runOnUiThread {
                        overlay.updateDisplay(
                            tracker.getGrid(),
                            "$guidance\n($capturedImageCount images captured)"
                        )
                        progressBar.progress = ((capturedCount.toFloat() / TOTAL_BINS) * 100).toInt()
                        if (capturedImageCount >= MIN_IMAGES_FOR_FINISH) {
                            btnFinish.visibility = View.VISIBLE
                        }
                    }

                    if (!isCameraBusy &&
                        remaining > 0 &&
                        gate.shouldCapture(orientation) &&
                        !tracker.isCurrentAreaCovered(orientation) &&
                        angularDistance > CAPTURE_ANGULAR_THRESHOLD
                    ) {
                        triggerImageCapture(curYaw, curPitch, orientation)
                    }

                    delay(CAPTURE_LOOP_INTERVAL_MS)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "startCaptureLogic failed", e)
            statusText.text = "Error: ${e.localizedMessage}"
        }
    }

    private fun triggerImageCapture(yaw: Float, pitch: Float, orientation: Orientation) {
        isCameraBusy = true
        val dir = getExternalFilesDir(null) ?: run {
            Log.e(TAG, "External files directory is null")
            isCameraBusy = false
            return
        }

        val tempFile = File(dir, "capture_${System.currentTimeMillis()}.jpg")
        Log.d(TAG, "Capturing to: ${tempFile.absolutePath}")

        camera.captureFrame(tempFile) {
            if (tempFile.exists() && tempFile.length() > 0) {
                tracker.update(orientation)
                lastCaptureYaw = yaw
                lastCapturePitch = pitch
                capturedImageCount++

                runOnUiThread {
                    statusText.text = "Captured: $capturedImageCount images"
                    Log.d(TAG, "Capture SUCCESS: ${tempFile.name}")
                    showCaptureFlash()
                    vibrateOnCapture()
                }
            } else {
                Log.w(TAG, "Capture file empty or missing: ${tempFile.absolutePath}")
            }
            isCameraBusy = false
        }
    }

    private fun showCaptureFlash() {
        captureFlashView.alpha = 0.6f
        captureFlashView.visibility = View.VISIBLE
        captureFlashView.animate()
            .alpha(0f)
            .setDuration(200)
            .withEndAction { captureFlashView.visibility = View.GONE }
            .start()
    }

    private fun vibrateOnCapture() {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val vibratorManager = getSystemService(VIBRATOR_MANAGER_SERVICE) as VibratorManager
                vibratorManager.defaultVibrator.vibrate(
                    VibrationEffect.createOneShot(50, VibrationEffect.DEFAULT_AMPLITUDE)
                )
            } else {
                @Suppress("DEPRECATION")
                val vibrator = getSystemService(VIBRATOR_SERVICE) as Vibrator
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    vibrator.vibrate(
                        VibrationEffect.createOneShot(50, VibrationEffect.DEFAULT_AMPLITUDE)
                    )
                } else {
                    @Suppress("DEPRECATION")
                    vibrator.vibrate(50)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Vibration not available", e)
        }
    }

    private fun clearOldCaptures() {
        val cacheDir = getExternalFilesDir(null)
        cacheDir?.listFiles()?.forEach {
            if (it.name.startsWith("capture_")) it.delete()
        }
    }

    // ──────────────────────────────────────────────────────────────
    // Stitching
    // ──────────────────────────────────────────────────────────────

    private fun initiateStitching() {
        if (!openCVInitialized) {
            statusText.text = "OpenCV not available - cannot stitch"
            Log.e(TAG, "Stitching attempted but OpenCV not initialized")
            return
        }

        val directory = getExternalFilesDir(null)
        if (directory == null) {
            statusText.text = "Error: Cannot access storage directory"
            return
        }
        directory.mkdirs()

        val files = directory.listFiles { f ->
            f.extension == "jpg" && f.name.startsWith("capture_")
        }?.sortedBy { it.lastModified() }?.toTypedArray() ?: emptyArray()

        Log.d(TAG, "Found ${files.size} images to stitch")

        if (files.size < MIN_IMAGES_FOR_STITCH) {
            statusText.text = "Need at least $MIN_IMAGES_FOR_STITCH images (found ${files.size})"
            return
        }

        btnFinish.isEnabled = false
        statusText.text = "Stitching ${files.size} images... Please wait."
        progressBar.isIndeterminate = true

        lifecycleScope.launch(Dispatchers.Default) {
            val imageMats = mutableListOf<Mat>()

            try {
                files.forEach { file ->
                    val src = Imgcodecs.imread(file.absolutePath, Imgcodecs.IMREAD_REDUCED_COLOR_4)
                    if (!src.empty()) {
                        imageMats.add(src)
                        Log.d(TAG, "Loaded: ${file.name} (${src.cols()}x${src.rows()})")
                    } else {
                        Log.w(TAG, "Failed to load: ${file.name}")
                    }
                }

                if (imageMats.size < MIN_IMAGES_FOR_STITCH) {
                    withContext(Dispatchers.Main) {
                        statusText.text = "Could not load enough images for stitching"
                        btnFinish.isEnabled = true
                        progressBar.isIndeterminate = false
                    }
                    return@launch
                }

                val stitcher = PanoramaStitcher()
                val result = stitcher.stitch(imageMats)

                withContext(Dispatchers.Main) {
                    when (result) {
                        is PanoramaStitcher.StitchResult.Success -> {
                            val resultFile = File(directory, "panorama_result.jpg")
                            Imgcodecs.imwrite(resultFile.absolutePath, result.panorama)
                            result.panorama.release()

                            val gallerySuccess = saveImageToGallery(
                                resultFile,
                                "panorama_${System.currentTimeMillis()}.jpg"
                            )

                            statusText.text = if (gallerySuccess) {
                                "Panorama saved to gallery!"
                            } else {
                                "Panorama saved locally"
                            }

                            Log.d(TAG, "Panorama saved: ${resultFile.absolutePath}")
                            openPanoramaViewer(resultFile.absolutePath)
                        }
                        is PanoramaStitcher.StitchResult.Error -> {
                            statusText.text = "Stitching failed: ${result.message}"
                            Log.e(TAG, "Stitch error: ${result.message}")
                        }
                    }

                    btnFinish.isEnabled = true
                    progressBar.isIndeterminate = false
                    progressBar.progress = 100
                }
            } catch (e: Exception) {
                Log.e(TAG, "Stitching exception", e)
                withContext(Dispatchers.Main) {
                    statusText.text = "Error: ${e.localizedMessage}"
                    btnFinish.isEnabled = true
                    progressBar.isIndeterminate = false
                }
            } finally {
                imageMats.forEach { it.release() }
            }
        }
    }

    private fun openPanoramaViewer(imagePath: String) {
        val intent = Intent(this, PanoramaViewerActivity::class.java)
        intent.putExtra("image_path", imagePath)
        startActivity(intent)
    }

    // ──────────────────────────────────────────────────────────────
    // Gallery Save
    // ──────────────────────────────────────────────────────────────

    private fun saveImageToGallery(sourceFile: File, displayName: String): Boolean {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val contentValues = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Capture360")
                }
                val uri = contentResolver.insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues
                )
                if (uri != null) {
                    contentResolver.openOutputStream(uri)?.use { outputStream ->
                        sourceFile.inputStream().use { inputStream ->
                            inputStream.copyTo(outputStream)
                        }
                    }
                    true
                } else {
                    false
                }
            } else {
                @Suppress("DEPRECATION")
                val destDir = File(
                    Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                    "Capture360"
                )
                destDir.mkdirs()
                val destFile = File(destDir, displayName)
                sourceFile.copyTo(destFile, overwrite = true)
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.insertImage(
                    contentResolver, destFile.absolutePath, destFile.name, null
                )
                true
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save to gallery", e)
            false
        }
    }

    // ──────────────────────────────────────────────────────────────
    // Utility
    // ──────────────────────────────────────────────────────────────

    private var openCVInitialized = false

    private fun initOpenCV() {
        try {
            openCVInitialized = org.opencv.android.OpenCVLoader.initDebug()
            if (!openCVInitialized) {
                Log.e(TAG, "OpenCV init failed")
            } else {
                Log.d(TAG, "OpenCV initialized successfully")
            }
        } catch (e: Exception) {
            Log.e(TAG, "OpenCV init exception", e)
            openCVInitialized = false
        }
    }

    private fun getScreenRotation(): Int {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
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
