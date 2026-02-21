package com.prasanna.capture360.camera

import android.content.Context
import android.util.Log
import android.view.Surface
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Manages CameraX operations: Preview and Image Capture.
 */
class CameraController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {

    private var imageCapture: ImageCapture? = null
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    companion object {
        private const val TAG = "CameraController"
    }

    /**
     * Initializes and starts camera preview with safety checks for display rotation.
     */
    fun startCamera(previewView: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()

                // SAFETY: If the view isn't fully attached, previewView.display is null.
                // We default to Surface.ROTATION_0 to prevent a crash.
                val rotation = try {
                    previewView.display?.rotation ?: Surface.ROTATION_0
                } catch (e: Exception) {
                    Surface.ROTATION_0
                }

                // 1. Build Preview Use Case
                val preview = Preview.Builder()
                    .setTargetRotation(rotation)
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                // 2. Build Image Capture Use Case
                imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .setJpegQuality(95)
                    .setTargetRotation(rotation)
                    .build()

                // 3. Select Back Camera
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                // 4. Bind Use Cases to Lifecycle
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageCapture
                )

                Log.d(TAG, "CameraX bound successfully for $lifecycleOwner")

            } catch (exc: Exception) {
                Log.e(TAG, "CameraX initialization failed", exc)
            }

        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Captures a single frame and saves to provided file.
     * Operates on a background executor to keep UI responsive.
     */
    fun captureFrame(
        file: File,
        onSaved: () -> Unit
    ) {
        // Ensure imageCapture is initialized
        val capture = imageCapture ?: run {
            Log.e(TAG, "Capture failed: imageCapture is null")
            return
        }

        val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()

        capture.takePicture(
            outputOptions,
            cameraExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    Log.d(TAG, "Image saved: ${file.absolutePath}")
                    onSaved()
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Image capture error: ${exception.message}", exception)
                }
            }
        )
    }

    /**
     * Safely shuts down the background executor.
     * Call this in Activity.onDestroy() or similar.
     */
    fun shutdown() {
        cameraExecutor.shutdown()
        Log.d(TAG, "Camera executor shut down")
    }
}

