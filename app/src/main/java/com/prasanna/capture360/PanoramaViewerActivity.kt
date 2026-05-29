package com.prasanna.capture360

import android.content.ContentValues
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import java.io.File

class PanoramaViewerActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "PanoramaViewer"
        private const val MIN_SCALE = 0.5f
        private const val MAX_SCALE = 5.0f
    }

    private lateinit var imageView: ImageView
    private val imageMatrix = Matrix()
    private var scaleFactor = 1.0f
    private var translateX = 0f
    private var translateY = 0f
    private var imagePath: String? = null

    private lateinit var scaleDetector: ScaleGestureDetector
    private lateinit var gestureDetector: GestureDetector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_panorama_viewer)

        imageView = findViewById(R.id.panoramaImage)
        val btnShare = findViewById<Button>(R.id.btnShare)
        val btnBack = findViewById<Button>(R.id.btnBack)

        imagePath = intent.getStringExtra("image_path")

        if (imagePath != null) {
            val options = BitmapFactory.Options().apply {
                inSampleSize = 1
            }
            val bitmap = BitmapFactory.decodeFile(imagePath, options)
            if (bitmap != null) {
                imageView.setImageBitmap(bitmap)
                imageView.scaleType = ImageView.ScaleType.MATRIX
                imageView.post { centerImage() }
            } else {
                Toast.makeText(this, "Failed to load panorama", Toast.LENGTH_SHORT).show()
                Log.e(TAG, "Could not decode: $imagePath")
            }
        }

        scaleDetector = ScaleGestureDetector(this, ScaleListener())
        gestureDetector = GestureDetector(this, PanListener())

        imageView.setOnTouchListener { _, event ->
            scaleDetector.onTouchEvent(event)
            gestureDetector.onTouchEvent(event)
            true
        }

        btnShare.setOnClickListener { shareImage() }
        btnBack.setOnClickListener { finish() }
    }

    private fun centerImage() {
        val drawable = imageView.drawable ?: return
        val dw = drawable.intrinsicWidth.toFloat()
        val dh = drawable.intrinsicHeight.toFloat()
        val vw = imageView.width.toFloat()
        val vh = imageView.height.toFloat()

        val fitScale = (vh / dh).coerceAtMost(vw / dw)
        scaleFactor = fitScale

        translateX = (vw - dw * fitScale) / 2f
        translateY = (vh - dh * fitScale) / 2f

        updateMatrix()
    }

    private fun updateMatrix() {
        imageMatrix.reset()
        imageMatrix.postScale(scaleFactor, scaleFactor)
        imageMatrix.postTranslate(translateX, translateY)
        imageView.imageMatrix = imageMatrix
    }

    private inner class ScaleListener : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            val prevScale = scaleFactor
            scaleFactor *= detector.scaleFactor
            scaleFactor = scaleFactor.coerceIn(MIN_SCALE, MAX_SCALE)

            val focusX = detector.focusX
            val focusY = detector.focusY
            val scaleChange = scaleFactor / prevScale
            translateX = focusX - (focusX - translateX) * scaleChange
            translateY = focusY - (focusY - translateY) * scaleChange

            updateMatrix()
            return true
        }
    }

    private inner class PanListener : GestureDetector.SimpleOnGestureListener() {
        override fun onScroll(
            e1: MotionEvent?,
            e2: MotionEvent,
            distanceX: Float,
            distanceY: Float
        ): Boolean {
            translateX -= distanceX
            translateY -= distanceY
            updateMatrix()
            return true
        }

        override fun onDoubleTap(e: MotionEvent): Boolean {
            centerImage()
            return true
        }
    }

    private fun shareImage() {
        val path = imagePath ?: return
        val file = File(path)
        if (!file.exists()) {
            Toast.makeText(this, "Panorama file not found", Toast.LENGTH_SHORT).show()
            return
        }
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val contentValues = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, "panorama_shared_${System.currentTimeMillis()}.jpg")
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Capture360")
                }
                val uri = contentResolver.insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues
                )
                if (uri != null) {
                    contentResolver.openOutputStream(uri)?.use { out ->
                        file.inputStream().use { it.copyTo(out) }
                    }
                    val shareIntent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
                        type = "image/jpeg"
                        putExtra(android.content.Intent.EXTRA_STREAM, uri)
                        addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                    startActivity(android.content.Intent.createChooser(shareIntent, "Share Panorama"))
                }
            } else {
                Toast.makeText(this, "Sharing requires Android 10+", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Share failed", e)
            Toast.makeText(this, "Share failed: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
        }
    }
}
