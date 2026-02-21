package com.prasanna.capture360

import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.ImageView
import android.widget.HorizontalScrollView
import androidx.appcompat.app.AppCompatActivity
import java.io.File

class ViewerActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val path = intent.getStringExtra("image_path")

        // HorizontalScrollView allows the user to pan across the wide panorama
        val imageView = ImageView(this).apply {
            adjustViewBounds = true
            if (path != null) {
                val bitmap = BitmapFactory.decodeFile(path)
                setImageBitmap(bitmap)
            }
        }

        val scrollView = HorizontalScrollView(this).apply {
            layoutParams = android.view.ViewGroup.LayoutParams(
                android.view.ViewGroup.LayoutParams.MATCH_PARENT,
                android.view.ViewGroup.LayoutParams.MATCH_PARENT
            )
            addView(imageView)
        }

        setContentView(scrollView)
    }
}

