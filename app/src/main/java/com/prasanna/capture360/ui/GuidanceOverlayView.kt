package com.prasanna.capture360.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

import android.graphics.*
import android.util.AttributeSet

class GuidanceOverlayView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    private var coverage: Array<BooleanArray>? = null
    private var instruction: String = "Wait..."

    private val fillPaint = Paint().apply { style = Paint.Style.FILL }
    private val strokePaint = Paint().apply {
        style = Paint.Style.STROKE
        color = Color.WHITE
        strokeWidth = 4f
    }

    private val textPaint = Paint().apply {
        color = Color.YELLOW
        textSize = 70f
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT_BOLD
        setShadowLayer(10f, 0f, 0f, Color.BLACK)
    }

    fun updateDisplay(grid: Array<BooleanArray>, guidance: String) {
        this.coverage = grid
        this.instruction = guidance
        postInvalidate() // Thread-safe redraw
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val cx = width / 2f
        val cy = height / 2f

        // 1. Draw Targeting Reticle (Center)
        canvas.drawCircle(cx, cy, 80f, strokePaint)
        canvas.drawLine(cx - 120, cy, cx + 120, cy, strokePaint)
        canvas.drawLine(cx, cy - 120, cx, cy + 120, strokePaint)

        // 2. Draw Instruction Text
        canvas.drawText(instruction, cx, cy - 200f, textPaint)

        // 3. Draw Mini-Map (Top Right)
        drawMiniMap(canvas)
    }

    private fun drawMiniMap(canvas: Canvas) {
        coverage?.let { grid ->
            val mapSize = 240f
            val cellW = mapSize / grid.size
            val cellH = (mapSize / 2) / grid[0].size
            val offsetX = width - mapSize - 40f
            val offsetY = 150f

            for (i in grid.indices) {
                for (j in grid[i].indices) {
                    fillPaint.color = if (grid[i][j]) Color.GREEN else Color.RED
                    fillPaint.alpha = 160
                    canvas.drawRect(
                        offsetX + (i * cellW), offsetY + (j * cellH),
                        offsetX + ((i + 1) * cellW), offsetY + ((j + 1) * cellH),
                        fillPaint
                    )
                }
            }
            canvas.drawRect(offsetX, offsetY, offsetX + mapSize, offsetY + (mapSize / 2), strokePaint)
        }
    }
}

