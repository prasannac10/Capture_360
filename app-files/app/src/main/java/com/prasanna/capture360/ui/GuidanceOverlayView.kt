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
        strokeWidth = 3f
    }

    private val instructionPaint = Paint().apply {
        color = Color.YELLOW
        textSize = 64f
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT_BOLD
        setShadowLayer(8f, 0f, 0f, Color.BLACK)
    }

    private val smallTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 36f
        textAlign = Paint.Align.CENTER
        setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }

    private val cellBorderPaint = Paint().apply {
        style = Paint.Style.STROKE
        color = Color.GRAY
        strokeWidth = 2f
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

        // Draw semi-transparent darkening overlay
        canvas.drawColor(Color.argb(20, 0, 0, 0))

        // 1. Draw Targeting Reticle (Center) - Enhanced design
        val reticleRadius = 100f
        val reticleColor = Color.rgb(0, 255, 150)
        
        strokePaint.strokeWidth = 4f
        strokePaint.color = reticleColor
        
        // Outer circle
        canvas.drawCircle(cx, cy, reticleRadius, strokePaint)
        
        // Inner circle
        canvas.drawCircle(cx, cy, reticleRadius / 2, strokePaint)
        
        // Crosshair lines
        val lineLength = 140f
        canvas.drawLine(cx - lineLength, cy, cx + lineLength, cy, strokePaint)
        canvas.drawLine(cx, cy - lineLength, cx, cy + lineLength, strokePaint)
        
        // Corner brackets
        val bracketSize = 50f
        canvas.drawLine(cx - reticleRadius, cy - reticleRadius, cx - reticleRadius + bracketSize, cy - reticleRadius, strokePaint)
        canvas.drawLine(cx - reticleRadius, cy - reticleRadius, cx - reticleRadius, cy - reticleRadius + bracketSize, strokePaint)
        canvas.drawLine(cx + reticleRadius, cy - reticleRadius, cx + reticleRadius - bracketSize, cy - reticleRadius, strokePaint)
        canvas.drawLine(cx + reticleRadius, cy - reticleRadius, cx + reticleRadius, cy - reticleRadius + bracketSize, strokePaint)
        canvas.drawLine(cx - reticleRadius, cy + reticleRadius, cx - reticleRadius + bracketSize, cy + reticleRadius, strokePaint)
        canvas.drawLine(cx - reticleRadius, cy + reticleRadius, cx - reticleRadius, cy + reticleRadius - bracketSize, strokePaint)
        canvas.drawLine(cx + reticleRadius, cy + reticleRadius, cx + reticleRadius - bracketSize, cy + reticleRadius, strokePaint)
        canvas.drawLine(cx + reticleRadius, cy + reticleRadius, cx + reticleRadius, cy + reticleRadius - bracketSize, strokePaint)

        // 2. Draw Instruction Text with background
        val textY = cy - 240f
        val textBGHeight = 100f
        val textBGPadding = 20f
        
        val instructionLines = instruction.split("\n")
        if (instructionLines.isNotEmpty()) {
            // Draw semi-transparent background for instruction
            fillPaint.color = Color.argb(100, 0, 0, 0)
            canvas.drawRoundRect(
                cx - 250f, textY - 50f,
                cx + 250f, textY + 50f,
                15f, 15f,
                fillPaint
            )
            
            // Draw instruction text
            instructionPaint.color = Color.YELLOW
            canvas.drawText(instructionLines[0], cx, textY, instructionPaint)
            
            if (instructionLines.size > 1) {
                canvas.drawText(instructionLines[1], cx, textY + 60f, smallTextPaint)
            }
        }

        // 3. Draw Mini-Map (Top Right Corner)
        drawMiniMap(canvas)
    }

    private fun drawMiniMap(canvas: Canvas) {
        coverage?.let { grid ->
            val mapSize = 200f
            val cellW = mapSize / grid.size
            val cellH = (mapSize / 2) / grid[0].size
            val offsetX = width - mapSize - 40f
            val offsetY = 120f

            // Draw map background
            fillPaint.color = Color.argb(150, 0, 0, 0)
            canvas.drawRoundRect(
                offsetX - 10f, offsetY - 10f,
                offsetX + mapSize + 10f, offsetY + (mapSize / 2) + 10f,
                10f, 10f,
                fillPaint
            )

            // Draw grid cells
            for (i in grid.indices) {
                for (j in grid[i].indices) {
                    val x1 = offsetX + (i * cellW)
                    val y1 = offsetY + (j * cellH)
                    val x2 = x1 + cellW
                    val y2 = y1 + cellH

                    // Draw cell color
                    fillPaint.color = if (grid[i][j]) Color.rgb(76, 175, 80) else Color.rgb(244, 67, 54)
                    fillPaint.alpha = 200
                    canvas.drawRect(x1, y1, x2, y2, fillPaint)
                    
                    // Draw cell border
                    canvas.drawRect(x1, y1, x2, y2, cellBorderPaint)
                }
            }

            // Draw outer border
            strokePaint.strokeWidth = 3f
            strokePaint.color = Color.WHITE
            canvas.drawRoundRect(
                offsetX - 10f, offsetY - 10f,
                offsetX + mapSize + 10f, offsetY + (mapSize / 2) + 10f,
                10f, 10f,
                strokePaint
            )

            // Draw "Coverage" label
            smallTextPaint.textSize = 28f
            smallTextPaint.color = Color.WHITE
            canvas.drawText("Coverage", offsetX + mapSize / 2, offsetY - 20f, smallTextPaint)
        }
    }
}

