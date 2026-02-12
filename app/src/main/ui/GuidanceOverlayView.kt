package com.yourcompany.capture360.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class GuidanceOverlayView(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var coverage: Array<BooleanArray>? = null

    fun setCoverage(grid: Array<BooleanArray>) {
        coverage = grid
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        coverage?.let { grid ->
            val paint = Paint()

            val cellWidth = width / grid.size.toFloat()
            val cellHeight = height / grid[0].size.toFloat()

            for (i in grid.indices) {
                for (j in grid[0].indices) {

                    paint.color = if (grid[i][j])
                        Color.GREEN else Color.RED

                    canvas.drawRect(
                        i * cellWidth,
                        j * cellHeight,
                        (i + 1) * cellWidth,
                        (j + 1) * cellHeight,
                        paint
                    )
                }
            }
        }
    }
}
