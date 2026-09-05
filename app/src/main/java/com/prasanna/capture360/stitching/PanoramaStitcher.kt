package com.prasanna.capture360.stitching

import com.prasanna.capture360.sensors.Orientation
import org.opencv.core.Mat
import java.io.File

interface PanoramaStitcher {
    fun stitch(frameFiles: List<File>, poses: List<Orientation>): Result<Mat>
}

enum class StitchingMode { OPENCV, AI }
