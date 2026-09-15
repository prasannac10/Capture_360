package com.prasanna.capture360.stitching

import com.prasanna.capture360.sensors.Orientation
import org.opencv.core.Mat
import java.io.File

enum class CameraProjection { FISHEYE_180, PINHOLE }
data class CameraProfile(val projection: CameraProjection, val fxNorm: Float=1.0f, val fyNorm: Float=1.0f, val cxNorm: Float=0.0f, val cyNorm: Float=0.0f)

interface PanoramaStitcher {
    fun stitch(frameFiles: List<File>, poses: List<Orientation>, cameraProfile: CameraProfile = CameraProfile(CameraProjection.FISHEYE_180)): Result<Mat>
}

enum class StitchingMode { OPENCV, AI }
