package com.prasanna.capture360.stitching

import com.prasanna.capture360.sensors.Orientation
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.photo.Stitcher
import java.io.File

class ClassicalStitcher : PanoramaStitcher {
    override fun stitch(frameFiles: List<File>, poses: List<Orientation>, cameraProfile: CameraProfile): Result<Mat> {
        val mats=mutableListOf<Mat>()
        return try { frameFiles.forEach{file->val src=Imgcodecs.imread(file.absolutePath,Imgcodecs.IMREAD_REDUCED_COLOR_4);if(!src.empty())mats+=src else throw IllegalArgumentException("Unable to decode ${file.name}")}; val panorama=Mat(); val status=Stitcher.create(Stitcher.PANORAMA).stitch(mats,panorama); if(status!=0){panorama.release();Result.failure(IllegalStateException("OpenCV stitching failed (status=$status)"))}else Result.success(panorama) } catch(t:Throwable){Result.failure(t)} finally{mats.forEach(Mat::release)}
}
