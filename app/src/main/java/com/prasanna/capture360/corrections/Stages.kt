package com.prasanna.capture360.corrections

import org.opencv.core.Mat

interface PanoramaCorrection {
    val name: String
    var enabled: Boolean
    fun apply(input: Mat): Mat
}

class CorrectionChain(private val stages: List<PanoramaCorrection>) {
    fun apply(input: Mat): Mat {
        var current = input
        stages.filter { it.enabled }.forEach { stage ->
            val next = stage.apply(current)
            if (next !== current) current.release()
            current = next
        }
        return current
    }
    fun setEnabled(name: String, enabled: Boolean) { stages.firstOrNull { it.name == name }?.enabled = enabled }
    companion object {
        fun defaultChain() = CorrectionChain(listOf(EquirectFinishCorrection(), NadirZenithCorrection(), GlareRemovalCorrection(), DotRemovalCorrection(), ColorCorrection(), SharpenCorrection()))
    }
}

class EquirectFinishCorrection : PanoramaCorrection {
    override val name = "equirect_finish"; override var enabled = false
    override fun apply(input: Mat): Mat {
        val targetWidth = input.rows() * 2
        if (input.cols() == targetWidth) return input
        if (input.cols() > targetWidth) return input.submat(org.opencv.core.Rect((input.cols()-targetWidth)/2, 0, targetWidth, input.rows())).clone()
        val result = Mat.zeros(input.rows(), targetWidth, input.type()); val left = (targetWidth-input.cols())/2
        input.copyTo(result.submat(org.opencv.core.Rect(left,0,input.cols(),input.rows())))
        if (left > 0) input.submat(org.opencv.core.Rect(input.cols()-left,0,left,input.rows())).copyTo(result.submat(org.opencv.core.Rect(0,0,left,input.rows())))
        val right = targetWidth-left-input.cols()
        if (right > 0) input.submat(org.opencv.core.Rect(0,0,right,input.rows())).copyTo(result.submat(org.opencv.core.Rect(left+input.cols(),0,right,input.rows())))
        return result
    }
}
class NadirZenithCorrection : PanoramaCorrection { override val name="nadir_zenith"; override var enabled=false; override fun apply(input:Mat)=input }
class GlareRemovalCorrection : PanoramaCorrection { override val name="glare_removal"; override var enabled=false; override fun apply(input:Mat)=input }
class ColorCorrection : PanoramaCorrection { override val name="color_correction"; override var enabled=false; override fun apply(input:Mat)=input }
class DotRemovalCorrection : PanoramaCorrection {
    override val name="dot_removal"; override var enabled=false
    override fun apply(input:Mat):Mat { val gray=Mat(); val mask=Mat(); val result=input.clone(); org.opencv.imgproc.Imgproc.cvtColor(input,gray,org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY); org.opencv.imgproc.Imgproc.threshold(gray,mask,250.0,255.0,org.opencv.imgproc.Imgproc.THRESH_BINARY); org.opencv.imgproc.Imgproc.inpaint(input,mask,result,3.0,org.opencv.imgproc.Imgproc.INPAINT_TELEA); gray.release(); mask.release(); return result }
}
class SharpenCorrection : PanoramaCorrection {
    override val name="sharpen"; override var enabled=false
    override fun apply(input:Mat):Mat { val blur=Mat(); val result=Mat(); org.opencv.imgproc.Imgproc.GaussianBlur(input,blur,org.opencv.core.Size(0.0,0.0),1.0); org.opencv.core.Core.addWeighted(input,1.5,blur,-0.5,0.0,result); blur.release(); return result }
}
