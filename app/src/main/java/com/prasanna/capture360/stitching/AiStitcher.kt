package com.prasanna.capture360.stitching

import android.content.Context
import ai.onnxruntime.*
import com.prasanna.capture360.sensors.Orientation
import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

class AiStitcher(private val context: Context) : PanoramaStitcher, AutoCloseable {
    companion object { private const val MODEL_ASSET="pano_model.onnx"; private const val MIN_N=4; private const val MAX_N=30; private const val S=224; private const val H=256; private const val W=512 }
    private var env: OrtEnvironment?=null; private var session: OrtSession?=null
    private fun ensureSession(){ if(session!=null)return; val file=AssetModelLoader.copyIfPresent(context,MODEL_ASSET)?:throw IllegalStateException("AI model not available yet: $MODEL_ASSET is not bundled"); env=OrtEnvironment.getEnvironment(); session=env!!.createSession(file.absolutePath,OrtSession.SessionOptions()) }
    override fun stitch(frameFiles:List<File>,poses:List<Orientation>,cameraProfile:CameraProfile):Result<Mat>=try{
        require(frameFiles.size==poses.size){"Frame/pose count mismatch"}; require(frameFiles.size in MIN_N..MAX_N){"AI stitcher supports $MIN_N-$MAX_N frames; got ${frameFiles.size}"}; ensureSession()
        val n=frameFiles.size; val images=FloatArray(n*3*S*S); val rotations=FloatArray(n*9); val camera=FloatArray(n*5)
        for(i in 0 until n){ preprocess(frameFiles[i]).copyInto(images,i*3*S*S); rotationMatrix(poses[i]).copyInto(rotations,i*9); camera[i*5]=cameraProfile.fxNorm; camera[i*5+1]=cameraProfile.fyNorm; camera[i*5+2]=cameraProfile.cxNorm; camera[i*5+3]=cameraProfile.cyNorm; camera[i*5+4]=if(cameraProfile.projection==CameraProjection.PINHOLE)1f else 0f }
        val ish=longArrayOf(1,n.toLong(),3,S.toLong(),S.toLong()); val rsh=longArrayOf(1,n.toLong(),3,3); val msh=longArrayOf(1,n.toLong()); val ksh=longArrayOf(1,n.toLong(),5); val mask=FloatArray(n){1f}
        OnnxTensor.createTensor(env!!,FloatBuffer.wrap(images),ish).use{ix->OnnxTensor.createTensor(env!!,FloatBuffer.wrap(rotations),rsh).use{rx->OnnxTensor.createTensor(env!!,FloatBuffer.wrap(mask),msh).use{mx->OnnxTensor.createTensor(env!!,FloatBuffer.wrap(camera),ksh).use{kx->session!!.run(mapOf("images" to ix,"rotations" to rx,"frame_mask" to mx,"camera_params" to kx)).use{out->@Suppress("UNCHECKED_CAST") val v=out[0].value as Array<Array<Array<FloatArray>>>;Result.success(toBgrMat(v[0]))}}}}}
    }catch(t:Throwable){Result.failure(t)}
    private fun preprocess(file:File):FloatArray{val bgr=Imgcodecs.imread(file.absolutePath,Imgcodecs.IMREAD_COLOR);require(!bgr.empty()){"Unable to decode ${file.name}"};val rgb=Mat();val canvas=Mat.zeros(S,S,CvType.CV_8UC3);try{Imgproc.cvtColor(bgr,rgb,Imgproc.COLOR_BGR2RGB);val scale=min(S.toDouble()/rgb.cols(),S.toDouble()/rgb.rows());val resized=Mat();val w=maxOf(1,(rgb.cols()*scale).toInt());val h=maxOf(1,(rgb.rows()*scale).toInt());Imgproc.resize(rgb,resized,Size(w.toDouble(),h.toDouble()),0.0,0.0,Imgproc.INTER_LINEAR);resized.copyTo(canvas.submat(Rect((S-w)/2,(S-h)/2,w,h)));resized.release();val f=Mat();canvas.convertTo(f,CvType.CV_32FC3,1.0/255.0);val data=FloatArray(S*S*3);f.get(0,0,data);f.release();val chw=FloatArray(data.size);val plane=S*S;for(p in 0 until plane){chw[p]=data[p*3];chw[plane+p]=data[p*3+1];chw[2*plane+p]=data[p*3+2]};return chw}finally{bgr.release();rgb.release();canvas.release()}}
    private fun rotationMatrix(o:Orientation):FloatArray{val y=o.yaw.toDouble();val p=o.pitch.toDouble();val r=o.roll.toDouble();val cy=cos(y);val sy=sin(y);val cp=cos(p);val sp=sin(p);val cr=cos(r);val sr=sin(r);return floatArrayOf((cy*cr+sy*sp*sr).toFloat(),(sr*cp).toFloat(),(-sy*cr+cy*sp*sr).toFloat(),(-cy*sr+sy*sp*cr).toFloat(),(cr*cp).toFloat(),(sr*sy+cy*sp*cr).toFloat(),(sy*cp).toFloat(),(-sp).toFloat(),(cy*cp).toFloat())}
    private fun toBgrMat(v:Array<Array<FloatArray>>):Mat{val inter=FloatArray(H*W*3);val plane=H*W;for(y in 0 until H)for(x in 0 until W){val p=y*W+x;inter[p*3]=v[2][y][x];inter[p*3+1]=v[1][y][x];inter[p*3+2]=v[0][y][x]};val f=Mat(H,W,CvType.CV_32FC3);f.put(0,0,inter);val out=Mat();f.convertTo(out,CvType.CV_8UC3,255.0);f.release();return out}
    override fun close(){session?.close();session=null;env?.close();env=null}
}
