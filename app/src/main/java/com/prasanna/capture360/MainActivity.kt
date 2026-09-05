package com.prasanna.capture360

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.prasanna.capture360.camera.CameraController
import com.prasanna.capture360.corrections.CorrectionChain
import com.prasanna.capture360.sensors.Orientation
import com.prasanna.capture360.sensors.SensorFusionManager
import com.prasanna.capture360.stitching.AiStitcher
import com.prasanna.capture360.stitching.ClassicalStitcher
import com.prasanna.capture360.stitching.PanoramaStitcher
import com.prasanna.capture360.stitching.StitchingMode
import com.prasanna.capture360.tracking.AngularCoverageTracker
import com.prasanna.capture360.tracking.FrameGate
import com.prasanna.capture360.ui.GuidanceOverlayView
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {
    private var lastCaptureYaw=0f; private var lastCapturePitch=0f; private var isCameraBusy=false
    private lateinit var camera:CameraController; private lateinit var fusion:SensorFusionManager; private lateinit var tracker:AngularCoverageTracker
    private val gate=FrameGate(); private val capturedFrames=mutableListOf<File>(); private val capturedPoses=mutableListOf<Orientation>()
    private lateinit var aiStitcher:AiStitcher; private val classicalStitcher=ClassicalStitcher(); private val correctionChain=CorrectionChain.defaultChain()
    private lateinit var preview:PreviewView; private lateinit var overlay:GuidanceOverlayView; private lateinit var progressBar:ProgressBar; private lateinit var statusText:TextView; private lateinit var btnFinish:Button; private lateinit var modeSwitch:Switch
    override fun onCreate(savedInstanceState:Bundle?){super.onCreate(savedInstanceState);initOpenCV();setContentView(R.layout.activity_main);preview=findViewById(R.id.preview);overlay=findViewById(R.id.overlay);progressBar=findViewById(R.id.progressBar);statusText=findViewById(R.id.statusText);btnFinish=findViewById(R.id.btnFinish);modeSwitch=findViewById(R.id.modeSwitch);aiStitcher=AiStitcher(this);modeSwitch.setOnCheckedChangeListener{_,checked->modeSwitch.text=if(checked)"AI" else "OpenCV"};btnFinish.setOnClickListener{initiateStitching()};if(allPermissionsGranted())startCaptureLogic()else requestRequiredPermissions()}
    override fun onDestroy(){if(::camera.isInitialized)camera.shutdown();if(::fusion.isInitialized)fusion.stop();aiStitcher.close();super.onDestroy()}
    override fun onRequestPermissionsResult(r:Int,p:Array<String>,g:IntArray){super.onRequestPermissionsResult(r,p,g);if(r==1001&&g.isNotEmpty()&&g.all{it==PackageManager.PERMISSION_GRANTED})startCaptureLogic()else if(r==1001)statusText.text="Permissions required: Camera + Storage"}
    private fun startCaptureLogic(){try{camera=CameraController(this,this);fusion=SensorFusionManager(this);tracker=AngularCoverageTracker(10,3);capturedFrames.clear();capturedPoses.clear();getExternalFilesDir(null)?.listFiles()?.filter{it.name.startsWith("capture_")}?.forEach{it.delete()};camera.startCamera(preview);fusion.start();statusText.text="Ready - Rotate to capture";lifecycleScope.launch{while(isActive){try{val o=fusion.getOrientation();val yaw=Math.toDegrees(o.yaw.toDouble()).toFloat();val pitch=Math.toDegrees(o.pitch.toDouble()).toFloat();val rem=tracker.getRemainingCount();val count=capturedFrames.size;overlay.updateDisplay(tracker.getGrid(),"${tracker.getTargetDirection(yaw,pitch,getScreenRotation())}\n($count/30 Captured)");progressBar.progress=((count/30f)*100).toInt();if(count>=20)btnFinish.visibility=View.VISIBLE;val yd=calculateYawDiff(yaw,lastCaptureYaw);val pd=abs(pitch-lastCapturePitch);if(!isCameraBusy&&rem>0&&gate.shouldCapture(o)&&!tracker.isCurrentAreaCovered(o)&&sqrt(yd.toDouble().pow(2.0)+pd.toDouble().pow(2.0))>8f)triggerImageCapture(yaw,pitch,o)}catch(e:Exception){Log.e("ControlLoop","Capture loop error",e)};delay(100)}}}catch(e:Exception){Log.e("MainActivity","Camera startup failed",e);statusText.text="Error: ${e.localizedMessage}"}}
    private fun triggerImageCapture(yaw:Float,pitch:Float,o:Orientation){if(isCameraBusy)return;isCameraBusy=true;val dir=getExternalFilesDir(null)?:run{isCameraBusy=false;return};val file=File(dir,"capture_${System.currentTimeMillis()}.jpg");camera.captureFrame(file){if(file.exists()&&file.length()>0){capturedFrames+=file;capturedPoses+=o;tracker.update(o);lastCaptureYaw=yaw;lastCapturePitch=pitch;runOnUiThread{statusText.text="Captured: ${capturedFrames.size}/30"}};isCameraBusy=false}}
    private fun initiateStitching(){val dir=getExternalFilesDir(null)?:run{statusText.text="Error: Cannot access capture directory";return};if(capturedFrames.size<2){statusText.text="Need at least 2 images to stitch! (Found ${capturedFrames.size})";return};if(capturedFrames.size!=capturedPoses.size){statusText.text="Error: frame/pose count mismatch";return};btnFinish.isEnabled=false;modeSwitch.isEnabled=false;progressBar.isIndeterminate=true;val ai=modeSwitch.isChecked;val stitcher:PanoramaStitcher=if(ai)aiStitcher else classicalStitcher;statusText.text="${if(ai)"AI" else "OpenCV"} stitching ${capturedFrames.size} images...";lifecycleScope.launch(Dispatchers.Default){var pano:Mat?=null;try{pano=stitcher.stitch(capturedFrames.toList(),capturedPoses.toList()).getOrElse{throw it};pano=correctionChain.apply(pano!!);val file=File(dir,"panorama_result.jpg");if(!Imgcodecs.imwrite(file.absolutePath,pano))throw IllegalStateException("Failed to write panorama_result.jpg");val saved=saveImageToGallery(file,"panorama_${System.currentTimeMillis()}.jpg");withContext(Dispatchers.Main){statusText.text="Success! Panorama saved${if(saved)" to gallery ✓" else ""}"}}catch(e:Throwable){withContext(Dispatchers.Main){statusText.text=e.message?:"Stitching failed";Toast.makeText(this@MainActivity,statusText.text,Toast.LENGTH_LONG).show()}}finally{pano?.release();withContext(Dispatchers.Main){btnFinish.isEnabled=true;modeSwitch.isEnabled=true;progressBar.isIndeterminate=false;progressBar.progress=100}}}}
    private fun saveImageToGallery(src:File,name:String):Boolean=try{if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.Q){val v=ContentValues().apply{put(MediaStore.Images.Media.DISPLAY_NAME,name);put(MediaStore.Images.Media.MIME_TYPE,"image/jpeg");put(MediaStore.Images.Media.RELATIVE_PATH,"Pictures/Capture360");put(MediaStore.Images.Media.IS_PENDING,1)};val uri=contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,v)?:return false;contentResolver.openOutputStream(uri)?.use{o->src.inputStream().use{i->i.copyTo(o)}};v.clear();v.put(MediaStore.Images.Media.IS_PENDING,0);contentResolver.update(uri,v,null,null);true}else{val d=File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),"Capture360/$name");d.parentFile?.mkdirs();src.copyTo(d,true);MediaStore.Images.Media.insertImage(contentResolver,d.absolutePath,d.name,null);true}}catch(e:Exception){Log.e("Gallery","Failed to save panorama",e);false}
    private fun requestRequiredPermissions(){val p=mutableListOf(Manifest.permission.CAMERA);if(Build.VERSION.SDK_INT<Build.VERSION_CODES.Q)p+=Manifest.permission.WRITE_EXTERNAL_STORAGE;if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.TIRAMISU)p+=Manifest.permission.READ_MEDIA_IMAGES else if(Build.VERSION.SDK_INT<Build.VERSION_CODES.Q)p+=Manifest.permission.READ_EXTERNAL_STORAGE;ActivityCompat.requestPermissions(this,p.toTypedArray(),1001)}
    private fun allPermissionsGranted():Boolean{if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED)return false;if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.TIRAMISU)return ContextCompat.checkSelfPermission(this,Manifest.permission.READ_MEDIA_IMAGES)==PackageManager.PERMISSION_GRANTED;if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.Q)return true;return ContextCompat.checkSelfPermission(this,Manifest.permission.READ_EXTERNAL_STORAGE)==PackageManager.PERMISSION_GRANTED}
    private fun initOpenCV(){if(!OpenCVLoader.initDebug())Log.e("OpenCV","OpenCV initialization failed")}
    private fun getScreenRotation():Int{if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.P)return display?.rotation?:0;@Suppress("DEPRECATION") val r=windowManager.defaultDisplay.rotation;return r}
    private fun calculateYawDiff(current:Float,target:Float):Float{var d=target-current;while(d<=-180)d+=360;while(d>180)d-=360;return d}
}
