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
import com.prasanna.capture360.stitching.CameraProjection
import com.prasanna.capture360.stitching.CameraProfile
import com.prasanna.capture360.stitching.ClassicalStitcher
import com.prasanna.capture360.stitching.PanoramaStitcher
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
    private lateinit var preview:PreviewView; private lateinit var overlay:GuidanceOverlayView; private lateinit var progressBar:ProgressBar; private lateinit var statusText:TextView; private lateinit var btnFinish:Button; private lateinit var modeSwitch:Switch; private lateinit var cameraProfileSwitch:Switch
    private var cameraProjection=CameraProjection.PINHOLE
    // Calibrate this once for the phone model. 70° is only a safe default, not a universal phone value.
    private val phoneHorizontalFovDeg=70.0
    private val minFramesPinhole=20; private val maxFramesPinhole=30; private val targetFramesFisheye=6; private val maxFramesFisheye=8
    private val fisheyeProfile=CameraProfile(CameraProjection.FISHEYE_180)
    private val pinholeProfile:CameraProfile get(){val f=1.0/Math.tan(Math.toRadians(phoneHorizontalFovDeg)/2.0);return CameraProfile(CameraProjection.PINHOLE,f.toFloat(),f.toFloat(),0f,0f)}

    override fun onCreate(savedInstanceState:Bundle?){
        super.onCreate(savedInstanceState); initOpenCV(); setContentView(R.layout.activity_main)
        preview=findViewById(R.id.preview); overlay=findViewById(R.id.overlay); progressBar=findViewById(R.id.progressBar); statusText=findViewById(R.id.statusText); btnFinish=findViewById(R.id.btnFinish); modeSwitch=findViewById(R.id.modeSwitch); cameraProfileSwitch=findViewById(R.id.cameraProfileSwitch)
        aiStitcher=AiStitcher(this)
        modeSwitch.setOnCheckedChangeListener{_,checked->modeSwitch.text=if(checked)"AI" else "OpenCV"}
        cameraProfileSwitch.setOnCheckedChangeListener{_,checked->cameraProjection=if(checked)CameraProjection.PINHOLE else CameraProjection.FISHEYE_180;resetCaptureState()}
        updateProfileUi(); btnFinish.setOnClickListener{initiateStitching()}
        if(allPermissionsGranted())startCaptureLogic()else requestRequiredPermissions()
    }
    override fun onDestroy(){if(::camera.isInitialized)camera.shutdown();if(::fusion.isInitialized)fusion.stop();aiStitcher.close();super.onDestroy()}
    override fun onRequestPermissionsResult(r:Int,p:Array<String>,g:IntArray){super.onRequestPermissionsResult(r,p,g);if(r==1001&&g.isNotEmpty()&&g.all{it==PackageManager.PERMISSION_GRANTED})startCaptureLogic()else if(r==1001)statusText.text="Permissions required: Camera + Storage"}
    private fun resetCaptureState(){capturedFrames.forEach{it.delete()};capturedFrames.clear();capturedPoses.clear();lastCaptureYaw=0f;lastCapturePitch=0f;if(::tracker.isInitialized)tracker=AngularCoverageTracker(10,3);btnFinish.visibility=View.GONE;progressBar.progress=0;updateProfileUi()}
    private fun updateProfileUi(){val fisheye=cameraProjection==CameraProjection.FISHEYE_180;cameraProfileSwitch.text=if(fisheye)"Fisheye 180°" else "Pinhole / Phone";statusText.text=if(fisheye)"Fisheye: capture 4–8 views (target 6)" else "Phone: capture 20–30 views"}
    private fun currentMinFrames()=if(cameraProjection==CameraProjection.FISHEYE_180)4 else minFramesPinhole
    private fun currentMaxFrames()=if(cameraProjection==CameraProjection.FISHEYE_180)maxFramesFisheye else maxFramesPinhole
    private fun currentTargetFrames()=if(cameraProjection==CameraProjection.FISHEYE_180)targetFramesFisheye else maxFramesPinhole
    private fun currentProfile()=if(cameraProjection==CameraProjection.FISHEYE_180)fisheyeProfile else pinholeProfile
    private fun startCaptureLogic(){try{camera=CameraController(this,this);fusion=SensorFusionManager(this);tracker=AngularCoverageTracker(10,3);resetCaptureState();getExternalFilesDir(null)?.listFiles()?.filter{it.name.startsWith("capture_")}?.forEach{it.delete()};camera.startCamera(preview);fusion.start();lifecycleScope.launch{while(isActive){try{val o=fusion.getOrientation();val yaw=Math.toDegrees(o.yaw.toDouble()).toFloat();val pitch=Math.toDegrees(o.pitch.toDouble()).toFloat();val rem=tracker.getRemainingCount();val count=capturedFrames.size;val target=currentTargetFrames();val max=currentMaxFrames();overlay.updateDisplay(tracker.getGrid(),"${tracker.getTargetDirection(yaw,pitch,getScreenRotation())}\n($count/$target Captured)");progressBar.progress=((count.toFloat()/target)*100).toInt().coerceAtMost(100);if(count>=currentMinFrames())btnFinish.visibility=View.VISIBLE;val yd=calculateYawDiff(yaw,lastCaptureYaw);val pd=abs(pitch-lastCapturePitch);if(count<max&&!isCameraBusy&&rem>0&&gate.shouldCapture(o)&&!tracker.isCurrentAreaCovered(o)&&sqrt(yd.toDouble().pow(2.0)+pd.toDouble().pow(2.0))>8f)triggerImageCapture(yaw,pitch,o)}catch(e:Exception){Log.e("ControlLoop","Capture loop error",e)};delay(100)}}}catch(e:Exception){Log.e("MainActivity","Camera startup failed",e);statusText.text="Error: ${e.localizedMessage}"}}
    private fun triggerImageCapture(yaw:Float,pitch:Float,o:Orientation){if(isCameraBusy)return;isCameraBusy=true;val dir=getExternalFilesDir(null)?:run{isCameraBusy=false;return};val file=File(dir,"capture_${System.currentTimeMillis()}.jpg");camera.captureFrame(file){if(file.exists()&&file.length()>0){capturedFrames+=file;capturedPoses+=o;tracker.update(o);runOnUiThread{lastCaptureYaw=yaw;lastCapturePitch=pitch;statusText.text="${if(cameraProjection==CameraProjection.FISHEYE_180)"Fisheye" else "Phone"} captured: ${capturedFrames.size}/${currentTargetFrames()}"}};isCameraBusy=false}}
    private fun initiateStitching(){val dir=getExternalFilesDir(null)?:run{statusText.text="Error: Cannot access capture directory";return};val n=capturedFrames.size;if(n<currentMinFrames()){statusText.text="Need at least ${currentMinFrames()} images to stitch (found $n)";return};if(n!=capturedPoses.size){statusText.text="Error: frame/pose count mismatch";return};btnFinish.isEnabled=false;modeSwitch.isEnabled=false;cameraProfileSwitch.isEnabled=false;progressBar.isIndeterminate=true;val ai=modeSwitch.isChecked;val stitcher:PanoramaStitcher=if(ai)aiStitcher else classicalStitcher;val profile=currentProfile();statusText.text="${if(ai)"AI" else "OpenCV"} ${if(profile.projection==CameraProjection.FISHEYE_180)"fisheye" else "pinhole"} stitching $n images...";lifecycleScope.launch(Dispatchers.Default){var pano:Mat?=null;try{pano=stitcher.stitch(capturedFrames.toList(),capturedPoses.toList(),profile).getOrElse{throw it};pano=correctionChain.apply(pano!!);val file=File(dir,"panorama_result.jpg");if(!Imgcodecs.imwrite(file.absolutePath,pano))throw IllegalStateException("Failed to write panorama_result.jpg");val saved=saveImageToGallery(file,"panorama_${System.currentTimeMillis()}.jpg");withContext(Dispatchers.Main){statusText.text="Success! Panorama saved${if(saved)" to gallery ✓" else ""}"}}catch(e:Throwable){withContext(Dispatchers.Main){statusText.text=e.message?:"Stitching failed";Toast.makeText(this@MainActivity,statusText.text,Toast.LENGTH_LONG).show()}}finally{pano?.release();withContext(Dispatchers.Main){btnFinish.isEnabled=true;modeSwitch.isEnabled=true;cameraProfileSwitch.isEnabled=true;progressBar.isIndeterminate=false;progressBar.progress=100}}}}
    private fun saveImageToGallery(src:File,name:String):Boolean=try{if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.Q){val v=ContentValues().apply{put(MediaStore.Images.Media.DISPLAY_NAME,name);put(MediaStore.Images.Media.MIME_TYPE,"image/jpeg");put(MediaStore.Images.Media.RELATIVE_PATH,"Pictures/Capture360");put(MediaStore.Images.Media.IS_PENDING,1)};val uri=contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,v)?:return false;contentResolver.openOutputStream(uri)?.use{o->src.inputStream().use{i->i.copyTo(o)}};v.clear();v.put(MediaStore.Images.Media.IS_PENDING,0);contentResolver.update(uri,v,null,null);true}else{val d=File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),"Capture360/$name");d.parentFile?.mkdirs();src.copyTo(d,true);MediaStore.Images.Media.insertImage(contentResolver,d.absolutePath,d.name,null);true}}catch(e:Exception){Log.e("Gallery","Failed to save panorama",e);false}
    private fun requestRequiredPermissions(){val p=mutableListOf(Manifest.permission.CAMERA);if(Build.VERSION.SDK_INT<Build.VERSION_CODES.Q)p+=Manifest.permission.WRITE_EXTERNAL_STORAGE;if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.TIRAMISU)p+=Manifest.permission.READ_MEDIA_IMAGES else if(Build.VERSION.SDK_INT<Build.VERSION_CODES.Q)p+=Manifest.permission.READ_EXTERNAL_STORAGE;ActivityCompat.requestPermissions(this,p.toTypedArray(),1001)}
    private fun allPermissionsGranted():Boolean{if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED)return false;if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.TIRAMISU)return ContextCompat.checkSelfPermission(this,Manifest.permission.READ_MEDIA_IMAGES)==PackageManager.PERMISSION_GRANTED;if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.Q)return true;return ContextCompat.checkSelfPermission(this,Manifest.permission.READ_EXTERNAL_STORAGE)==PackageManager.PERMISSION_GRANTED}
    private fun initOpenCV(){if(!OpenCVLoader.initDebug())Log.e("OpenCV","OpenCV initialization failed")}
    private fun getScreenRotation():Int{if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.P)return display?.rotation?:0;@Suppress("DEPRECATION") val r=windowManager.defaultDisplay.rotation;return r}
    private fun calculateYawDiff(current:Float,target:Float):Float{var d=target-current;while(d<=-180)d+=360;while(d>180)d-=360;return d}
}
