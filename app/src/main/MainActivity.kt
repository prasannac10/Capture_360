package com.yourcompany.capture360

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.lifecycle.lifecycleScope
import com.yourcompany.capture360.camera.CameraController
import com.yourcompany.capture360.sensors.SensorFusionManager
import com.yourcompany.capture360.tracking.AngularCoverageTracker
import com.yourcompany.capture360.tracking.FrameGate
import com.yourcompany.capture360.ui.GuidanceOverlayView
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var camera: CameraController
    private lateinit var fusion: SensorFusionManager
    private lateinit var tracker: AngularCoverageTracker
    private val gate = FrameGate()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val preview = findViewById<PreviewView>(R.id.preview)
        val overlay = findViewById<GuidanceOverlayView>(R.id.overlay)

        camera = CameraController(this, this)
        fusion = SensorFusionManager(this)
        tracker = AngularCoverageTracker()

        camera.startCamera(preview)
        fusion.start()

        lifecycleScope.launch {
            while (isActive) {

                val orientation = fusion.getOrientation()

                if (gate.shouldCapture(orientation)) {

                    val file = File(
                        getExternalFilesDir(null),
                        "${System.currentTimeMillis()}.jpg"
                    )

                    camera.captureFrame(file) {
                        tracker.update(orientation)
                        overlay.setCoverage(tracker.getGrid())
                    }
                }

                delay(80)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        fusion.stop()
    }
}
