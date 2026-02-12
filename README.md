# Capture360 Android

Guided 360° capture system with:

- IMU sensor fusion (gyro + accel + magnetometer)
- Drift correction
- Angular coverage tracking
- Real-time heatmap visualization
- Frame gating for stable capture

## Requirements

- Android Studio Flamingo+
- Min SDK 26
- Physical device with gyroscope

## Build

1. Clone repo
2. Open in Android Studio
3. Sync Gradle
4. Run on device

## Architecture

Camera → Sensor Fusion → Coverage Tracker → Frame Gate → UI Overlay
