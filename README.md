# Capture360 Android

A guided 360° image capture system for Android that tracks device orientation and captures frames with optimal spatial distribution. The app combines device sensors with intelligent analysis to ensure comprehensive coverage of a spherical view.

## Key Features

- **IMU Sensor Fusion (Gyro + Accel + Magnetometer)**  
  Combines data from gyroscope, accelerometer, and magnetometer using a complementary filter to provide accurate, drift-resistant device orientation estimates. Enables real-time tracking of device position in 3D space.

- **Drift Correction**  
  Automatically corrects gyroscopic integration drift over time. Complements high-frequency gyroscope data with slower, more stable magnetometer and accelerometer readings to maintain long-term heading accuracy.

- **Angular Coverage Tracking**  
  Monitors which regions of the 360° sphere have been adequately captured based on device orientation. Provides spatial distribution analysis to guide users toward uncaptured areas and ensure uniform coverage across all angles.

- **Real-time Heatmap Visualization**  
  Displays a live heatmap overlay on the camera preview, showing coverage density. Helps users identify gaps and balance capture distribution without having to check separate statistics.

- **Frame Gating for Stable Capture**  
  Automatically filters frames based on device motion stability thresholds. Discards frames captured during jerky movement and captures only when the device orientation is stable, improving stitching quality and reducing motion artifacts.

## Requirements

- Android Studio Flamingo+
- Min SDK 26
- Physical device with gyroscope

## Tools & Environment

### For Docker builds:

- **Docker** — the only requirement
- The Docker image includes JDK 11+, Android SDK, build-tools, and NDK (if needed for native code)
- You can build the project entirely within a container; no local SDK/NDK/JDK installation required

### For Android Studio / local command-line builds:

- Java JDK 11 or 17 (ensure `JAVA_HOME` points to the JDK)
- Android SDK (platforms and build-tools for API 26+)
- Android NDK (if building native code in `app/cpp`)
- Gradle wrapper (project includes `gradlew` and wrapper files)

### Network and caching notes:

The Docker build and Gradle wrapper download require network access to `services.gradle.org`. If your environment restricts outbound TLS traffic, pre-download the Gradle distribution or provide CA certs in the build image.

## Build

### Android Studio (recommended)

1. Clone the repository
2. Open the project in Android Studio
3. Let Android Studio sync Gradle and download dependencies
4. Build and run on a connected device

### Command-line (Linux / macOS / Windows WSL / PowerShell)

```bash
# From the project root
./gradlew assembleDebug
./gradlew installDebug    # installs on a connected device
```

### Build inside Docker (optional)

1. Build the Docker image:

```bash
docker build -t capture360 .
```

2. If the Docker build fails while downloading Gradle, ensure HTTPS is allowed and CA certificates are present in the base image. Alternatively, pre-populate the Gradle wrapper cache on the host and mount it into `/root/.gradle` when building.

## Running and testing

- Use a physical Android device with USB debugging enabled for best results.
- If using an emulator, enable sensors and verify gyroscope support; behavior can differ from real hardware.

## Formatting and linters

- Kotlin: `ktlint` (project may configure ktlint via Gradle)
- Java: `google-java-format` or Gradle `spotless` (if configured)

To run formatters via Gradle (if configured):

```bash
./gradlew ktlintFormat spotlessApply
```

If Gradle cannot download required formatting tools, check network/SSL settings or run formatters locally in Android Studio.

## Common troubleshooting

- "zip END header not found" / Gradle download errors: ensure `gradle/wrapper/gradle-wrapper.properties` uses `https://` (this repo was updated to HTTPS) and the build environment can access `services.gradle.org`.
- SSL / certificate errors in container: install CA certificates in the base image (e.g., `ca-certificates` package) or configure proxy/CA trust.
- "elmName / root is null" or XML parsing errors: inspect layout XML files under `app/src/main/res/layout` for malformed XML or null bytes. Open files with a binary-capable editor to detect embedded NULs.

---

If you'd like, I can add:

- a small CI job to run formatters and build the APK
- pre-download Gradle into the Docker image to avoid network errors
- run the formatters now (requires network access for Gradle dependencies)

Tell me which option you prefer and I'll implement it.
