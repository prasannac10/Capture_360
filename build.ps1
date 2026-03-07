# Quick build script with caching

# First build (15-20 minutes)
Write-Host "Starting Docker build with volume caching..." -ForegroundColor Green
docker-compose build

# Wait for container to finish
Write-Host "Build complete!" -ForegroundColor Green

# Find and install the APK
$apkPath = "app\build\outputs\apk\debug\app-debug.apk"
if (Test-Path $apkPath) {
    Write-Host "Found APK at: $apkPath" -ForegroundColor Green
    
    # Copy to Desktop for easy access
    $desktopPath = "$env:USERPROFILE\Desktop\capture360.apk"
    Copy-Item $apkPath $desktopPath -Force
    Write-Host "Copied APK to Desktop: $desktopPath" -ForegroundColor Green
    
    # Also copy to local directory
    $localCopyPath = ".\capture360.apk"
    Copy-Item $apkPath $localCopyPath -Force
    Write-Host "Copied APK to current directory: $localCopyPath" -ForegroundColor Green
    
    Write-Host "Installing to device..." -ForegroundColor Green
    adb install -r $apkPath
    Write-Host "Installation complete!" -ForegroundColor Green
} else {
    Write-Host "APK not found at $apkPath" -ForegroundColor Red
}
