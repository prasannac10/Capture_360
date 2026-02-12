# Preserve CameraX
-keep class androidx.camera.** { *; }
-dontwarn androidx.camera.**

# Preserve Lifecycle components
-keep class androidx.lifecycle.** { *; }
-dontwarn androidx.lifecycle.**

# Keep Kotlin metadata
-keep class kotlin.Metadata { *; }

# Coroutines
-keep class kotlinx.coroutines.** { *; }
-dontwarn kotlinx.coroutines.**

# Keep your app package
-keep class com.prasanna.capture360.** { *; }

# Remove logs in release
-assumenosideeffects class android.util.Log {
    public static *** d(...);
    public static *** v(...);
}
