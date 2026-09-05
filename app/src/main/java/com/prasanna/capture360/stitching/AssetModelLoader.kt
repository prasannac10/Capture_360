package com.prasanna.capture360.stitching

import java.io.File

object AssetModelLoader {
    fun copyIfPresent(context: android.content.Context, assetName: String): File? = try {
        context.assets.open(assetName).use { input ->
            val output = File(context.filesDir, assetName); output.parentFile?.mkdirs(); output.outputStream().use { input.copyTo(it) }; output
        }
    } catch (_: Exception) { null }
}
