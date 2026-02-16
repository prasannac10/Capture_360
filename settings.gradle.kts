pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.PREFER_SETTINGS)

    repositories {
        google()
        mavenCentral()

        flatDir {
            dirs("app/libs")
        }
    }
}

rootProject.name = "capture360-android"
include(":app")

