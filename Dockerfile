FROM eclipse-temurin:17-jdk-jammy

ENV ANDROID_HOME=/sdk
ENV PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools
ENV GRADLE_USER_HOME=/gradle_cache

# Install tools once
RUN apt-get update && \
    apt-get install -y wget unzip git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Download & setup Android SDK (this layer is cached)
RUN mkdir -p $ANDROID_HOME/cmdline-tools && \
    wget https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip -O tools.zip && \
    unzip tools.zip -d $ANDROID_HOME/cmdline-tools && \
    rm tools.zip && \
    mv $ANDROID_HOME/cmdline-tools/cmdline-tools $ANDROID_HOME/cmdline-tools/latest && \
    yes | sdkmanager --licenses && \
    sdkmanager "platform-tools" "platforms;android-34" "build-tools;34.0.0"

WORKDIR /app

# Copy only gradle configs first (maximizes caching)
COPY gradlew .
COPY gradle ./gradle
COPY settings.gradle.kts build.gradle.kts gradle.properties ./
COPY app/build.gradle.kts app/proguard-rules.pro ./app/

RUN chmod +x gradlew && \
    ./gradlew --no-daemon dependencies --stacktrace

# Now copy the rest (source code changes won't re-download SDK/Gradle)
COPY . .

# Build the APK
RUN chmod +x gradlew && ./gradlew assembleDebug --no-daemon \
    -Dorg.gradle.jvmargs="-Xmx2048m" \
    --stacktrace