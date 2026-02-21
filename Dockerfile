FROM eclipse-temurin:17-jdk-jammy

ENV ANDROID_HOME=/sdk
ENV PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools

RUN apt-get update && \
    apt-get install -y wget unzip git && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p $ANDROID_HOME/cmdline-tools

RUN wget https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip -O tools.zip && \
    unzip tools.zip -d $ANDROID_HOME/cmdline-tools && \
    rm tools.zip && \
    mv $ANDROID_HOME/cmdline-tools/cmdline-tools $ANDROID_HOME/cmdline-tools/latest

RUN yes | sdkmanager --licenses

RUN sdkmanager \
    "platform-tools" \
    "platforms;android-34" \
    "build-tools;34.0.0"

WORKDIR /app

RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates

COPY app/libs app/libs
RUN ls -la app/libs
# Copy Gradle wrapper & configs first (for cache optimization)

COPY gradlew .
COPY gradle gradle
COPY settings.gradle.kts .
COPY build.gradle.kts .
COPY gradle.properties .
COPY app/build.gradle.kts app/

RUN chmod +x gradlew

# Download dependencies (cache layer)
RUN ./gradlew --no-daemon dependencies

# Now copy full source
COPY . .

# Build
RUN ./gradlew assembleDebug --no-daemon \
    -Dorg.gradle.jvmargs="-Xmx3072m -XX:+UseParallelGC" \
    -Dorg.gradle.workers.max=2 \
    --stacktrace
u