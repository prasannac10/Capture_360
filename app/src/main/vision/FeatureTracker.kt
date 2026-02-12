class FeatureTracker {

    private val orb = ORB.create()

    private var lastDescriptors: Mat? = null
    private var lastKeypoints: MatOfKeyPoint? = null

    fun estimateYawCorrection(currentFrame: Mat): Float {

        val gray = Mat()
        Imgproc.cvtColor(currentFrame, gray, Imgproc.COLOR_BGR2GRAY)

        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()

        orb.detectAndCompute(gray, Mat(), keypoints, descriptors)

        if (lastDescriptors == null) {
            lastDescriptors = descriptors
            lastKeypoints = keypoints
            return 0f
        }

        val matcher = BFMatcher.create(Core.NORM_HAMMING, true)
        val matches = MatOfDMatch()
        matcher.match(lastDescriptors, descriptors, matches)

        val matchList = matches.toList()

        if (matchList.size < 15) return 0f

        // Approximate yaw drift correction
        val avgShift = matchList.map { it.distance }.average()

        lastDescriptors = descriptors
        lastKeypoints = keypoints

        return (avgShift * 0.0005f).toFloat()
    }
}
