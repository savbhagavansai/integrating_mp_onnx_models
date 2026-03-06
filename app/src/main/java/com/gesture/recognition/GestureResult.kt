package com.gesture.recognition

/**
 * Result of gesture recognition with performance timing data
 * Contains gesture name, confidence, probabilities, and detailed timing breakdown
 *
 * NEW: Separate timing for HandDetector, Landmarks, and Gesture classifier
 */
data class GestureResult(
    val gesture: String,
    val confidence: Float,
    val allProbabilities: FloatArray,
    val handDetected: Boolean = true,
    val bufferProgress: Float = 1f,
    val isStable: Boolean = false,

    // ═══════════════════════════════════════════════════════════
    // UPDATED: Performance timing breakdown (in milliseconds)
    // ═══════════════════════════════════════════════════════════
    val handDetectorTimeMs: Double = 0.0,   // Stage 1: Palm/hand detection
    val landmarksTimeMs: Double = 0.0,      // Stage 2: Landmark extraction
    val gestureTimeMs: Double = 0.0,        // Stage 3: Gesture classification
    val totalTimeMs: Double = 0.0,          // Total pipeline time
    val wasTracking: Boolean = false,       // Was in tracking mode (vs detection)

    // Keep old names for backward compatibility (deprecated)
    @Deprecated("Use handDetectorTimeMs and landmarksTimeMs instead")
    val mediaPipeTimeMs: Double = 0.0,
    @Deprecated("Use gestureTimeMs instead")
    val onnxTimeMs: Double = 0.0
) {
    /**
     * Check if prediction meets confidence threshold
     */
    fun meetsThreshold(): Boolean {
        return confidence >= Config.CONFIDENCE_THRESHOLD
    }

    /**
     * Get formatted gesture name (replace underscores with spaces)
     */
    fun getFormattedGesture(): String {
        return gesture.replace('_', ' ')
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as GestureResult

        if (gesture != other.gesture) return false
        if (confidence != other.confidence) return false
        if (!allProbabilities.contentEquals(other.allProbabilities)) return false
        if (handDetected != other.handDetected) return false
        if (bufferProgress != other.bufferProgress) return false
        if (isStable != other.isStable) return false
        // Timing fields not compared (vary frame to frame)

        return true
    }

    override fun hashCode(): Int {
        var result = gesture.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + allProbabilities.contentHashCode()
        result = 31 * result + handDetected.hashCode()
        result = 31 * result + bufferProgress.hashCode()
        result = 31 * result + isStable.hashCode()
        // Timing fields not included in hash (vary frame to frame)
        return result
    }
}