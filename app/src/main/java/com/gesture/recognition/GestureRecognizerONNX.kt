package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log

/**
 * Complete Gesture Recognition System with ONNX Hand Tracking
 *
 * Pipeline:
 * 1. HandTrackerONNX: Detect hand + extract 21 landmarks (NPU)
 * 2. LandmarkNormalizer: Normalize landmarks
 * 3. SequenceBuffer: Buffer 15 frames
 * 4. ONNXInference: Classify gesture (NPU)
 *
 * All models run on NPU for maximum performance!
 */
class GestureRecognizerONNX(private val context: Context) {

    companion object {
        private const val TAG = "GestureRecognizerONNX"
    }

    // Components
    val handTracker: HandTrackerONNX
    private val normalizer: LandmarkNormalizer
    val sequenceBuffer: SequenceBuffer
    val gestureClassifier: ONNXInference

    // State
    private var lastLandmarks: FloatArray? = null

    init {
        Log.d(TAG, "Initializing Gesture Recognizer ONNX...")

        handTracker = HandTrackerONNX(context)
        normalizer = LandmarkNormalizer()
        sequenceBuffer = SequenceBuffer(Config.SEQUENCE_LENGTH)
        gestureClassifier = ONNXInference(context)

        Log.d(TAG, "✓ Gesture Recognizer ONNX ready")
    }

    /**
     * Process frame through complete pipeline
     *
     * @param bitmap Input frame
     * @return GestureResultONNX or null if no hand detected
     */
    fun processFrame(bitmap: Bitmap): GestureResultONNX? {
        // Stage 1: Hand tracking (detector + landmarks on NPU)
        val trackingResult = handTracker.processFrame(bitmap)

        if (trackingResult == null) {
            // No hand detected
            lastLandmarks = null
            sequenceBuffer.clear()
            return null
        }

        // Stage 2: Normalize landmarks
        val landmarksFlat = flattenLandmarks(trackingResult.landmarks)
        val normalizedLandmarks = normalizer.normalize(landmarksFlat)

        // Store for external access
        lastLandmarks = landmarksFlat

        // Stage 3: Add to sequence buffer
        sequenceBuffer.addLandmarks(normalizedLandmarks)

        if (!sequenceBuffer.isFull()) {
            // Buffer not full yet, no gesture prediction
            return GestureResultONNX(
                gestureResult = null,
                landmarks = landmarksFlat,
                handTracking = trackingResult,
                bufferSize = sequenceBuffer.getCurrentSize()
            )
        }

        // Stage 4: Gesture classification (on NPU)
        val sequence = sequenceBuffer.getSequence()
        val prediction = gestureClassifier.predictWithConfidence(sequence)

        if (prediction == null) {
            // No confident prediction
            return GestureResultONNX(
                gestureResult = null,
                landmarks = landmarksFlat,
                handTracking = trackingResult,
                bufferSize = sequenceBuffer.getCurrentSize()
            )
        }

        val (gesture, confidence, probabilities) = prediction

        // Create gesture result
        val gestureResult = GestureResult(
            gesture = gesture,
            confidence = confidence,
            allProbabilities = probabilities,
            mediaPipeTimeMs = trackingResult.totalTimeMs,  // Hand tracking time
            onnxTimeMs = 0f,  // Will be set below
            totalTimeMs = 0f   // Will be set below
        )

        return GestureResultONNX(
            gestureResult = gestureResult,
            landmarks = landmarksFlat,
            handTracking = trackingResult,
            bufferSize = sequenceBuffer.getCurrentSize()
        )
    }

    /**
     * Flatten 2D landmarks to 1D array
     */
    private fun flattenLandmarks(landmarks: Array<FloatArray>): FloatArray {
        val flat = FloatArray(63)
        for (i in 0 until 21) {
            flat[i * 3] = landmarks[i][0]
            flat[i * 3 + 1] = landmarks[i][1]
            flat[i * 3 + 2] = landmarks[i][2]
        }
        return flat
    }

    /**
     * Get last detected landmarks
     */
    fun getLastLandmarks(): FloatArray? = lastLandmarks

    /**
     * Get buffer size
     */
    fun getBufferSize(): Int = sequenceBuffer.getCurrentSize()

    /**
     * Get accelerator info
     */
    fun getHandTrackingAccelerator(): String {
        return "NNAPI (NPU/GPU)"  // Both detector and landmarks use NNAPI
    }

    fun getGestureAccelerator(): String {
        return gestureClassifier.getActualAccelerator()
    }

    /**
     * Reset tracking state
     */
    fun reset() {
        handTracker.resetTracking()
        sequenceBuffer.clear()
        lastLandmarks = null
    }

    fun close() {
        handTracker.close()
        gestureClassifier.close()
        Log.d(TAG, "Gesture Recognizer ONNX closed")
    }
}

/**
 * Complete gesture recognition result with ONNX tracking
 */
data class GestureResultONNX(
    val gestureResult: GestureResult?,      // Gesture classification result (may be null if buffer not full)
    val landmarks: FloatArray,              // Raw landmarks (63 values)
    val handTracking: HandTrackingResult,   // Hand tracking details
    val bufferSize: Int                     // Current buffer size
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as GestureResultONNX

        if (gestureResult != other.gestureResult) return false
        if (!landmarks.contentEquals(other.landmarks)) return false
        if (handTracking != other.handTracking) return false
        if (bufferSize != other.bufferSize) return false

        return true
    }

    override fun hashCode(): Int {
        var result = gestureResult?.hashCode() ?: 0
        result = 31 * result + landmarks.contentHashCode()
        result = 31 * result + handTracking.hashCode()
        result = 31 * result + bufferSize
        return result
    }
}