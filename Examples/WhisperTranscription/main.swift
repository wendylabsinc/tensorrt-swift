// WhisperTranscription - Audio transcription with Whisper encoder
//
// This example demonstrates:
// 1. Audio preprocessing (mel spectrogram simulation)
// 2. Running Whisper-style encoder inference
// 3. Decoding tokens to text
// 4. Handling variable-length audio
//
// Note: Uses simulated model since real weights are large
//
// Run with: ./scripts/swiftw run WhisperTranscription
import TensorRT
import FoundationEssentials

@main
struct WhisperTranscription {
    // Whisper-style configuration
    static let sampleRate = 16000
    static let chunkLength = 30  // seconds
    static let nMels = 80
    static let nFFT = 400
    static let hopLength = 160
    static let maxFrames = 3000  // 30 seconds at 10ms per frame

    // Simulated vocabulary (subset of Whisper's actual vocabulary)
    static let vocabulary = [
        "<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>",
        "<|endoftext|>", " ", "the", "a", "to", "is", "and", "of", "in", "it",
        "that", "for", "you", "was", "on", "are", "with", "this", "be", "have",
        "from", "or", "one", "had", "by", "but", "not", "what", "all", "were",
        "we", "when", "your", "can", "said", "there", "use", "an", "each",
        "hello", "world", "swift", "programming", "audio", "speech", "recognition",
        "transcription", "artificial", "intelligence", "machine", "learning",
        "neural", "network", "model", "inference", "gpu", "cuda", "tensorrt"
    ]

    static func main() async throws {
        print("=== Whisper Transcription Example ===\n")
        print("This example demonstrates audio transcription with a Whisper-style model.")
        print("Using simulated audio and model for demonstration.\n")

        // Step 1: Build encoder model
        print("1. Building Whisper encoder model...")
        let encoderOutputSize = 512  // Encoder hidden dimension
        let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: encoderOutputSize)
        let engine = try TensorRTRuntime().deserializeEngine(from: plan)
        let context = try engine.makeExecutionContext()
        print("   Encoder: \(nMels) mel channels, \(maxFrames) max frames")
        print("   Output: \(encoderOutputSize) hidden dimensions")

        // Step 2: Generate synthetic audio
        print("\n2. Generating synthetic audio...")
        let audioDurations = [5.0, 10.0, 30.0]  // seconds

        for duration in audioDurations {
            let numSamples = Int(duration * Double(sampleRate))
            let audio = generateSyntheticAudio(numSamples: numSamples)
            print("   Generated \(duration)s audio: \(audio.count) samples")
        }

        // Step 3: Audio preprocessing pipeline
        print("\n3. Audio preprocessing pipeline...")
        let testAudio = generateSyntheticAudio(numSamples: sampleRate * 10)  // 10 seconds

        let preprocessStart = ContinuousClock.now

        // a) Pad/trim to chunk length
        let paddedAudio = padOrTrimAudio(testAudio, targetLength: sampleRate * chunkLength)
        print("   a) Padded to \(chunkLength)s: \(paddedAudio.count) samples")

        // b) Compute mel spectrogram (simulated)
        let melSpec = computeMelSpectrogram(paddedAudio)
        print("   b) Mel spectrogram: \(melSpec.count / nMels) frames x \(nMels) mels")

        // c) Normalize
        let normalized = normalizeMelSpectrogram(melSpec)
        print("   c) Normalized: min=\(formatDouble(Double(normalized.min() ?? 0), decimals: 2)), max=\(formatDouble(Double(normalized.max() ?? 0), decimals: 2))")

        let preprocessDuration = ContinuousClock.now - preprocessStart
        print("   Preprocessing time: \(preprocessDuration)")

        // Step 4: Run encoder
        print("\n4. Running encoder inference...")
        let inferenceStart = ContinuousClock.now

        // In real impl, this would be the actual encoder inference
        var encoderOutput: [Float] = []
        try await context.enqueueF32(
            inputName: "input",
            input: Array(normalized.prefix(encoderOutputSize)),
            outputName: "output",
            output: &encoderOutput
        )

        let inferenceDuration = ContinuousClock.now - inferenceStart
        print("   Encoder output shape: [\(encoderOutput.count)]")
        print("   Inference time: \(inferenceDuration)")

        // Step 5: Decode tokens (simulated)
        print("\n5. Decoding transcription...")
        let decodeStart = ContinuousClock.now

        let tokens = simulateDecoding(encoderOutput: encoderOutput)
        let transcription = decodeTokens(tokens)

        let decodeDuration = ContinuousClock.now - decodeStart
        print("   Decoded \(tokens.count) tokens")
        print("   Decode time: \(decodeDuration)")

        // Step 6: Display transcription
        print("\n6. Transcription Result:")
        print("   ┌" + String(repeating: "─", count: 60) + "┐")
        print("   │ " + transcription.padding(toLength: 58, withPad: " ", startingAt: 0) + " │")
        print("   └" + String(repeating: "─", count: 60) + "┘")

        // Step 7: Batch transcription demo
        print("\n7. Batch Transcription Demo:")
        let audioClips = [
            generateSyntheticAudio(numSamples: sampleRate * 5),
            generateSyntheticAudio(numSamples: sampleRate * 8),
            generateSyntheticAudio(numSamples: sampleRate * 12),
        ]

        let batchStart = ContinuousClock.now

        for (i, clip) in audioClips.enumerated() {
            let padded = padOrTrimAudio(clip, targetLength: sampleRate * chunkLength)
            let mel = computeMelSpectrogram(padded)
            let norm = normalizeMelSpectrogram(mel)

            var output: [Float] = []
            try await context.enqueueF32(
                inputName: "input",
                input: Array(norm.prefix(encoderOutputSize)),
                outputName: "output",
                output: &output
            )

            let tokens = simulateDecoding(encoderOutput: output)
            let text = decodeTokens(tokens)

            let duration = Double(clip.count) / Double(sampleRate)
            print("   Clip \(i + 1) (\(formatDouble(duration, decimals: 1))s): \"\(text.prefix(40))...\"")
        }

        let batchDuration = ContinuousClock.now - batchStart
        print("   Total batch time: \(batchDuration)")

        // Step 8: Streaming transcription pattern
        print("\n8. Streaming Transcription Pattern:")
        print("   Simulating real-time audio processing...")

        let streamChunkMs = 100  // 100ms chunks
        let streamChunkSamples = sampleRate * streamChunkMs / 1000
        let totalStreamDuration = 3.0  // 3 seconds
        let numStreamChunks = Int(totalStreamDuration * 1000) / streamChunkMs

        var audioBuffer: [Float] = []
        let streamStart = ContinuousClock.now

        for chunk in 0..<numStreamChunks {
            // Simulate receiving audio chunk
            let newAudio = generateSyntheticAudio(numSamples: streamChunkSamples)
            audioBuffer.append(contentsOf: newAudio)

            // Process when we have enough audio (e.g., every second)
            if audioBuffer.count >= sampleRate {
                let processingChunk = Array(audioBuffer.prefix(sampleRate))
                audioBuffer = Array(audioBuffer.dropFirst(sampleRate))

                let mel = computeMelSpectrogram(processingChunk)
                let norm = normalizeMelSpectrogram(mel)

                var output: [Float] = []
                try await context.enqueueF32(
                    inputName: "input",
                    input: Array(norm.prefix(encoderOutputSize)),
                    outputName: "output",
                    output: &output
                )

                print("   Chunk \(chunk + 1): processed \(processingChunk.count) samples")
            }
        }

        let streamDuration = ContinuousClock.now - streamStart
        let rtf = durationToSeconds(streamDuration) / totalStreamDuration
        print("   Stream processing time: \(streamDuration)")
        print("   Real-time factor: \(formatDouble(rtf, decimals: 3))x")

        // Step 9: Performance summary
        print("\n9. Performance Summary:")
        print("   ┌─────────────────────┬────────────────────┐")
        print("   │ Stage               │ Time               │")
        print("   ├─────────────────────┼────────────────────┤")
        print("   │ Preprocessing       │ \(formatDuration(preprocessDuration).padding(toLength: 18, withPad: " ", startingAt: 0)) │")
        print("   │ Encoder Inference   │ \(formatDuration(inferenceDuration).padding(toLength: 18, withPad: " ", startingAt: 0)) │")
        print("   │ Token Decoding      │ \(formatDuration(decodeDuration).padding(toLength: 18, withPad: " ", startingAt: 0)) │")
        print("   └─────────────────────┴────────────────────┘")

        print("\n=== Whisper Transcription Complete ===")
    }

    /// Generates synthetic audio samples
    static func generateSyntheticAudio(numSamples: Int) -> [Float] {
        (0..<numSamples).map { i in
            // Mix of frequencies to simulate speech
            let t = Float(i) / Float(sampleRate)
            let f1 = sinApprox(2 * Float.pi * 200 * t)  // Fundamental
            let f2 = 0.5 * sinApprox(2 * Float.pi * 400 * t)  // Harmonic
            let f3 = 0.3 * sinApprox(2 * Float.pi * 800 * t)  // Higher harmonic
            let noise = Float.random(in: -0.1...0.1)
            return 0.3 * (f1 + f2 + f3 + noise)
        }
    }

    /// Taylor series approximation of sin
    static func sinApprox(_ x: Float) -> Float {
        // Normalize to [-pi, pi]
        var angle = x
        while angle > Float.pi { angle -= 2 * Float.pi }
        while angle < -Float.pi { angle += 2 * Float.pi }

        // Taylor series: sin(x) = x - x^3/3! + x^5/5! - ...
        var result: Float = 0
        var term = angle
        var sign: Float = 1

        for i in stride(from: 1, through: 11, by: 2) {
            result += sign * term
            term *= angle * angle / Float((i + 1) * (i + 2))
            sign *= -1
        }
        return result
    }

    /// Natural logarithm approximation
    static func log(_ x: Float) -> Float {
        guard x > 0 else { return -.infinity }
        // Use series expansion: ln(x) = 2 * sum((y^(2n+1))/(2n+1)) where y = (x-1)/(x+1)
        let y = (x - 1) / (x + 1)
        var result: Float = 0
        var term = y
        for n in 0..<20 {
            result += term / Float(2 * n + 1)
            term *= y * y
        }
        return 2 * result
    }

    /// Square root approximation using Newton's method
    static func sqrt(_ x: Float) -> Float {
        guard x >= 0 else { return .nan }
        guard x > 0 else { return 0 }

        var guess = x / 2
        for _ in 0..<10 {
            guess = (guess + x / guess) / 2
        }
        return guess
    }

    /// Pads or trims audio to target length
    static func padOrTrimAudio(_ audio: [Float], targetLength: Int) -> [Float] {
        if audio.count >= targetLength {
            return Array(audio.prefix(targetLength))
        } else {
            return audio + [Float](repeating: 0, count: targetLength - audio.count)
        }
    }

    /// Computes mel spectrogram (simplified simulation)
    static func computeMelSpectrogram(_ audio: [Float]) -> [Float] {
        let numFrames = min(audio.count / hopLength, maxFrames)
        var melSpec = [Float](repeating: 0, count: numFrames * nMels)

        for frame in 0..<numFrames {
            let start = frame * hopLength
            let frameData = Array(audio[start..<min(start + nFFT, audio.count)])

            // Simplified "FFT" - just compute energy in bands
            for mel in 0..<nMels {
                var energy: Float = 0
                let bandStart = mel * frameData.count / nMels
                let bandEnd = (mel + 1) * frameData.count / nMels
                for i in bandStart..<bandEnd {
                    if i < frameData.count {
                        energy += frameData[i] * frameData[i]
                    }
                }
                melSpec[frame * nMels + mel] = log(max(energy, 1e-10))
            }
        }

        return melSpec
    }

    /// Normalizes mel spectrogram
    static func normalizeMelSpectrogram(_ melSpec: [Float]) -> [Float] {
        let mean = melSpec.reduce(0, +) / Float(melSpec.count)
        var variance: Float = 0
        for val in melSpec {
            variance += (val - mean) * (val - mean)
        }
        variance /= Float(melSpec.count)
        let std = sqrt(variance + 1e-6)

        return melSpec.map { ($0 - mean) / std }
    }

    /// Simulates token decoding from encoder output
    static func simulateDecoding(encoderOutput: [Float]) -> [Int] {
        // Generate plausible token sequence based on encoder output
        var tokens: [Int] = [0, 1, 2, 3]  // Start tokens

        // Simulate decoding based on output values
        let sum = encoderOutput.reduce(0, +)
        let seed = Int(abs(sum * 1000)) % 100

        let words = [44, 45, 46, 47, 48, 49, 50, 51]  // "hello", "world", etc.
        for i in 0..<8 {
            let idx = (seed + i * 7) % words.count
            tokens.append(words[idx])
        }

        tokens.append(4)  // End token
        return tokens
    }

    /// Decodes token IDs to text
    static func decodeTokens(_ tokens: [Int]) -> String {
        var text = ""
        for token in tokens {
            if token >= 0 && token < vocabulary.count {
                let word = vocabulary[token]
                if !word.hasPrefix("<|") {
                    text += word
                }
            }
        }
        return text.trimmingCharacters(in: .whitespaces)
    }

    static func durationToSeconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    static func formatDouble(_ value: Double, decimals: Int) -> String {
        if decimals <= 0 { return String(Int(value.rounded())) }
        var multiplier = 1.0
        for _ in 0..<decimals { multiplier *= 10.0 }
        let rounded = (value * multiplier).rounded() / multiplier
        let intPart = Int(rounded)
        let fracPart = abs(Int((rounded - Double(intPart)) * multiplier))
        return "\(intPart).\(fracPart)"
    }

    static func formatDuration(_ duration: Duration) -> String {
        let us = durationToSeconds(duration) * 1_000_000
        if us < 1000 { return "\(formatDouble(us, decimals: 0)) µs" }
        else if us < 1_000_000 { return "\(formatDouble(us / 1000, decimals: 2)) ms" }
        else { return "\(formatDouble(us / 1_000_000, decimals: 2)) s" }
    }
}

extension String {
    func padding(toLength length: Int, withPad padString: String, startingAt: Int) -> String {
        if self.count >= length { return String(self.prefix(length)) }
        var result = self
        while result.count < length { result += padString }
        return String(result.prefix(length))
    }

    func trimmingCharacters(in characterSet: CharacterSet) -> String {
        var result = self
        while let first = result.first, characterSet.contains(first.unicodeScalars.first!) {
            result.removeFirst()
        }
        while let last = result.last, characterSet.contains(last.unicodeScalars.first!) {
            result.removeLast()
        }
        return result
    }
}

struct CharacterSet {
    static let whitespaces = CharacterSet(characters: [" ", "\t", "\n", "\r"])

    private let characters: Set<Character>

    init(characters: [Character]) {
        self.characters = Set(characters)
    }

    func contains(_ scalar: Unicode.Scalar) -> Bool {
        characters.contains(Character(scalar))
    }
}

extension Float {
    static var pi: Float { 3.14159265359 }
}
