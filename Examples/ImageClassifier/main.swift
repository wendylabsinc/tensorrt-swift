// ImageClassifier - End-to-end image classification pipeline
//
// This example demonstrates:
// 1. Image preprocessing (resize, normalize)
// 2. TensorRT inference with image data
// 3. Postprocessing (softmax, top-k extraction)
// 4. Complete classification pipeline
//
// Note: Uses a simulated model since we don't bundle real weights
//
// Run with: swift run ImageClassifier

import TensorRTLLM
import FoundationEssentials

@main
struct ImageClassifier {
    // ImageNet-style configuration
    static let imageSize = 224
    static let channels = 3
    static let numClasses = 1000

    // ImageNet normalization constants
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    // Sample class names (subset of ImageNet)
    static let classNames = [
        "tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead",
        "electric_ray", "stingray", "cock", "hen", "ostrich",
        "brambling", "goldfinch", "house_finch", "junco", "indigo_bunting",
        "robin", "bulbul", "jay", "magpie", "chickadee"
    ]

    static func main() async throws {
        print("=== Image Classifier Example ===\n")
        print("This example demonstrates an end-to-end image classification pipeline.")
        print("Using a simulated model (identity-based) for demonstration.\n")

        // Step 1: Create a simulated classification model
        print("1. Building classification model...")
        let inputSize = imageSize * imageSize * channels
        let plan = try TensorRTLLMSystem.buildIdentityEnginePlan(elementCount: inputSize)
        let engine = try TensorRTLLMRuntime().deserializeEngine(from: plan)
        print("   Model input: \(channels)x\(imageSize)x\(imageSize) = \(inputSize) floats")

        // Step 2: Create execution context
        print("\n2. Creating execution context...")
        let context = try engine.makeExecutionContext()

        // Step 3: Simulate loading an image
        print("\n3. Simulating image loading...")
        let rawImage = generateSyntheticImage(width: imageSize, height: imageSize, channels: channels)
        print("   Generated synthetic \(imageSize)x\(imageSize) RGB image")

        // Step 4: Preprocess the image
        print("\n4. Preprocessing image...")
        let preprocessStart = ContinuousClock.now
        let preprocessed = preprocessImage(rawImage, width: imageSize, height: imageSize)
        let preprocessDuration = ContinuousClock.now - preprocessStart
        print("   Applied normalization (mean subtraction, std division)")
        print("   Preprocessing time: \(preprocessDuration)")

        // Show some statistics
        let minVal = preprocessed.min() ?? 0
        let maxVal = preprocessed.max() ?? 0
        let avgVal = preprocessed.reduce(0, +) / Float(preprocessed.count)
        print("   Tensor stats: min=\(formatFloat(minVal)), max=\(formatFloat(maxVal)), mean=\(formatFloat(avgVal))")

        // Step 5: Run inference
        print("\n5. Running inference...")
        let inputData = preprocessed.withUnsafeBufferPointer { buffer in
            Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
        }

        let inputDesc = engine.description.inputs[0].descriptor
        let batch = InferenceBatch(inputs: [
            "input": TensorValue(descriptor: inputDesc, storage: .host(inputData))
        ])

        let inferenceStart = ContinuousClock.now
        let result = try await context.enqueue(batch, synchronously: true)
        let inferenceDuration = ContinuousClock.now - inferenceStart
        print("   Inference time: \(inferenceDuration)")

        // Step 6: Postprocess results
        print("\n6. Postprocessing results...")

        guard let outputValue = result.outputs["output"],
              case .host(let outputData) = outputValue.storage else {
            throw TensorRTLLMError.invalidBinding("Missing output")
        }

        var logits = [Float](repeating: 0, count: inputSize)
        logits.withUnsafeMutableBytes { outBytes in
            outputData.withUnsafeBytes { inBytes in
                outBytes.copyBytes(from: inBytes)
            }
        }

        // Simulate class logits by taking first numClasses values
        let classLogits = Array(logits.prefix(min(numClasses, logits.count)))

        // Apply softmax
        let postprocessStart = ContinuousClock.now
        let probabilities = softmax(classLogits)
        let postprocessDuration = ContinuousClock.now - postprocessStart
        print("   Applied softmax normalization")
        print("   Postprocessing time: \(postprocessDuration)")

        // Step 7: Get top-k predictions
        print("\n7. Top-5 Predictions:")
        let topK = getTopK(probabilities: probabilities, k: 5)

        for (rank, (index, prob)) in topK.enumerated() {
            let className = index < classNames.count ? classNames[index] : "class_\(index)"
            let bar = String(repeating: "█", count: Int(prob * 50))
            print("   \(rank + 1). \(className.padding(toLength: 20, withPad: " ", startingAt: 0)) \(formatFloat(prob * 100))% \(bar)")
        }

        // Step 8: Batch classification demo
        print("\n8. Batch classification demo (5 images)...")
        let batchImages = (0..<5).map { i in
            generateSyntheticImage(width: imageSize, height: imageSize, channels: channels, seed: i)
        }

        let batchStart = ContinuousClock.now
        for (i, image) in batchImages.enumerated() {
            let processed = preprocessImage(image, width: imageSize, height: imageSize)
            let data = processed.withUnsafeBufferPointer { buffer in
                Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
            }

            let batch = InferenceBatch(inputs: [
                "input": TensorValue(descriptor: inputDesc, storage: .host(data))
            ])
            _ = try await context.enqueue(batch, synchronously: true)
            print("   Image \(i + 1): classified")
        }
        let batchDuration = ContinuousClock.now - batchStart

        print("   Total batch time: \(batchDuration)")
        print("   Average per image: \(batchDuration / 5)")

        // Step 9: Summary
        print("\n9. Pipeline Summary:")
        print("   ┌─────────────────────┬────────────────────┐")
        print("   │ Stage               │ Time               │")
        print("   ├─────────────────────┼────────────────────┤")
        print("   │ Preprocessing       │ \(formatDuration(preprocessDuration).padding(toLength: 18, withPad: " ", startingAt: 0)) │")
        print("   │ Inference           │ \(formatDuration(inferenceDuration).padding(toLength: 18, withPad: " ", startingAt: 0)) │")
        print("   │ Postprocessing      │ \(formatDuration(postprocessDuration).padding(toLength: 18, withPad: " ", startingAt: 0)) │")
        print("   └─────────────────────┴────────────────────┘")

        print("\n=== Image Classifier Complete ===")
    }

    /// Generates a synthetic image with values in [0, 255]
    static func generateSyntheticImage(width: Int, height: Int, channels: Int, seed: Int = 42) -> [UInt8] {
        var image = [UInt8]()
        image.reserveCapacity(width * height * channels)

        for y in 0..<height {
            for x in 0..<width {
                for c in 0..<channels {
                    // Generate pseudo-random pattern
                    let value = UInt8((x + y + c + seed * 17) % 256)
                    image.append(value)
                }
            }
        }

        return image
    }

    /// Preprocesses image: converts to float, normalizes with ImageNet mean/std
    static func preprocessImage(_ image: [UInt8], width: Int, height: Int) -> [Float] {
        var output = [Float]()
        output.reserveCapacity(image.count)

        // Convert to float and normalize
        for (i, pixel) in image.enumerated() {
            let channelIdx = i % channels
            let normalized = (Float(pixel) / 255.0 - mean[channelIdx]) / std[channelIdx]
            output.append(normalized)
        }

        return output
    }

    /// Applies softmax normalization
    static func softmax(_ logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }

        // Subtract max for numerical stability
        let maxVal = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxVal) }
        let sumExp = exps.reduce(0, +)

        return exps.map { $0 / sumExp }
    }

    /// Returns top-k indices and probabilities
    static func getTopK(probabilities: [Float], k: Int) -> [(index: Int, probability: Float)] {
        let indexed = probabilities.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 > $1.1 }
        return Array(sorted.prefix(k))
    }

    static func exp(_ x: Float) -> Float {
        var result: Float = 1.0
        var term: Float = 1.0
        let clampedX = Swift.min(Swift.max(x, -20), 20)  // Clamp to avoid overflow

        for i in 1...20 {
            term *= clampedX / Float(i)
            result += term
        }
        return result
    }

    static func formatFloat(_ value: Float) -> String {
        let intPart = Int(value)
        let fracPart = abs(Int((value - Float(intPart)) * 100))
        let fracStr = fracPart < 10 ? "0\(fracPart)" : "\(fracPart)"
        if value < 0 && intPart == 0 {
            return "-0.\(fracStr)"
        }
        return "\(intPart).\(fracStr)"
    }

    static func formatDuration(_ duration: Duration) -> String {
        let us = durationToSeconds(duration) * 1_000_000
        if us < 1000 { return "\(formatFloat(Float(us))) µs" }
        else { return "\(formatFloat(Float(us / 1000))) ms" }
    }

    static func durationToSeconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}

extension String {
    func padding(toLength length: Int, withPad padString: String, startingAt: Int) -> String {
        if self.count >= length { return String(self.prefix(length)) }
        var result = self
        while result.count < length { result += padString }
        return String(result.prefix(length))
    }
}
