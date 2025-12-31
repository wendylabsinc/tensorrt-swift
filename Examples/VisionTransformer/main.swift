// VisionTransformer - ViT image classification
//
// This example demonstrates:
// 1. Image patch embedding for ViT
// 2. Vision Transformer inference pipeline
// 3. Attention visualization (conceptual)
// 4. Classification with confidence scores
//
// Note: Uses simulated model since real weights are large
//
// Run with: swift run VisionTransformer

import TensorRTLLM
import FoundationEssentials

@main
struct VisionTransformer {
    // ViT configuration (similar to ViT-B/16)
    static let imageSize = 224
    static let patchSize = 16
    static let numPatches = (imageSize / patchSize) * (imageSize / patchSize)  // 196
    static let hiddenSize = 768
    static let numHeads = 12
    static let numLayers = 12
    static let numClasses = 1000

    // ImageNet class names (subset)
    static let classNames = [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
        "electric ray", "stingray", "cock", "hen", "ostrich",
        "brambling", "goldfinch", "house finch", "junco", "indigo bunting",
        "robin", "bulbul", "jay", "magpie", "chickadee",
        "water ouzel", "kite", "bald eagle", "vulture", "great grey owl"
    ]

    static func main() async throws {
        print("=== Vision Transformer Example ===\n")
        print("This example demonstrates ViT-style image classification.")
        print("Using a simulated model for demonstration.\n")

        // Step 1: Build ViT model components
        print("1. Building ViT model...")
        let plan = try TensorRTLLMSystem.buildIdentityEnginePlan(elementCount: hiddenSize)
        let engine = try TensorRTLLMRuntime().deserializeEngine(from: plan)
        let context = try engine.makeExecutionContext()

        print("   Architecture:")
        print("   - Image size: \(imageSize)x\(imageSize)")
        print("   - Patch size: \(patchSize)x\(patchSize)")
        print("   - Num patches: \(numPatches) + 1 (CLS token)")
        print("   - Hidden size: \(hiddenSize)")
        print("   - Attention heads: \(numHeads)")
        print("   - Transformer layers: \(numLayers)")

        // Step 2: Generate test images
        print("\n2. Generating test images...")
        let testImages = [
            ("gradient", generateGradientImage()),
            ("checkerboard", generateCheckerboardImage()),
            ("noise", generateNoiseImage()),
        ]

        for (name, _) in testImages {
            print("   Generated: \(name) (\(imageSize)x\(imageSize))")
        }

        // Step 3: Image preprocessing
        print("\n3. Preprocessing pipeline...")
        let (_, testImage) = testImages[0]

        let preprocessStart = ContinuousClock.now

        // a) Resize (already correct size)
        print("   a) Image size: \(imageSize)x\(imageSize) ✓")

        // b) Normalize
        let normalized = normalizeImage(testImage)
        print("   b) Normalized with ImageNet mean/std")

        // c) Patch embedding
        let patches = extractPatches(normalized)
        print("   c) Extracted \(patches.count) patches of size \(patchSize)x\(patchSize)")

        // d) Flatten and project patches
        let patchEmbeddings = embedPatches(patches)
        print("   d) Patch embeddings: [\(patchEmbeddings.count / hiddenSize), \(hiddenSize)]")

        // e) Add CLS token and positional embeddings
        let inputEmbeddings = addPositionalEmbeddings(patchEmbeddings)
        print("   e) Added CLS token and positional embeddings")

        let preprocessDuration = ContinuousClock.now - preprocessStart
        print("   Preprocessing time: \(preprocessDuration)")

        // Step 4: Run transformer inference
        print("\n4. Running ViT inference...")
        let inferenceStart = ContinuousClock.now

        // Simulate multi-layer transformer
        var hidden = inputEmbeddings
        for layer in 0..<numLayers {
            // In real impl, this would be full transformer layer
            var layerOutput: [Float] = []
            try await context.enqueueF32(
                inputName: "input",
                input: Array(hidden.prefix(hiddenSize)),
                outputName: "output",
                output: &layerOutput
            )
            hidden = layerOutput + Array(hidden.dropFirst(hiddenSize))

            if layer == 0 || layer == numLayers - 1 {
                print("   Layer \(layer + 1)/\(numLayers): processed")
            } else if layer == 1 {
                print("   ...")
            }
        }

        let inferenceDuration = ContinuousClock.now - inferenceStart
        print("   Inference time: \(inferenceDuration)")

        // Step 5: Classification head
        print("\n5. Classification...")
        let classStart = ContinuousClock.now

        // Extract CLS token (first position)
        let clsToken = Array(hidden.prefix(hiddenSize))

        // Project to class logits (simulated)
        let logits = projectToClasses(clsToken)

        // Apply softmax
        let probabilities = softmax(logits)

        let classDuration = ContinuousClock.now - classStart
        print("   Classification time: \(classDuration)")

        // Step 6: Display results
        print("\n6. Classification Results:")
        let topK = getTopK(probabilities, k: 5)

        print("   ┌─────┬─────────────────────┬────────────┬──────────────────────┐")
        print("   │ Rank│ Class               │ Confidence │ Bar                  │")
        print("   ├─────┼─────────────────────┼────────────┼──────────────────────┤")

        for (rank, (idx, prob)) in topK.enumerated() {
            let className = idx < classNames.count ? classNames[idx] : "class_\(idx)"
            let classStr = className.padding(toLength: 19, withPad: " ", startingAt: 0)
            let probStr = (formatDouble(Double(prob) * 100, decimals: 1) + "%").padding(toLength: 10, withPad: " ", startingAt: 0)
            let barLen = Int(prob * 20)
            let bar = String(repeating: "█", count: barLen).padding(toLength: 20, withPad: " ", startingAt: 0)

            print("   │ \(rank + 1)   │ \(classStr) │ \(probStr) │ \(bar) │")
        }

        print("   └─────┴─────────────────────┴────────────┴──────────────────────┘")

        // Step 7: Attention visualization (conceptual)
        print("\n7. Attention Analysis (Conceptual):")
        let attentionWeights = simulateAttentionWeights()

        print("   CLS token attention to patches (top 5):")
        let sortedAttn = attentionWeights.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(5)

        for (patchIdx, weight) in sortedAttn {
            let row = patchIdx / (imageSize / patchSize)
            let col = patchIdx % (imageSize / patchSize)
            print("   - Patch (\(row), \(col)): \(formatDouble(Double(weight), decimals: 3))")
        }

        // Step 8: Batch classification
        print("\n8. Batch Classification:")
        let batchStart = ContinuousClock.now

        for (name, image) in testImages {
            let norm = normalizeImage(image)
            let patches = extractPatches(norm)
            let embeds = embedPatches(patches)
            let input = addPositionalEmbeddings(embeds)

            var output: [Float] = []
            try await context.enqueueF32(
                inputName: "input",
                input: Array(input.prefix(hiddenSize)),
                outputName: "output",
                output: &output
            )

            let logits = projectToClasses(output)
            let probs = softmax(logits)
            let (topIdx, topProb) = probs.enumerated().max(by: { $0.element < $1.element })!
            let topClass = topIdx < classNames.count ? classNames[topIdx] : "class_\(topIdx)"

            print("   \(name.padding(toLength: 12, withPad: " ", startingAt: 0)): \(topClass) (\(formatDouble(Double(topProb) * 100, decimals: 1))%)")
        }

        let batchDuration = ContinuousClock.now - batchStart
        print("   Batch time: \(batchDuration)")

        // Step 9: Performance summary
        print("\n9. Performance Summary:")
        print("   ┌─────────────────────┬────────────────────┐")
        print("   │ Stage               │ Time               │")
        print("   ├─────────────────────┼────────────────────┤")
        print("   │ Preprocessing       │ \(formatDuration(preprocessDuration).padding(toLength: 18, withPad: " ", startingAt: 0)) │")
        print("   │ Transformer         │ \(formatDuration(inferenceDuration).padding(toLength: 18, withPad: " ", startingAt: 0)) │")
        print("   │ Classification      │ \(formatDuration(classDuration).padding(toLength: 18, withPad: " ", startingAt: 0)) │")
        print("   └─────────────────────┴────────────────────┘")

        // Step 10: ViT vs CNN comparison (conceptual)
        print("\n10. ViT Architecture Notes:")
        print("   ┌─────────────────────────────────────────────────────────────┐")
        print("   │ Advantages of Vision Transformers:                          │")
        print("   │ - Global attention from the first layer                     │")
        print("   │ - Better scaling with more data and compute                 │")
        print("   │ - Unified architecture for vision and language              │")
        print("   ├─────────────────────────────────────────────────────────────┤")
        print("   │ Considerations:                                             │")
        print("   │ - Requires more data than CNNs to train from scratch        │")
        print("   │ - Patch size affects resolution vs. computation trade-off   │")
        print("   │ - Positional embeddings are crucial for spatial awareness   │")
        print("   └─────────────────────────────────────────────────────────────┘")

        print("\n=== Vision Transformer Complete ===")
    }

    // Image generation functions
    static func generateGradientImage() -> [UInt8] {
        var image = [UInt8]()
        for y in 0..<imageSize {
            for x in 0..<imageSize {
                let r = UInt8(x * 255 / imageSize)
                let g = UInt8(y * 255 / imageSize)
                let b = UInt8(128)
                image.append(contentsOf: [r, g, b])
            }
        }
        return image
    }

    static func generateCheckerboardImage() -> [UInt8] {
        var image = [UInt8]()
        let squareSize = 28
        for y in 0..<imageSize {
            for x in 0..<imageSize {
                let isWhite = ((x / squareSize) + (y / squareSize)) % 2 == 0
                let value: UInt8 = isWhite ? 255 : 0
                image.append(contentsOf: [value, value, value])
            }
        }
        return image
    }

    static func generateNoiseImage() -> [UInt8] {
        (0..<imageSize * imageSize * 3).map { _ in UInt8.random(in: 0...255) }
    }

    // Preprocessing functions
    static func normalizeImage(_ image: [UInt8]) -> [Float] {
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]

        var normalized = [Float]()
        for (i, pixel) in image.enumerated() {
            let channel = i % 3
            let value = (Float(pixel) / 255.0 - mean[channel]) / std[channel]
            normalized.append(value)
        }
        return normalized
    }

    static func extractPatches(_ image: [Float]) -> [[Float]] {
        var patches: [[Float]] = []
        let patchesPerRow = imageSize / patchSize

        for py in 0..<patchesPerRow {
            for px in 0..<patchesPerRow {
                var patch = [Float]()
                for dy in 0..<patchSize {
                    for dx in 0..<patchSize {
                        let x = px * patchSize + dx
                        let y = py * patchSize + dy
                        let idx = (y * imageSize + x) * 3
                        patch.append(contentsOf: image[idx..<idx+3])
                    }
                }
                patches.append(patch)
            }
        }
        return patches
    }

    static func embedPatches(_ patches: [[Float]]) -> [Float] {
        var embeddings = [Float]()
        for patch in patches {
            // Simple linear projection (simulated)
            var embedding = [Float](repeating: 0, count: hiddenSize)
            for (i, val) in patch.enumerated() {
                embedding[i % hiddenSize] += val * 0.1
            }
            embeddings.append(contentsOf: embedding)
        }
        return embeddings
    }

    static func addPositionalEmbeddings(_ embeddings: [Float]) -> [Float] {
        // Add CLS token
        var clsToken = [Float](repeating: 0, count: hiddenSize)
        for i in 0..<hiddenSize {
            clsToken[i] = Float(i) * 0.01
        }

        var result = clsToken + embeddings

        // Add positional embeddings
        let numPositions = result.count / hiddenSize
        for pos in 0..<numPositions {
            for i in 0..<hiddenSize {
                let pe = sin(Float(pos) / pow(10000, Float(2 * i) / Float(hiddenSize)))
                result[pos * hiddenSize + i] += pe * 0.1
            }
        }

        return result
    }

    static func projectToClasses(_ clsToken: [Float]) -> [Float] {
        // Simulate classification head
        var logits = [Float](repeating: 0, count: numClasses)
        for i in 0..<numClasses {
            for (j, val) in clsToken.enumerated() {
                logits[i] += val * Float((i * 17 + j * 31) % 100 - 50) / 100.0
            }
        }
        return logits
    }

    static func softmax(_ logits: [Float]) -> [Float] {
        let maxVal = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxVal) }
        let sum = exps.reduce(0, +)
        return exps.map { $0 / sum }
    }

    static func exp(_ x: Float) -> Float {
        var result: Float = 1.0
        var term: Float = 1.0
        let clampedX = min(max(x, -20), 20)
        for i in 1...20 {
            term *= clampedX / Float(i)
            result += term
        }
        return result
    }

    static func sin(_ x: Float) -> Float {
        var result: Float = 0
        var term = x
        var sign: Float = 1
        for i in stride(from: 1, through: 15, by: 2) {
            result += sign * term
            term *= x * x / Float((i + 1) * (i + 2))
            sign *= -1
        }
        return result
    }

    static func pow(_ base: Float, _ exp: Float) -> Float {
        var result: Float = 1
        for _ in 0..<Int(exp) {
            result *= base
        }
        return result
    }

    static func getTopK(_ probs: [Float], k: Int) -> [(Int, Float)] {
        let indexed = probs.enumerated().map { ($0.offset, $0.element) }
        return Array(indexed.sorted { $0.1 > $1.1 }.prefix(k))
    }

    static func simulateAttentionWeights() -> [Float] {
        (0..<numPatches).map { _ in Float.random(in: 0...1) }
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
        else { return "\(formatDouble(us / 1000, decimals: 2)) ms" }
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
