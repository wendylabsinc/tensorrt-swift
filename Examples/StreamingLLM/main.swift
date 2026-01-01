// StreamingLLM - Token-by-token generation with KV-cache simulation
//
// This example demonstrates:
// 1. Autoregressive token generation
// 2. KV-cache pattern for efficient inference
// 3. Streaming output token-by-token
// 4. Managing context length and memory
//
// Note: Uses simulated model since real LLM weights are large
//
// Run with: ./scripts/swiftw run StreamingLLM
import TensorRT
import FoundationEssentials

#if canImport(TensorRTNative)
import TensorRTNative
#endif

@main
struct StreamingLLM {
    // Simulated LLM configuration
    static let vocabSize = 1000
    static let hiddenSize = 256
    static let maxSeqLen = 128
    static let numLayers = 4

    // Simple token vocabulary for demo
    static let vocabulary = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "must", "shall", "can", "need", "dare", "ought", "used", "to",
        "and", "but", "or", "nor", "for", "yet", "so", "although", "because", "since",
        "hello", "world", "swift", "tensorrt", "llm", "inference", "gpu", "cuda", "fast", "model"
    ]

    static func main() async throws {
        print("=== Streaming LLM Example ===\n")
        print("This example demonstrates token-by-token generation with KV-cache pattern.")
        print("Using a simulated model for demonstration purposes.\n")

#if canImport(TensorRTNative)
        // Step 1: Build a simple model (simulating LLM decoder layer)
        print("1. Building simulated LLM model...")
        let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: hiddenSize)
        let engine = try TensorRTRuntime().deserializeEngine(from: plan)
        let context = try engine.makeExecutionContext()
        print("   Model: \(numLayers) layers, hidden_size=\(hiddenSize)")

        // Step 2: Initialize KV-cache (simulated)
        print("\n2. Initializing KV-cache...")
        var kvCache = KVCache(layers: numLayers, maxSeqLen: maxSeqLen, hiddenSize: hiddenSize)
        print("   Cache capacity: \(maxSeqLen) tokens")
        print("   Memory per layer: \(hiddenSize * maxSeqLen * 2 * 4) bytes (K + V)")

        // Step 3: Encode prompt
        print("\n3. Processing prompt...")
        let prompt = "hello world swift tensorrt"
        let promptTokens = tokenize(prompt)
        print("   Prompt: \"\(prompt)\"")
        print("   Tokens: \(promptTokens)")

        // Prefill: process all prompt tokens at once
        let prefillStart = ContinuousClock.now
        var hiddenState = [Float](repeating: 0, count: hiddenSize)

        for (i, token) in promptTokens.enumerated() {
            // Embed token (simulated)
            let embedding = embedToken(token)

            // Run through model layers
            hiddenState = try await runModelLayer(
                context: context,
                input: embedding,
                position: i
            )

            // Update KV-cache
            kvCache.append(position: i, keys: embedding, values: hiddenState)
        }

        let prefillDuration = ContinuousClock.now - prefillStart
        print("   Prefill time: \(prefillDuration) (\(promptTokens.count) tokens)")
        print("   KV-cache filled: \(kvCache.currentLength)/\(maxSeqLen)")

        // Step 4: Autoregressive generation
        print("\n4. Generating tokens (streaming)...")
        let maxNewTokens = 20
        var generatedTokens: [Int] = []
        var totalGenerateTime: Duration = .zero

        print("   Output: ", terminator: "")

        for i in 0..<maxNewTokens {
            let position = promptTokens.count + i
            let genStart = ContinuousClock.now

            // Run model with current hidden state
            let output = try await runModelLayer(
                context: context,
                input: hiddenState,
                position: position
            )

            // Sample next token (simulated - just pick based on output)
            let nextToken = sampleToken(logits: output)
            generatedTokens.append(nextToken)

            // Update hidden state for next iteration
            hiddenState = output

            // Update KV-cache
            let tokenEmbedding = embedToken(nextToken)
            kvCache.append(position: position, keys: tokenEmbedding, values: output)

            let genDuration = ContinuousClock.now - genStart
            totalGenerateTime += genDuration

            // Stream output (print token as soon as it's generated)
            let tokenStr = detokenize([nextToken])
            print(tokenStr, terminator: " ")
            fflush(nil)  // Force output

            // Check for end token (simulated)
            if nextToken == 0 {
                break
            }
        }

        print("\n")

        // Step 5: Generation statistics
        print("5. Generation Statistics:")
        let tokensGenerated = generatedTokens.count
        let avgTokenTime = totalGenerateTime / tokensGenerated
        let tokensPerSec = Double(tokensGenerated) / durationToSeconds(totalGenerateTime)

        print("   Tokens generated: \(tokensGenerated)")
        print("   Total generate time: \(totalGenerateTime)")
        print("   Avg time per token: \(avgTokenTime)")
        print("   Throughput: \(formatDouble(tokensPerSec, decimals: 1)) tokens/sec")
        print("   Final KV-cache usage: \(kvCache.currentLength)/\(maxSeqLen)")

        // Step 6: Demonstrate KV-cache reuse
        print("\n6. KV-cache reuse demo (continuation)...")
        print("   Generating 5 more tokens from cached context...")

        let continuationStart = ContinuousClock.now
        for i in 0..<5 {
            let position = promptTokens.count + generatedTokens.count + i

            let output = try await runModelLayer(
                context: context,
                input: hiddenState,
                position: position
            )

            let nextToken = sampleToken(logits: output)
            hiddenState = output

            let tokenStr = detokenize([nextToken])
            print("   Token \(i + 1): \(tokenStr)")
        }
        let continuationDuration = ContinuousClock.now - continuationStart
        print("   Continuation time: \(continuationDuration)")

        // Step 7: Memory analysis
        print("\n7. Memory Analysis:")
        let kvCacheBytes = numLayers * maxSeqLen * hiddenSize * 2 * 4  // K + V, float32
        let hiddenStateBytes = hiddenSize * 4
        print("   KV-cache size: \(kvCacheBytes / 1024) KB")
        print("   Hidden state size: \(hiddenStateBytes) bytes")
        print("   Total inference memory: ~\((kvCacheBytes + hiddenStateBytes) / 1024) KB")

        // Step 8: Full output
        print("\n8. Full Generated Text:")
        let fullPrompt = prompt
        let fullGeneration = detokenize(generatedTokens)
        print("   \"\(fullPrompt) \(fullGeneration)\"")

        print("\n=== Streaming LLM Complete ===")

#else
        print("This example requires TensorRTNative (Linux with TensorRT)")
#endif
    }

    /// Simulated KV-cache
    struct KVCache {
        let layers: Int
        let maxSeqLen: Int
        let hiddenSize: Int
        var currentLength: Int = 0

        // In real implementation, these would be device buffers
        var keys: [[Float]]
        var values: [[Float]]

        init(layers: Int, maxSeqLen: Int, hiddenSize: Int) {
            self.layers = layers
            self.maxSeqLen = maxSeqLen
            self.hiddenSize = hiddenSize
            self.keys = Array(repeating: [Float](repeating: 0, count: maxSeqLen * hiddenSize), count: layers)
            self.values = Array(repeating: [Float](repeating: 0, count: maxSeqLen * hiddenSize), count: layers)
        }

        mutating func append(position: Int, keys newKeys: [Float], values newValues: [Float]) {
            guard position < maxSeqLen else { return }
            currentLength = max(currentLength, position + 1)
            // In real impl, copy to device buffer at position offset
        }
    }

    /// Tokenize text into token IDs
    static func tokenize(_ text: String) -> [Int] {
        let words = text.lowercased().split(separator: " ").map(String.init)
        return words.compactMap { word in
            vocabulary.firstIndex(of: word) ?? (word.hashValue % vocabSize)
        }
    }

    /// Convert token IDs back to text
    static func detokenize(_ tokens: [Int]) -> String {
        tokens.map { token in
            token < vocabulary.count ? vocabulary[token] : "<unk>"
        }.joined(separator: " ")
    }

    /// Create embedding for token (simulated)
    static func embedToken(_ token: Int) -> [Float] {
        // Simple hash-based embedding
        var embedding = [Float](repeating: 0, count: hiddenSize)
        for i in 0..<hiddenSize {
            embedding[i] = Float((token * 17 + i * 31) % 1000) / 1000.0 - 0.5
        }
        return embedding
    }

    /// Run through model layer (simulated)
    static func runModelLayer(
        context: ExecutionContext,
        input: [Float],
        position: Int
    ) async throws -> [Float] {
        // Add positional encoding
        var posInput = input
        for i in 0..<min(input.count, hiddenSize) {
            posInput[i] += Float(position) * 0.01
        }

        // Run through TensorRT (identity for demo)
        var output: [Float] = []
        try await context.enqueueF32(
            inputName: "input",
            input: Array(posInput.prefix(hiddenSize)),
            outputName: "output",
            output: &output
        )

        return output
    }

    /// Sample next token from logits (simulated)
    static func sampleToken(logits: [Float]) -> Int {
        // Simple argmax with some randomness
        let sum = logits.reduce(0, +)
        let normalized = Int(abs(sum * 1000)) % vocabSize
        return normalized % vocabulary.count
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
}

func fflush(_ stream: Any?) {
    // Placeholder - Swift doesn't need explicit flushing for print
}
