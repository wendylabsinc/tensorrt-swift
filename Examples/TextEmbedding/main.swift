// TextEmbedding - Sentence transformer model for text embeddings
//
// This example demonstrates:
// 1. Text tokenization and encoding
// 2. Running a sentence transformer model
// 3. Computing cosine similarity between embeddings
// 4. Semantic search use case
//
// Note: Uses simulated model since real weights are large
//
// Run with: swift run TextEmbedding

import TensorRTLLM
import FoundationEssentials

@main
struct TextEmbedding {
    // Simulated model configuration (similar to all-MiniLM-L6-v2)
    static let vocabSize = 30522
    static let maxSeqLen = 128
    static let hiddenSize = 384
    static let embeddingDim = 384

    // Simple vocabulary for demonstration
    static let vocabulary: [String: Int] = [
        "[CLS]": 101, "[SEP]": 102, "[PAD]": 0, "[UNK]": 100,
        "the": 1996, "a": 1037, "is": 2003, "are": 2024, "this": 2023,
        "that": 2008, "it": 2009, "for": 2005, "on": 2006, "with": 2007,
        "hello": 7592, "world": 2088, "swift": 10806, "programming": 4730,
        "language": 2653, "fast": 3435, "efficient": 8114, "code": 3642,
        "computer": 3274, "science": 2671, "machine": 3698, "learning": 4083,
        "deep": 2784, "neural": 11345, "network": 2897, "tensor": 20483,
        "gpu": 2204, "cuda": 26407, "inference": 17718, "model": 2944,
        "transformer": 10938, "attention": 3086, "embedding": 11373, "vector": 5566,
        "semantic": 22441, "search": 3945, "similarity": 14668, "text": 3793,
        "natural": 3019, "processing": 6364, "nlp": 17953, "ai": 9932,
    ]

    static func main() async throws {
        print("=== Text Embedding Example ===\n")
        print("This example demonstrates semantic text embedding and similarity search.")
        print("Using a simulated sentence transformer model.\n")

        // Step 1: Build embedding model
        print("1. Building embedding model...")
        let plan = try TensorRTLLMSystem.buildIdentityEnginePlan(elementCount: embeddingDim)
        let engine = try TensorRTLLMRuntime().deserializeEngine(from: plan)
        let context = try engine.makeExecutionContext()
        print("   Model: hidden_size=\(hiddenSize), embedding_dim=\(embeddingDim)")

        // Step 2: Define document corpus
        print("\n2. Document corpus:")
        let documents = [
            "Swift is a fast and efficient programming language",
            "Machine learning models use neural networks",
            "GPU acceleration enables fast deep learning inference",
            "TensorRT optimizes model inference on NVIDIA GPUs",
            "Natural language processing transforms text into vectors",
            "Semantic search finds relevant documents using embeddings",
            "The transformer architecture revolutionized NLP",
            "CUDA enables parallel computing on graphics cards",
        ]

        for (i, doc) in documents.enumerated() {
            print("   [\(i)] \(doc)")
        }

        // Step 3: Compute embeddings for all documents
        print("\n3. Computing document embeddings...")
        var documentEmbeddings: [[Float]] = []

        let embedStart = ContinuousClock.now

        for (i, doc) in documents.enumerated() {
            let embedding = try await computeEmbedding(text: doc, context: context)
            documentEmbeddings.append(embedding)
            print("   Document \(i): embedded (\(embedding.count) dims)")
        }

        let embedDuration = ContinuousClock.now - embedStart
        print("   Total embedding time: \(embedDuration)")
        print("   Avg per document: \(embedDuration / documents.count)")

        // Step 4: Semantic search demo
        print("\n4. Semantic Search Demo:")
        let queries = [
            "fast GPU computing",
            "neural network training",
            "text understanding AI",
        ]

        for query in queries {
            print("\n   Query: \"\(query)\"")

            let queryEmbedding = try await computeEmbedding(text: query, context: context)

            // Compute similarities
            var similarities: [(index: Int, score: Float)] = []
            for (i, docEmb) in documentEmbeddings.enumerated() {
                let score = cosineSimilarity(queryEmbedding, docEmb)
                similarities.append((i, score))
            }

            // Sort by similarity
            similarities.sort { $0.score > $1.score }

            // Show top 3 results
            print("   Top 3 results:")
            for rank in 0..<min(3, similarities.count) {
                let (idx, score) = similarities[rank]
                print("   \(rank + 1). [\(formatDouble(Double(score), decimals: 3))] \(documents[idx])")
            }
        }

        // Step 5: Embedding space visualization (simplified)
        print("\n5. Embedding Space Analysis:")

        // Compute pairwise similarities
        print("   Pairwise document similarities (top pairs):")
        var pairs: [(i: Int, j: Int, score: Float)] = []

        for i in 0..<documents.count {
            for j in (i + 1)..<documents.count {
                let score = cosineSimilarity(documentEmbeddings[i], documentEmbeddings[j])
                pairs.append((i, j, score))
            }
        }

        pairs.sort { $0.score > $1.score }

        for k in 0..<min(5, pairs.count) {
            let (i, j, score) = pairs[k]
            print("   [\(formatDouble(Double(score), decimals: 3))] Doc \(i) <-> Doc \(j)")
        }

        // Step 6: Batch embedding performance
        print("\n6. Batch Embedding Performance:")
        let batchSizes = [1, 4, 8]

        for batchSize in batchSizes {
            let batchStart = ContinuousClock.now
            let iterations = 50

            for _ in 0..<iterations {
                for bi in 0..<batchSize {
                    _ = try await computeEmbedding(text: documents[bi % documents.count], context: context)
                }
            }

            let batchDuration = ContinuousClock.now - batchStart
            let throughput = Double(iterations * batchSize) / durationToSeconds(batchDuration)

            print("   Batch size \(batchSize): \(formatDouble(throughput, decimals: 1)) embeddings/sec")
        }

        // Step 7: Embedding statistics
        print("\n7. Embedding Statistics:")
        let allValues = documentEmbeddings.flatMap { $0 }
        let minVal = allValues.min() ?? 0
        let maxVal = allValues.max() ?? 0
        let avgVal = allValues.reduce(0, +) / Float(allValues.count)

        print("   Embedding dimensions: \(embeddingDim)")
        print("   Value range: [\(formatDouble(Double(minVal), decimals: 3)), \(formatDouble(Double(maxVal), decimals: 3))]")
        print("   Mean value: \(formatDouble(Double(avgVal), decimals: 4))")

        // Compute average embedding norm
        var totalNorm: Float = 0
        for emb in documentEmbeddings {
            totalNorm += sqrt(emb.map { $0 * $0 }.reduce(0, +))
        }
        let avgNorm = totalNorm / Float(documentEmbeddings.count)
        print("   Avg L2 norm: \(formatDouble(Double(avgNorm), decimals: 3))")

        print("\n=== Text Embedding Example Complete ===")
    }

    /// Tokenizes and encodes text for the model
    static func tokenize(_ text: String) -> [Int] {
        var tokens = [vocabulary["[CLS]"]!]

        let words = text.lowercased()
            .split(separator: " ")
            .map(String.init)

        for word in words {
            if let tokenId = vocabulary[word] {
                tokens.append(tokenId)
            } else {
                tokens.append(vocabulary["[UNK]"]!)
            }
        }

        tokens.append(vocabulary["[SEP]"]!)

        // Pad or truncate to maxSeqLen
        while tokens.count < maxSeqLen {
            tokens.append(vocabulary["[PAD]"]!)
        }

        return Array(tokens.prefix(maxSeqLen))
    }

    /// Computes embedding for text
    static func computeEmbedding(text: String, context: ExecutionContext) async throws -> [Float] {
        // Tokenize
        let tokens = tokenize(text)

        // Create mock embedding from tokens (in real impl, this would be model inference)
        var embedding = [Float](repeating: 0, count: embeddingDim)

        // Simple hash-based embedding for demonstration
        for (i, token) in tokens.enumerated() where token != 0 {
            for d in 0..<embeddingDim {
                embedding[d] += Float((token * 17 + d * 31 + i * 7) % 1000) / 1000.0 - 0.5
            }
        }

        // Run through TensorRT (identity for demo, but shows the pattern)
        var output: [Float] = []
        try await context.enqueueF32(
            inputName: "input",
            input: Array(embedding.prefix(embeddingDim)),
            outputName: "output",
            output: &output
        )

        // Normalize to unit vector
        let norm = sqrt(output.map { $0 * $0 }.reduce(0, +))
        if norm > 1e-6 {
            output = output.map { $0 / norm }
        }

        return output
    }

    /// Computes cosine similarity between two vectors
    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }

        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        let denom = sqrt(normA) * sqrt(normB)
        return denom > 1e-6 ? dot / denom : 0
    }

    static func durationToSeconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
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

    static func formatDouble(_ value: Double, decimals: Int) -> String {
        if decimals <= 0 { return String(Int(value.rounded())) }
        var multiplier = 1.0
        for _ in 0..<decimals { multiplier *= 10.0 }
        let rounded = (value * multiplier).rounded() / multiplier
        let intPart = Int(rounded)
        let fracPart = abs(Int((rounded - Double(intPart)) * multiplier))
        let fracStr = String(fracPart)
        let paddedFrac = String(repeating: "0", count: max(0, decimals - fracStr.count)) + fracStr
        return "\(intPart).\(paddedFrac)"
    }
}
