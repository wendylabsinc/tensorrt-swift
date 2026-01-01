// DynamicBatching - Handle variable batch sizes at runtime
//
// This example demonstrates:
// 1. Building an engine with dynamic shapes
// 2. Reshaping inputs at runtime for different batch sizes
// 3. Processing variable-length batches efficiently
//
// Run with: ./scripts/swiftw run DynamicBatching
import TensorRT
import FoundationEssentials

@main
struct DynamicBatching {
    static func main() async throws {
        print("=== Dynamic Batching Example ===\n")

        // Configuration
        let featureSize = 8       // Features per sample
        let minBatch = 1
        let optBatch = 16
        let maxBatch = 64

        print("Configuration:")
        print("  Feature size: \(featureSize)")
        print("  Batch range: [\(minBatch), \(optBatch), \(maxBatch)]")

        // Step 1: Create a dynamic ONNX model
        print("\n1. Creating dynamic ONNX model...")
        let onnxURL = try createDynamicONNXModel()
        defer { try? FileManager.default.removeItem(at: onnxURL.deletingLastPathComponent()) }
        print("   Created temporary model at: \(onnxURL.lastPathComponent)")

        // Step 2: Build engine with optimization profile for dynamic batch
        print("\n2. Building engine with dynamic batch profile...")

        let profile = OptimizationProfile(
            name: "dynamic_batch",
            axes: [:],
            bindingRanges: [
                "input": .init(
                    min: TensorShape([minBatch]),
                    optimal: TensorShape([optBatch]),
                    max: TensorShape([maxBatch])
                )
            ]
        )

        let runtime = TensorRTRuntime()
        let engine = try runtime.buildEngine(
            onnxURL: onnxURL,
            options: EngineBuildOptions(
                precision: [.fp32],
                profiles: [profile]
            )
        )

        print("   Engine built successfully")
        print("   Input shape: \(engine.description.inputs[0].descriptor.shape.dimensions) (dynamic)")

        // Step 3: Create execution context
        print("\n3. Creating execution context...")
        let context = try engine.makeExecutionContext()

        // Step 4: Process batches of different sizes
        print("\n4. Processing variable batch sizes...")

        let batchSizes = [1, 4, 8, 16, 32, 64]
        var results: [(batchSize: Int, latency: Duration, throughput: Double)] = []

        for batchSize in batchSizes {
            // Reshape for this batch size
            try await context.reshape(bindings: ["input": TensorShape([batchSize])])

            // Generate input data
            let input: [Float] = (0..<batchSize).map { Float($0) }
            let inputData = input.withUnsafeBufferPointer { buffer in
                Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
            }

            let inputDesc = engine.description.inputs[0].descriptor
            let batch = InferenceBatch(inputs: [
                "input": TensorValue(descriptor: inputDesc, storage: .host(inputData))
            ])

            // Warm-up
            _ = try await context.enqueue(batch, synchronously: true)

            // Measure latency (average of 10 runs)
            let iterations = 10
            let start = ContinuousClock.now

            for _ in 0..<iterations {
                _ = try await context.enqueue(batch, synchronously: true)
            }

            let totalDuration = ContinuousClock.now - start
            let avgLatency = totalDuration / iterations
            let throughput = Double(batchSize * iterations) / durationToSeconds(totalDuration)

            results.append((batchSize, avgLatency, throughput))
            print("   Batch \(batchSize): latency=\(avgLatency), throughput=\(formatDouble(throughput, decimals: 1)) samples/sec")
        }

        // Step 5: Demonstrate on-the-fly batch switching
        print("\n5. On-the-fly batch switching demo...")

        let switchPattern = [8, 1, 32, 4, 16]
        print("   Pattern: \(switchPattern)")

        for (i, batchSize) in switchPattern.enumerated() {
            try await context.reshape(bindings: ["input": TensorShape([batchSize])])

            let input: [Float] = (0..<batchSize).map { Float($0) + Float(i * 100) }
            let inputData = input.withUnsafeBufferPointer { buffer in
                Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
            }

            let inputDesc = engine.description.inputs[0].descriptor
            let batch = InferenceBatch(inputs: [
                "input": TensorValue(descriptor: inputDesc, storage: .host(inputData))
            ])

            let result = try await context.enqueue(batch, synchronously: true)

            if let output = result.outputs["output"], case .host(let data) = output.storage {
                let outputCount = data.count / MemoryLayout<Float>.stride
                print("   Step \(i + 1): batch=\(batchSize), output_elements=\(outputCount)")
            }
        }

        // Step 6: Summary
        print("\n6. Performance Summary:")
        print("   ┌────────────┬─────────────────┬──────────────────┐")
        print("   │ Batch Size │ Avg Latency     │ Throughput       │")
        print("   ├────────────┼─────────────────┼──────────────────┤")
        for r in results {
            let latencyStr = formatDuration(r.latency)
            let throughputStr = formatDouble(r.throughput, decimals: 1)
            print("   │ \(String(r.batchSize).padding(toLength: 10, withPad: " ", startingAt: 0)) │ \(latencyStr.padding(toLength: 15, withPad: " ", startingAt: 0)) │ \(throughputStr.padding(toLength: 12, withPad: " ", startingAt: 0)) s/s │")
        }
        print("   └────────────┴─────────────────┴──────────────────┘")

        print("\n=== Dynamic Batching Complete ===")
    }

    /// Creates a minimal dynamic ONNX identity model
    static func createDynamicONNXModel() throws -> URL {
        // ONNX identity model with dynamic first dimension
        let onnxBase64 = "CAc6VAoZCgVpbnB1dBIGb3V0cHV0IghJZGVudGl0eRIQRHluSWRlbnRpdHlHcmFwaFoRCgVpbnB1dBIICgYIARICCgBiEgoGb3V0cHV0EggKBggBEgIKAEIECgAQDQ=="

        guard let onnxData = Data(base64Encoded: onnxBase64) else {
            throw TensorRTError.runtimeUnavailable("Failed to decode ONNX model")
        }

        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tensorrt-dynamic-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)

        let onnxURL = tmpDir.appendingPathComponent("dynamic.onnx")
        try onnxData.write(to: onnxURL)

        return onnxURL
    }

    static func durationToSeconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    static func formatDouble(_ value: Double, decimals: Int) -> String {
        if decimals <= 0 {
            return String(Int(value.rounded()))
        }
        var multiplier = 1.0
        for _ in 0..<decimals { multiplier *= 10.0 }
        let rounded = (value * multiplier).rounded() / multiplier
        let intPart = Int(rounded)
        let fracPart = abs(Int((rounded - Double(intPart)) * multiplier))
        return "\(intPart).\(fracPart)"
    }

    static func formatDuration(_ duration: Duration) -> String {
        let us = durationToSeconds(duration) * 1_000_000
        if us < 1000 {
            return "\(formatDouble(us, decimals: 1)) µs"
        } else {
            return "\(formatDouble(us / 1000, decimals: 2)) ms"
        }
    }
}

extension String {
    func padding(toLength length: Int, withPad padString: String, startingAt padIndex: Int) -> String {
        if self.count >= length {
            return String(self.prefix(length))
        }
        var result = self
        while result.count < length {
            result += padString
        }
        return String(result.prefix(length))
    }
}
