// MultiProfile - Multiple optimization profiles for different workloads
//
// This example demonstrates:
// 1. Building an engine with multiple optimization profiles
// 2. Switching between profiles at runtime
// 3. Optimizing for different batch size ranges
//
// Run with: swift run MultiProfile

import TensorRTLLM
import FoundationEssentials

@main
struct MultiProfile {
    static func main() async throws {
        print("=== Multi-Profile Example ===\n")

        // Step 1: Define optimization profiles for different workloads
        print("1. Defining optimization profiles...")

        // Profile 0: Small batches (real-time, low latency)
        let smallBatchProfile = OptimizationProfile(
            name: "small_batch",
            axes: [:],
            bindingRanges: [
                "input": .init(
                    min: TensorShape([1]),
                    optimal: TensorShape([4]),
                    max: TensorShape([8])
                )
            ]
        )
        print("   Profile 0 (small_batch): [1, 4, 8]")

        // Profile 1: Medium batches (balanced)
        let mediumBatchProfile = OptimizationProfile(
            name: "medium_batch",
            axes: [:],
            bindingRanges: [
                "input": .init(
                    min: TensorShape([8]),
                    optimal: TensorShape([32]),
                    max: TensorShape([64])
                )
            ]
        )
        print("   Profile 1 (medium_batch): [8, 32, 64]")

        // Profile 2: Large batches (throughput optimized)
        let largeBatchProfile = OptimizationProfile(
            name: "large_batch",
            axes: [:],
            bindingRanges: [
                "input": .init(
                    min: TensorShape([64]),
                    optimal: TensorShape([128]),
                    max: TensorShape([256])
                )
            ]
        )
        print("   Profile 2 (large_batch): [64, 128, 256]")

        // Step 2: Create dynamic ONNX model
        print("\n2. Creating dynamic ONNX model...")
        let onnxURL = try createDynamicONNXModel()
        defer { try? FileManager.default.removeItem(at: onnxURL.deletingLastPathComponent()) }

        // Step 3: Build engine with all profiles
        print("\n3. Building engine with 3 profiles...")
        let buildStart = ContinuousClock.now

        let runtime = TensorRTLLMRuntime()
        let engine = try runtime.buildEngine(
            onnxURL: onnxURL,
            options: EngineBuildOptions(
                precision: [.fp32],
                profiles: [smallBatchProfile, mediumBatchProfile, largeBatchProfile]
            )
        )

        let buildDuration = ContinuousClock.now - buildStart
        print("   Build time: \(buildDuration)")
        print("   Profiles in engine: \(engine.description.profileNames.count)")

        // Step 4: Create execution context
        print("\n4. Creating execution context...")
        let context = try engine.makeExecutionContext()

        // Step 5: Benchmark each profile
        print("\n5. Benchmarking each profile...")

        let profileTests: [(name: String, index: String, batchSizes: [Int])] = [
            ("small_batch", "0", [1, 4, 8]),
            ("medium_batch", "1", [16, 32, 64]),
            ("large_batch", "2", [64, 128, 256])
        ]

        var allResults: [(profile: String, batchSize: Int, latency: Duration, throughput: Double)] = []

        for test in profileTests {
            print("\n   --- Profile: \(test.name) ---")

            // Switch to this profile
            try await context.setOptimizationProfile(named: test.index)

            for batchSize in test.batchSizes {
                // Reshape for this batch
                try await context.reshape(bindings: ["input": TensorShape([batchSize])])

                // Prepare input
                let input: [Float] = (0..<batchSize).map { Float($0) }
                let inputData = input.withUnsafeBufferPointer { buffer in
                    Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
                }

                let inputDesc = engine.description.inputs[0].descriptor
                let batch = InferenceBatch(inputs: [
                    "input": TensorValue(descriptor: inputDesc, storage: .host(inputData))
                ])

                // Warm-up
                for _ in 0..<3 {
                    _ = try await context.enqueue(batch, synchronously: true)
                }

                // Benchmark
                let iterations = 20
                let start = ContinuousClock.now

                for _ in 0..<iterations {
                    _ = try await context.enqueue(batch, synchronously: true)
                }

                let totalDuration = ContinuousClock.now - start
                let avgLatency = totalDuration / iterations
                let throughput = Double(batchSize * iterations) / durationToSeconds(totalDuration)

                allResults.append((test.name, batchSize, avgLatency, throughput))
                print("   Batch \(batchSize): \(formatDuration(avgLatency)), \(formatDouble(throughput, decimals: 0)) samples/sec")
            }
        }

        // Step 6: Demonstrate dynamic profile switching
        print("\n6. Dynamic profile switching scenario...")
        print("   Simulating workload changes:\n")

        let workloadScenario: [(profile: String, batch: Int, description: String)] = [
            ("0", 1, "Single request (real-time)"),
            ("0", 4, "Small batch (interactive)"),
            ("1", 32, "Medium batch (API serving)"),
            ("2", 128, "Large batch (batch processing)"),
            ("1", 16, "Back to medium (API)"),
            ("0", 1, "Single request again"),
        ]

        for (i, scenario) in workloadScenario.enumerated() {
            try await context.setOptimizationProfile(named: scenario.profile)
            try await context.reshape(bindings: ["input": TensorShape([scenario.batch])])

            let input: [Float] = (0..<scenario.batch).map { Float($0) }
            let inputData = input.withUnsafeBufferPointer { buffer in
                Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
            }

            let inputDesc = engine.description.inputs[0].descriptor
            let batch = InferenceBatch(inputs: [
                "input": TensorValue(descriptor: inputDesc, storage: .host(inputData))
            ])

            let start = ContinuousClock.now
            _ = try await context.enqueue(batch, synchronously: true)
            let latency = ContinuousClock.now - start

            print("   Step \(i + 1): \(scenario.description)")
            print("          Profile=\(scenario.profile), Batch=\(scenario.batch), Latency=\(formatDuration(latency))")
        }

        // Step 7: Summary
        print("\n7. Summary:")
        print("   ┌──────────────┬────────────┬─────────────────┬──────────────────┐")
        print("   │ Profile      │ Batch Size │ Avg Latency     │ Throughput       │")
        print("   ├──────────────┼────────────┼─────────────────┼──────────────────┤")
        for r in allResults {
            let profileStr = r.profile.padding(toLength: 12, withPad: " ", startingAt: 0)
            let batchStr = String(r.batchSize).padding(toLength: 10, withPad: " ", startingAt: 0)
            let latencyStr = formatDuration(r.latency).padding(toLength: 15, withPad: " ", startingAt: 0)
            let throughputStr = (formatDouble(r.throughput, decimals: 0) + " s/s").padding(toLength: 16, withPad: " ", startingAt: 0)
            print("   │ \(profileStr) │ \(batchStr) │ \(latencyStr) │ \(throughputStr) │")
        }
        print("   └──────────────┴────────────┴─────────────────┴──────────────────┘")

        print("\n=== Multi-Profile Example Complete ===")
    }

    static func createDynamicONNXModel() throws -> URL {
        let onnxBase64 = "CAc6VAoZCgVpbnB1dBIGb3V0cHV0IghJZGVudGl0eRIQRHluSWRlbnRpdHlHcmFwaFoRCgVpbnB1dBIICgYIARICCgBiEgoGb3V0cHV0EggKBggBEgIKAEIECgAQDQ=="

        guard let onnxData = Data(base64Encoded: onnxBase64) else {
            throw TensorRTLLMError.runtimeUnavailable("Failed to decode ONNX model")
        }

        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tensorrt-multiprofile-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)

        let onnxURL = tmpDir.appendingPathComponent("dynamic.onnx")
        try onnxData.write(to: onnxURL)

        return onnxURL
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
        if us < 1000 { return "\(formatDouble(us, decimals: 1)) µs" }
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
