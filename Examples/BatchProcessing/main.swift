// BatchProcessing - Process multiple inputs through TensorRT
//
// This example demonstrates:
// 1. Reusing an ExecutionContext for multiple inferences
// 2. Processing a "dataset" of inputs sequentially
// 3. Collecting and aggregating results
// 4. Measuring per-batch and total latency
//
// Run with: ./scripts/swiftw run BatchProcessing
import TensorRT
import FoundationEssentials

@main
struct BatchProcessing {
    static func main() async throws {
        print("=== Batch Processing Example ===\n")

        // Configuration
        let elementCount = 16          // Elements per input tensor
        let numBatches = 20            // Number of batches to process
        let printEveryN = 5            // Print progress every N batches

        print("Configuration:")
        print("  Elements per tensor: \(elementCount)")
        print("  Number of batches: \(numBatches)")

        // Step 1: Build the engine
        print("\n1. Building TensorRT engine...")
        let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: elementCount)
        let runtime = TensorRTRuntime()
        let engine = try runtime.deserializeEngine(from: plan)
        print("   Engine ready (plan size: \(plan.count) bytes)")

        // Step 2: Create execution context (reused for all batches)
        print("\n2. Creating execution context...")
        let context = try engine.makeExecutionContext()
        print("   Context ready")

        // Step 3: Generate synthetic dataset
        print("\n3. Generating synthetic dataset...")
        let dataset = generateDataset(numBatches: numBatches, elementsPerBatch: elementCount)
        print("   Generated \(dataset.count) batches")

        // Step 4: Process all batches
        print("\n4. Processing batches...")
        var results: [[Float]] = []
        var latencies: [Duration] = []
        results.reserveCapacity(numBatches)
        latencies.reserveCapacity(numBatches)

        let totalStart = ContinuousClock.now

        for (index, inputData) in dataset.enumerated() {
            let batchStart = ContinuousClock.now

            // Run inference
            var output: [Float] = []
            try await context.enqueueF32(
                inputName: "input",
                input: inputData,
                outputName: "output",
                output: &output
            )

            let batchDuration = ContinuousClock.now - batchStart
            latencies.append(batchDuration)
            results.append(output)

            // Progress reporting
            if (index + 1) % printEveryN == 0 || index == 0 {
                print("   Batch \(index + 1)/\(numBatches): latency = \(batchDuration)")
            }
        }

        let totalDuration = ContinuousClock.now - totalStart

        // Step 5: Verify results
        print("\n5. Verifying results...")
        var allCorrect = true
        var errorCount = 0

        for (index, (input, output)) in zip(dataset, results).enumerated() {
            if input != output {
                allCorrect = false
                errorCount += 1
                if errorCount <= 3 {
                    print("   Mismatch in batch \(index):")
                    print("     Input:  \(input.prefix(4))...")
                    print("     Output: \(output.prefix(4))...")
                }
            }
        }

        if allCorrect {
            print("   All \(numBatches) batches verified correctly!")
        } else {
            print("   Verification FAILED: \(errorCount) batches had mismatches")
        }

        // Step 6: Statistics
        print("\n6. Performance Statistics:")
        let stats = computeStatistics(latencies)

        print("   Total time: \(totalDuration)")
        print("   Batches processed: \(numBatches)")
        print("   Latency (min):  \(stats.min)")
        print("   Latency (max):  \(stats.max)")
        print("   Latency (avg):  \(stats.avg)")
        print("   Latency (p50):  \(stats.p50)")
        print("   Latency (p95):  \(stats.p95)")
        print("   Latency (p99):  \(stats.p99)")

        let throughput = Double(numBatches) / durationToSeconds(totalDuration)
        let elementsPerSecond = Double(numBatches * elementCount) / durationToSeconds(totalDuration)

        print("   Throughput: \(formatDouble(throughput, decimals: 1)) batches/sec")
        print("   Elements/sec: \(formatDouble(elementsPerSecond, decimals: 0))")

        // Step 7: Demonstrate different input patterns
        print("\n7. Processing special patterns...")

        // All zeros
        var zerosOutput: [Float] = []
        try await context.enqueueF32(
            inputName: "input",
            input: [Float](repeating: 0, count: elementCount),
            outputName: "output",
            output: &zerosOutput
        )
        print("   Zeros pattern: \(zerosOutput.allSatisfy { $0 == 0 } ? "PASS" : "FAIL")")

        // All ones
        var onesOutput: [Float] = []
        try await context.enqueueF32(
            inputName: "input",
            input: [Float](repeating: 1, count: elementCount),
            outputName: "output",
            output: &onesOutput
        )
        print("   Ones pattern: \(onesOutput.allSatisfy { $0 == 1 } ? "PASS" : "FAIL")")

        // Large values
        var largeOutput: [Float] = []
        let largeInput = (0..<elementCount).map { Float($0) * 1000.0 }
        try await context.enqueueF32(
            inputName: "input",
            input: largeInput,
            outputName: "output",
            output: &largeOutput
        )
        print("   Large values pattern: \(largeOutput == largeInput ? "PASS" : "FAIL")")

        // Negative values
        var negativeOutput: [Float] = []
        let negativeInput = (0..<elementCount).map { -Float($0) }
        try await context.enqueueF32(
            inputName: "input",
            input: negativeInput,
            outputName: "output",
            output: &negativeOutput
        )
        print("   Negative values pattern: \(negativeOutput == negativeInput ? "PASS" : "FAIL")")

        print("\n=== Batch Processing Complete ===")
        print("Processed \(numBatches) batches with \(elementCount) elements each.")
        print("Total elements: \(numBatches * elementCount)")
        print("All tests: \(allCorrect ? "PASSED" : "FAILED")")
    }

    /// Generates synthetic dataset with varied patterns
    static func generateDataset(numBatches: Int, elementsPerBatch: Int) -> [[Float]] {
        var dataset: [[Float]] = []
        dataset.reserveCapacity(numBatches)

        for batchIndex in 0..<numBatches {
            let offset = Float(batchIndex * elementsPerBatch)
            let batch = (0..<elementsPerBatch).map { offset + Float($0) }
            dataset.append(batch)
        }

        return dataset
    }

    struct LatencyStats {
        var min: Duration
        var max: Duration
        var avg: Duration
        var p50: Duration
        var p95: Duration
        var p99: Duration
    }

    /// Computes latency statistics
    static func computeStatistics(_ latencies: [Duration]) -> LatencyStats {
        guard !latencies.isEmpty else {
            return LatencyStats(
                min: .zero, max: .zero, avg: .zero,
                p50: .zero, p95: .zero, p99: .zero
            )
        }

        let sorted = latencies.sorted { durationToSeconds($0) < durationToSeconds($1) }
        let count = sorted.count

        func percentile(_ p: Double) -> Duration {
            let index = Int(Double(count - 1) * p)
            return sorted[min(index, count - 1)]
        }

        let totalNanos = latencies.reduce(0.0) { $0 + durationToSeconds($1) }
        let avgSeconds = totalNanos / Double(count)

        return LatencyStats(
            min: sorted.first!,
            max: sorted.last!,
            avg: .seconds(avgSeconds),
            p50: percentile(0.50),
            p95: percentile(0.95),
            p99: percentile(0.99)
        )
    }

    static func durationToSeconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    /// Simple decimal formatting without Foundation
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
}
