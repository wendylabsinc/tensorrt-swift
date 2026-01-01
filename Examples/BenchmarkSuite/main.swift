// BenchmarkSuite - Comprehensive performance measurement
//
// This example demonstrates:
// 1. Measuring throughput across different batch sizes
// 2. Latency percentile analysis (p50, p90, p95, p99)
// 3. Warm-up handling and statistical rigor
// 4. Memory bandwidth estimation
//
// Run with: ./scripts/swiftw run BenchmarkSuite
import TensorRT
import FoundationEssentials

#if canImport(TensorRTNative)
import TensorRTNative
#endif

@main
struct BenchmarkSuite {
    static func main() async throws {
        print("=== TensorRT Benchmark Suite ===\n")

#if canImport(TensorRTNative)
        // Benchmark configurations
        let elementCounts = [64, 256, 1024, 4096, 16384]
        let warmupIterations = 50
        let benchmarkIterations = 500

        print("Configuration:")
        print("  Element counts: \(elementCounts)")
        print("  Warmup iterations: \(warmupIterations)")
        print("  Benchmark iterations: \(benchmarkIterations)")

        // Check GPU
        let deviceCount = try TensorRTSystem.cudaDeviceCount()
        print("  CUDA devices: \(deviceCount)")

        var allResults: [BenchmarkResult] = []

        // Step 1: Benchmark each configuration
        print("\n" + "=".repeated(70))
        print("Running Benchmarks")
        print("=".repeated(70))

        for (idx, elementCount) in elementCounts.enumerated() {
            print("\n[\(idx + 1)/\(elementCounts.count)] Benchmarking \(elementCount) elements...")

            let result = try await runBenchmark(
                elementCount: elementCount,
                warmupIterations: warmupIterations,
                benchmarkIterations: benchmarkIterations
            )

            allResults.append(result)

            // Print progress
            print("  Throughput: \(formatDouble(result.throughputOpsPerSec, decimals: 0)) ops/sec")
            print("  Latency p50: \(formatDuration(result.p50))")
            print("  Latency p99: \(formatDuration(result.p99))")
        }

        // Step 2: Summary table
        print("\n" + "=".repeated(70))
        print("Benchmark Results Summary")
        print("=".repeated(70))

        print("\n┌──────────┬────────────┬────────────┬────────────┬────────────┬────────────┐")
        print("│ Elements │ Throughput │ p50        │ p90        │ p95        │ p99        │")
        print("├──────────┼────────────┼────────────┼────────────┼────────────┼────────────┤")

        for result in allResults {
            let elemStr = String(result.elementCount).padding(toLength: 8, withPad: " ", startingAt: 0)
            let tputStr = (formatDouble(result.throughputOpsPerSec / 1000, decimals: 1) + "K").padding(toLength: 10, withPad: " ", startingAt: 0)
            let p50Str = formatDuration(result.p50).padding(toLength: 10, withPad: " ", startingAt: 0)
            let p90Str = formatDuration(result.p90).padding(toLength: 10, withPad: " ", startingAt: 0)
            let p95Str = formatDuration(result.p95).padding(toLength: 10, withPad: " ", startingAt: 0)
            let p99Str = formatDuration(result.p99).padding(toLength: 10, withPad: " ", startingAt: 0)

            print("│ \(elemStr) │ \(tputStr) │ \(p50Str) │ \(p90Str) │ \(p95Str) │ \(p99Str) │")
        }

        print("└──────────┴────────────┴────────────┴────────────┴────────────┴────────────┘")

        // Step 3: Memory bandwidth analysis
        print("\n" + "=".repeated(70))
        print("Memory Bandwidth Analysis")
        print("=".repeated(70))

        print("\n┌──────────┬────────────────┬────────────────┬────────────────┐")
        print("│ Elements │ Bytes/Op       │ Bandwidth      │ Efficiency     │")
        print("├──────────┼────────────────┼────────────────┼────────────────┤")

        // Theoretical peak (varies by GPU, using conservative estimate)
        let theoreticalPeakGBps = 500.0  // ~500 GB/s for modern GPUs

        for result in allResults {
            let bytesPerOp = result.elementCount * MemoryLayout<Float>.stride * 2  // Read + Write
            let bandwidthGBps = Double(bytesPerOp) * result.throughputOpsPerSec / 1e9
            let efficiency = (bandwidthGBps / theoreticalPeakGBps) * 100

            let elemStr = String(result.elementCount).padding(toLength: 8, withPad: " ", startingAt: 0)
            let bytesStr = formatBytes(bytesPerOp).padding(toLength: 14, withPad: " ", startingAt: 0)
            let bwStr = (formatDouble(bandwidthGBps, decimals: 2) + " GB/s").padding(toLength: 14, withPad: " ", startingAt: 0)
            let effStr = (formatDouble(efficiency, decimals: 1) + "%").padding(toLength: 14, withPad: " ", startingAt: 0)

            print("│ \(elemStr) │ \(bytesStr) │ \(bwStr) │ \(effStr) │")
        }

        print("└──────────┴────────────────┴────────────────┴────────────────┘")

        // Step 4: Latency distribution for largest config
        if let largestResult = allResults.last {
            print("\n" + "=".repeated(70))
            print("Latency Distribution (\(largestResult.elementCount) elements)")
            print("=".repeated(70))

            printHistogram(latencies: largestResult.allLatencies)
        }

        // Step 5: Scaling analysis
        print("\n" + "=".repeated(70))
        print("Scaling Analysis")
        print("=".repeated(70))

        if allResults.count >= 2 {
            let smallest = allResults.first!
            let largest = allResults.last!

            let sizeRatio = Double(largest.elementCount) / Double(smallest.elementCount)
            let latencyRatio = durationToSeconds(largest.p50) / durationToSeconds(smallest.p50)
            let throughputRatio = smallest.throughputOpsPerSec / largest.throughputOpsPerSec

            print("\n  Size increase: \(formatDouble(sizeRatio, decimals: 0))x")
            print("  Latency increase: \(formatDouble(latencyRatio, decimals: 2))x")
            print("  Throughput decrease: \(formatDouble(throughputRatio, decimals: 2))x")
            print("\n  Observation: \(latencyRatio < sizeRatio ? "Sub-linear scaling (good!)" : "Linear or super-linear scaling")")
        }

        print("\n=== Benchmark Suite Complete ===")

#else
        print("This example requires TensorRTNative (Linux with TensorRT)")
#endif
    }

    struct BenchmarkResult {
        let elementCount: Int
        let totalDuration: Duration
        let iterations: Int
        let allLatencies: [Duration]

        var throughputOpsPerSec: Double {
            Double(iterations) / durationToSeconds(totalDuration)
        }

        var avgLatency: Duration {
            totalDuration / iterations
        }

        var p50: Duration { percentile(0.50) }
        var p90: Duration { percentile(0.90) }
        var p95: Duration { percentile(0.95) }
        var p99: Duration { percentile(0.99) }
        var minLatency: Duration { allLatencies.min() ?? .zero }
        var maxLatency: Duration { allLatencies.max() ?? .zero }

        func percentile(_ p: Double) -> Duration {
            let sorted = allLatencies.sorted { durationToSeconds($0) < durationToSeconds($1) }
            let index = Int(Double(sorted.count - 1) * p)
            return sorted[min(index, sorted.count - 1)]
        }
    }

    static func runBenchmark(
        elementCount: Int,
        warmupIterations: Int,
        benchmarkIterations: Int
    ) async throws -> BenchmarkResult {
        // Build engine
        let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: elementCount)
        let engine = try TensorRTRuntime().deserializeEngine(from: plan)
        let context = try engine.makeExecutionContext()

        // Prepare input
        let input: [Float] = (0..<elementCount).map { Float($0) }
        var output: [Float] = []

        // Warmup
        for _ in 0..<warmupIterations {
            try await context.enqueueF32(
                inputName: "input",
                input: input,
                outputName: "output",
                output: &output
            )
        }

        // Benchmark
        var latencies: [Duration] = []
        latencies.reserveCapacity(benchmarkIterations)

        let totalStart = ContinuousClock.now

        for _ in 0..<benchmarkIterations {
            let iterStart = ContinuousClock.now

            try await context.enqueueF32(
                inputName: "input",
                input: input,
                outputName: "output",
                output: &output
            )

            let iterDuration = ContinuousClock.now - iterStart
            latencies.append(iterDuration)
        }

        let totalDuration = ContinuousClock.now - totalStart

        return BenchmarkResult(
            elementCount: elementCount,
            totalDuration: totalDuration,
            iterations: benchmarkIterations,
            allLatencies: latencies
        )
    }

    static func printHistogram(latencies: [Duration]) {
        let sorted = latencies.map { durationToSeconds($0) * 1_000_000 }.sorted()  // Convert to microseconds
        guard !sorted.isEmpty else { return }

        let minVal = sorted.first!
        let maxVal = sorted.last!
        let range = maxVal - minVal
        let bucketCount = 10
        let bucketSize = range / Double(bucketCount)

        var buckets = [Int](repeating: 0, count: bucketCount)

        for val in sorted {
            var bucketIdx = Int((val - minVal) / bucketSize)
            bucketIdx = min(bucketIdx, bucketCount - 1)
            buckets[bucketIdx] += 1
        }

        let maxCount = buckets.max() ?? 1
        let barWidth = 40

        print("")
        for (i, count) in buckets.enumerated() {
            let rangeStart = minVal + Double(i) * bucketSize
            let rangeEnd = minVal + Double(i + 1) * bucketSize
            let barLength = Int(Double(count) / Double(maxCount) * Double(barWidth))
            let bar = String(repeating: "█", count: barLength)

            let rangeStr = "\(formatDouble(rangeStart, decimals: 0))-\(formatDouble(rangeEnd, decimals: 0))µs"
            print("  \(rangeStr.padding(toLength: 16, withPad: " ", startingAt: 0)) │\(bar) (\(count))")
        }
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

    static func formatBytes(_ bytes: Int) -> String {
        if bytes < 1024 { return "\(bytes) B" }
        else if bytes < 1024 * 1024 { return "\(formatDouble(Double(bytes) / 1024, decimals: 1)) KB" }
        else { return "\(formatDouble(Double(bytes) / (1024 * 1024), decimals: 2)) MB" }
    }
}

extension String {
    func repeated(_ count: Int) -> String {
        String(repeating: self, count: count)
    }

    func padding(toLength length: Int, withPad padString: String, startingAt: Int) -> String {
        if self.count >= length { return String(self.prefix(length)) }
        var result = self
        while result.count < length { result += padString }
        return String(result.prefix(length))
    }
}
