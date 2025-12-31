// FP16Quantization - Compare FP32 vs FP16 precision engines
//
// This example demonstrates:
// 1. Building engines with different precision modes
// 2. Comparing accuracy between FP32 and FP16
// 3. Measuring performance differences
// 4. Understanding precision trade-offs
//
// Run with: swift run FP16Quantization

import TensorRTLLM
import FoundationEssentials

@main
struct FP16Quantization {
    static func main() async throws {
        print("=== FP16 Quantization Example ===\n")
        print("This example compares FP32 and FP16 precision for TensorRT inference.")
        print("Note: Using identity model - real models show more dramatic differences.\n")

        // Configuration
        let elementCount = 4096
        let benchmarkIterations = 200

        print("Configuration:")
        print("  Elements: \(elementCount)")
        print("  Benchmark iterations: \(benchmarkIterations)")

        // Step 1: Create ONNX model for both precision tests
        print("\n1. Creating ONNX model...")
        let onnxURL = try createONNXModel(elementCount: elementCount)
        defer { try? FileManager.default.removeItem(at: onnxURL.deletingLastPathComponent()) }
        print("   Model created: \(elementCount) float elements")

        let runtime = TensorRTLLMRuntime()

        // Step 2: Build FP32 engine
        print("\n2. Building FP32 engine...")
        let fp32Start = ContinuousClock.now
        let fp32Engine = try runtime.buildEngine(
            onnxURL: onnxURL,
            options: EngineBuildOptions(precision: [.fp32])
        )
        let fp32BuildTime = ContinuousClock.now - fp32Start
        print("   FP32 build time: \(fp32BuildTime)")
        print("   FP32 plan size: \(fp32Engine.serialized?.count ?? 0) bytes")

        // Step 3: Build FP16 engine
        print("\n3. Building FP16 engine...")
        let fp16Start = ContinuousClock.now
        let fp16Engine = try runtime.buildEngine(
            onnxURL: onnxURL,
            options: EngineBuildOptions(precision: [.fp16])
        )
        let fp16BuildTime = ContinuousClock.now - fp16Start
        print("   FP16 build time: \(fp16BuildTime)")
        print("   FP16 plan size: \(fp16Engine.serialized?.count ?? 0) bytes")

        // Step 4: Compare plan sizes
        let fp32Size = fp32Engine.serialized?.count ?? 0
        let fp16Size = fp16Engine.serialized?.count ?? 0
        let sizeReduction = 100.0 * (1.0 - Double(fp16Size) / Double(fp32Size))

        print("\n4. Plan Size Comparison:")
        print("   FP32: \(formatBytes(fp32Size))")
        print("   FP16: \(formatBytes(fp16Size))")
        print("   Reduction: \(formatDouble(sizeReduction, decimals: 1))%")

        // Step 5: Create execution contexts
        print("\n5. Creating execution contexts...")
        let fp32Context = try fp32Engine.makeExecutionContext()
        let fp16Context = try fp16Engine.makeExecutionContext()

        // Step 6: Prepare test data with various magnitudes
        print("\n6. Preparing test data...")
        let testCases: [(name: String, data: [Float])] = [
            ("Normal range", (0..<elementCount).map { Float($0) / Float(elementCount) }),
            ("Large values", (0..<elementCount).map { Float($0) * 100.0 }),
            ("Small values", (0..<elementCount).map { Float($0) * 0.0001 }),
            ("Mixed signs", (0..<elementCount).map { Float($0 % 2 == 0 ? $0 : -$0) }),
        ]

        // Step 7: Accuracy comparison
        print("\n7. Accuracy Comparison:")
        print("   ┌─────────────────┬──────────────┬──────────────┬──────────────┐")
        print("   │ Test Case       │ Max Diff     │ Avg Diff     │ Rel Error    │")
        print("   ├─────────────────┼──────────────┼──────────────┼──────────────┤")

        for (name, input) in testCases {
            // Run FP32
            var fp32Output: [Float] = []
            try await fp32Context.enqueueF32(
                inputName: "input",
                input: input,
                outputName: "output",
                output: &fp32Output
            )

            // Run FP16
            var fp16Output: [Float] = []
            try await fp16Context.enqueueF32(
                inputName: "input",
                input: input,
                outputName: "output",
                output: &fp16Output
            )

            // Calculate differences
            var maxDiff: Float = 0
            var sumDiff: Float = 0
            var sumRelErr: Float = 0

            for i in 0..<min(fp32Output.count, fp16Output.count) {
                let diff = abs(fp32Output[i] - fp16Output[i])
                maxDiff = max(maxDiff, diff)
                sumDiff += diff

                if abs(fp32Output[i]) > 1e-6 {
                    sumRelErr += diff / abs(fp32Output[i])
                }
            }

            let avgDiff = sumDiff / Float(fp32Output.count)
            let avgRelErr = (sumRelErr / Float(fp32Output.count)) * 100

            let nameStr = name.padding(toLength: 15, withPad: " ", startingAt: 0)
            let maxStr = formatScientific(maxDiff).padding(toLength: 12, withPad: " ", startingAt: 0)
            let avgStr = formatScientific(avgDiff).padding(toLength: 12, withPad: " ", startingAt: 0)
            let relStr = (formatDouble(Double(avgRelErr), decimals: 4) + "%").padding(toLength: 12, withPad: " ", startingAt: 0)

            print("   │ \(nameStr) │ \(maxStr) │ \(avgStr) │ \(relStr) │")
        }

        print("   └─────────────────┴──────────────┴──────────────┴──────────────┘")

        // Step 8: Performance comparison
        print("\n8. Performance Comparison:")

        let normalInput: [Float] = (0..<elementCount).map { Float($0) / Float(elementCount) }

        // Warmup
        for _ in 0..<50 {
            var output: [Float] = []
            try await fp32Context.enqueueF32(inputName: "input", input: normalInput, outputName: "output", output: &output)
            try await fp16Context.enqueueF32(inputName: "input", input: normalInput, outputName: "output", output: &output)
        }

        // Benchmark FP32
        let fp32BenchStart = ContinuousClock.now
        for _ in 0..<benchmarkIterations {
            var output: [Float] = []
            try await fp32Context.enqueueF32(inputName: "input", input: normalInput, outputName: "output", output: &output)
        }
        let fp32BenchTime = ContinuousClock.now - fp32BenchStart

        // Benchmark FP16
        let fp16BenchStart = ContinuousClock.now
        for _ in 0..<benchmarkIterations {
            var output: [Float] = []
            try await fp16Context.enqueueF32(inputName: "input", input: normalInput, outputName: "output", output: &output)
        }
        let fp16BenchTime = ContinuousClock.now - fp16BenchStart

        let fp32Throughput = Double(benchmarkIterations) / durationToSeconds(fp32BenchTime)
        let fp16Throughput = Double(benchmarkIterations) / durationToSeconds(fp16BenchTime)
        let speedup = fp16Throughput / fp32Throughput

        print("   ┌─────────────┬────────────────────┬────────────────────┬──────────┐")
        print("   │ Precision   │ Total Time         │ Throughput         │ Speedup  │")
        print("   ├─────────────┼────────────────────┼────────────────────┼──────────┤")
        print("   │ FP32        │ \(formatDuration(fp32BenchTime).padding(toLength: 18, withPad: " ", startingAt: 0)) │ \(formatDouble(fp32Throughput, decimals: 0).padding(toLength: 14, withPad: " ", startingAt: 0)) ops/s │ 1.00x    │")
        print("   │ FP16        │ \(formatDuration(fp16BenchTime).padding(toLength: 18, withPad: " ", startingAt: 0)) │ \(formatDouble(fp16Throughput, decimals: 0).padding(toLength: 14, withPad: " ", startingAt: 0)) ops/s │ \(formatDouble(speedup, decimals: 2))x    │")
        print("   └─────────────┴────────────────────┴────────────────────┴──────────┘")

        // Step 9: Memory bandwidth comparison
        print("\n9. Memory Bandwidth Analysis:")
        let bytesPerOp = elementCount * MemoryLayout<Float>.stride * 2
        let fp32Bandwidth = Double(bytesPerOp) * fp32Throughput / 1e9
        let fp16Bandwidth = Double(bytesPerOp) * fp16Throughput / 1e9

        print("   FP32 effective bandwidth: \(formatDouble(fp32Bandwidth, decimals: 2)) GB/s")
        print("   FP16 effective bandwidth: \(formatDouble(fp16Bandwidth, decimals: 2)) GB/s")

        // Step 10: Recommendations
        print("\n10. Precision Selection Guidelines:")
        print("   ┌─────────────────────────────────────────────────────────────────┐")
        print("   │ Use FP32 when:                                                  │")
        print("   │   - Training models (gradients need precision)                  │")
        print("   │   - Scientific computing requiring exact results                │")
        print("   │   - Financial calculations                                      │")
        print("   ├─────────────────────────────────────────────────────────────────┤")
        print("   │ Use FP16 when:                                                  │")
        print("   │   - Inference workloads (most common case)                      │")
        print("   │   - Memory-constrained environments                             │")
        print("   │   - Latency-critical applications                               │")
        print("   │   - Accuracy loss is acceptable (most vision/NLP models)        │")
        print("   └─────────────────────────────────────────────────────────────────┘")

        print("\n=== FP16 Quantization Example Complete ===")
    }

    static func createONNXModel(elementCount: Int) throws -> URL {
        // Using dynamic ONNX model
        let onnxBase64 = "CAc6VAoZCgVpbnB1dBIGb3V0cHV0IghJZGVudGl0eRIQRHluSWRlbnRpdHlHcmFwaFoRCgVpbnB1dBIICgYIARICCgBiEgoGb3V0cHV0EggKBggBEgIKAEIECgAQDQ=="

        guard let onnxData = Data(base64Encoded: onnxBase64) else {
            throw TensorRTLLMError.runtimeUnavailable("Failed to decode ONNX model")
        }

        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tensorrt-fp16-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)

        let onnxURL = tmpDir.appendingPathComponent("model.onnx")
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
        let fracStr = String(fracPart)
        let paddedFrac = String(repeating: "0", count: max(0, decimals - fracStr.count)) + fracStr
        return "\(intPart).\(paddedFrac)"
    }

    static func formatScientific(_ value: Float) -> String {
        if value == 0 { return "0.00e+00" }
        let absVal = abs(value)
        var exponent = 0
        var mantissa = Double(absVal)

        while mantissa >= 10 {
            mantissa /= 10
            exponent += 1
        }
        while mantissa < 1 && mantissa > 0 {
            mantissa *= 10
            exponent -= 1
        }

        let sign = value < 0 ? "-" : ""
        let expSign = exponent >= 0 ? "+" : ""
        return "\(sign)\(formatDouble(mantissa, decimals: 2))e\(expSign)\(exponent)"
    }

    static func formatDuration(_ duration: Duration) -> String {
        let seconds = durationToSeconds(duration)
        if seconds < 0.001 { return "\(formatDouble(seconds * 1_000_000, decimals: 0)) µs" }
        else if seconds < 1 { return "\(formatDouble(seconds * 1000, decimals: 2)) ms" }
        else { return "\(formatDouble(seconds, decimals: 3)) s" }
    }

    static func formatBytes(_ bytes: Int) -> String {
        if bytes < 1024 { return "\(bytes) B" }
        else if bytes < 1024 * 1024 { return "\(formatDouble(Double(bytes) / 1024, decimals: 1)) KB" }
        else { return "\(formatDouble(Double(bytes) / (1024 * 1024), decimals: 2)) MB" }
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
