// CUDAEventPipelining - Overlap compute with data transfer using events
//
// This example demonstrates:
// 1. Using CUDA events to track operation completion
// 2. Overlapping H2D transfer, compute, and D2H transfer
// 3. Double/triple buffering for maximum throughput
// 4. Event-based synchronization patterns
//
// Run with: ./scripts/swiftw run CUDAEventPipelining
import TensorRT
import FoundationEssentials

#if canImport(TensorRTNative)
import TensorRTNative
#endif

@main
struct CUDAEventPipelining {
    static func main() async throws {
        print("=== CUDA Event Pipelining Example ===\n")

#if canImport(TensorRTNative)
        // Configuration
        let elementCount = 8192
        let numIterations = 200
        let byteCount = elementCount * MemoryLayout<Float>.stride

        print("Configuration:")
        print("  Elements: \(elementCount)")
        print("  Iterations: \(numIterations)")
        print("  Data size: \(byteCount / 1024) KB per transfer")

        // Step 1: Build engine
        print("\n1. Building TensorRT engine...")
        let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: elementCount)
        let engine = try TensorRTRuntime().deserializeEngine(from: plan)

        // Step 2: Create multiple streams for pipelining
        print("\n2. Creating CUDA streams...")
        var computeStream: UInt64 = 0
        var h2dStream: UInt64 = 0
        var d2hStream: UInt64 = 0

        guard trt_cuda_stream_create(&computeStream) == 0,
              trt_cuda_stream_create(&h2dStream) == 0,
              trt_cuda_stream_create(&d2hStream) == 0 else {
            throw TensorRTError.runtimeUnavailable("Failed to create streams")
        }
        defer {
            _ = trt_cuda_stream_destroy(computeStream)
            _ = trt_cuda_stream_destroy(h2dStream)
            _ = trt_cuda_stream_destroy(d2hStream)
        }

        print("   Compute stream: 0x\(String(computeStream, radix: 16))")
        print("   H2D stream: 0x\(String(h2dStream, radix: 16))")
        print("   D2H stream: 0x\(String(d2hStream, radix: 16))")

        // Create context on compute stream
        let context = try engine.makeExecutionContext(queue: .external(streamIdentifier: computeStream))

        // Step 3: Allocate double buffers
        print("\n3. Allocating double buffers...")

        var dInput0: UInt64 = 0, dInput1: UInt64 = 0
        var dOutput0: UInt64 = 0, dOutput1: UInt64 = 0

        guard trt_cuda_malloc(byteCount, &dInput0) == 0,
              trt_cuda_malloc(byteCount, &dInput1) == 0,
              trt_cuda_malloc(byteCount, &dOutput0) == 0,
              trt_cuda_malloc(byteCount, &dOutput1) == 0 else {
            throw TensorRTError.runtimeUnavailable("Failed to allocate buffers")
        }
        defer {
            _ = trt_cuda_free(dInput0)
            _ = trt_cuda_free(dInput1)
            _ = trt_cuda_free(dOutput0)
            _ = trt_cuda_free(dOutput1)
        }

        let inputBuffers = [dInput0, dInput1]
        let outputBuffers = [dOutput0, dOutput1]

        print("   Input buffers:  [0x\(String(dInput0, radix: 16)), 0x\(String(dInput1, radix: 16))]")
        print("   Output buffers: [0x\(String(dOutput0, radix: 16)), 0x\(String(dOutput1, radix: 16))]")

        // Step 4: Create event for synchronization
        print("\n4. Creating CUDA event...")
        let computeComplete = try TensorRTSystem.CUDAEvent()
        print("   Event created for compute synchronization")

        // Prepare host data
        let inputs: [[Float]] = (0..<numIterations).map { i in
            (0..<elementCount).map { Float($0 + i * elementCount) }
        }
        var outputs = [[Float]](repeating: [Float](repeating: 0, count: elementCount), count: numIterations)

        // Step 5: Sequential baseline
        print("\n5. Sequential baseline (no pipelining)...")

        let seqStart = ContinuousClock.now

        for i in 0..<numIterations {
            let bufIdx = 0

            // H2D
            inputs[i].withUnsafeBytes { raw in
                _ = trt_cuda_memcpy_htod(inputBuffers[bufIdx], raw.baseAddress, raw.count)
            }

            // Compute
            try await context.enqueueDevice(
                inputs: ["input": (address: inputBuffers[bufIdx], length: byteCount)],
                outputs: ["output": (address: outputBuffers[bufIdx], length: byteCount)],
                synchronously: true
            )

            // D2H
            outputs[i].withUnsafeMutableBytes { raw in
                _ = trt_cuda_memcpy_dtoh(raw.baseAddress, outputBuffers[bufIdx], raw.count)
            }
        }

        let seqDuration = ContinuousClock.now - seqStart
        print("   Sequential time: \(seqDuration)")

        // Step 6: Double-buffered pipelining
        print("\n6. Double-buffered pipelining...")
        print("   Pipeline stages: H2D[i+1] | Compute[i] | D2H[i-1]")

        // Reset outputs
        outputs = [[Float]](repeating: [Float](repeating: 0, count: elementCount), count: numIterations)

        let pipeStart = ContinuousClock.now

        // Prime the pipeline: load first input
        inputs[0].withUnsafeBytes { raw in
            _ = trt_cuda_memcpy_htod(inputBuffers[0], raw.baseAddress, raw.count)
        }

        for i in 0..<numIterations {
            let currBuf = i % 2
            let nextBuf = (i + 1) % 2

            // Start H2D for next iteration (if not last)
            if i < numIterations - 1 {
                inputs[i + 1].withUnsafeBytes { raw in
                    // In production, use async memcpy on h2dStream
                    _ = trt_cuda_memcpy_htod(inputBuffers[nextBuf], raw.baseAddress, raw.count)
                }
            }

            // Compute on current buffer
            try await context.enqueueDevice(
                inputs: ["input": (address: inputBuffers[currBuf], length: byteCount)],
                outputs: ["output": (address: outputBuffers[currBuf], length: byteCount)],
                synchronously: false
            )
            try await context.recordEvent(computeComplete)

            // D2H for current result
            try computeComplete.synchronize()
            outputs[i].withUnsafeMutableBytes { raw in
                _ = trt_cuda_memcpy_dtoh(raw.baseAddress, outputBuffers[currBuf], raw.count)
            }
        }

        let pipeDuration = ContinuousClock.now - pipeStart
        print("   Pipelined time: \(pipeDuration)")

        // Step 7: Event-based fine-grained synchronization demo
        print("\n7. Event-based synchronization patterns...")

        // Pattern 1: Wait for specific operation
        print("   a) Wait for compute completion:")
        try await context.enqueueDevice(
            inputs: ["input": (address: inputBuffers[0], length: byteCount)],
            outputs: ["output": (address: outputBuffers[0], length: byteCount)],
            synchronously: false
        )
        try await context.recordEvent(computeComplete)

        var pollCount = 0
        while !(try computeComplete.isReady()) {
            pollCount += 1
        }
        print("      Polled \(pollCount) times until ready")

        // Pattern 2: Blocking sync
        print("   b) Blocking synchronization:")
        try await context.enqueueDevice(
            inputs: ["input": (address: inputBuffers[0], length: byteCount)],
            outputs: ["output": (address: outputBuffers[0], length: byteCount)],
            synchronously: false
        )
        try await context.recordEvent(computeComplete)

        let syncStart = ContinuousClock.now
        try computeComplete.synchronize()
        let syncDuration = ContinuousClock.now - syncStart
        print("      Sync took: \(syncDuration)")

        // Step 8: Verify correctness
        print("\n8. Verifying results...")
        var allCorrect = true
        for i in 0..<min(10, numIterations) {
            if outputs[i] != inputs[i] {
                allCorrect = false
                print("   Mismatch at iteration \(i)")
                break
            }
        }
        print("   Verification: \(allCorrect ? "PASSED" : "FAILED")")

        // Step 9: Summary
        print("\n9. Performance Summary:")
        let seqThroughput = Double(numIterations) / durationToSeconds(seqDuration)
        let pipeThroughput = Double(numIterations) / durationToSeconds(pipeDuration)
        let speedup = durationToSeconds(seqDuration) / durationToSeconds(pipeDuration)

        print("   ┌───────────────────────┬────────────────────┬─────────────────┐")
        print("   │ Mode                  │ Time               │ Throughput      │")
        print("   ├───────────────────────┼────────────────────┼─────────────────┤")
        print("   │ Sequential            │ \(formatDuration(seqDuration).padding(toLength: 18, withPad: " ", startingAt: 0)) │ \(formatDouble(seqThroughput, decimals: 0).padding(toLength: 11, withPad: " ", startingAt: 0)) it/s │")
        print("   │ Double-buffered       │ \(formatDuration(pipeDuration).padding(toLength: 18, withPad: " ", startingAt: 0)) │ \(formatDouble(pipeThroughput, decimals: 0).padding(toLength: 11, withPad: " ", startingAt: 0)) it/s │")
        print("   └───────────────────────┴────────────────────┴─────────────────┘")
        print("\n   Speedup from pipelining: \(formatDouble(speedup, decimals: 2))x")

        print("\n=== CUDA Event Pipelining Complete ===")

#else
        print("This example requires TensorRTNative (Linux with TensorRT)")
#endif
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
        let seconds = durationToSeconds(duration)
        if seconds < 0.001 { return "\(formatDouble(seconds * 1_000_000, decimals: 0)) µs" }
        else if seconds < 1 { return "\(formatDouble(seconds * 1000, decimals: 2)) ms" }
        else { return "\(formatDouble(seconds, decimals: 3)) s" }
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
