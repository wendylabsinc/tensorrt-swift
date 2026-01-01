// AsyncInference - Non-blocking inference with CUDA streams
//
// This example demonstrates:
// 1. Using external CUDA streams for async execution
// 2. Enqueuing work without blocking
// 3. Using CUDA events to track completion
// 4. Overlapping CPU work with GPU execution
//
// Run with: ./scripts/swiftw run AsyncInference
import TensorRT
import FoundationEssentials

#if canImport(TensorRTNative)
import TensorRTNative
#endif

@main
struct AsyncInference {
    static func main() async throws {
        print("=== Async Inference Example ===\n")

#if canImport(TensorRTNative)
        // Configuration
        let elementCount = 1024
        let numIterations = 50

        print("Configuration:")
        print("  Elements per tensor: \(elementCount)")
        print("  Iterations: \(numIterations)")

        // Step 1: Build engine
        print("\n1. Building TensorRT engine...")
        let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: elementCount)
        let engine = try TensorRTRuntime().deserializeEngine(from: plan)
        print("   Engine ready")

        // Step 2: Create CUDA stream
        print("\n2. Creating CUDA stream...")
        var stream: UInt64 = 0
        guard trt_cuda_stream_create(&stream) == 0 else {
            throw TensorRTError.runtimeUnavailable("Failed to create CUDA stream")
        }
        defer { _ = trt_cuda_stream_destroy(stream) }
        print("   Stream created: 0x\(String(stream, radix: 16))")

        // Step 3: Create context with external stream
        print("\n3. Creating execution context with external stream...")
        let context = try engine.makeExecutionContext(queue: .external(streamIdentifier: stream))

        // Step 4: Allocate device memory
        print("\n4. Allocating device memory...")
        let byteCount = elementCount * MemoryLayout<Float>.stride

        var dInput: UInt64 = 0
        var dOutput: UInt64 = 0
        guard trt_cuda_malloc(byteCount, &dInput) == 0 else {
            throw TensorRTError.runtimeUnavailable("Failed to allocate input buffer")
        }
        defer { _ = trt_cuda_free(dInput) }

        guard trt_cuda_malloc(byteCount, &dOutput) == 0 else {
            throw TensorRTError.runtimeUnavailable("Failed to allocate output buffer")
        }
        defer { _ = trt_cuda_free(dOutput) }

        print("   Input buffer:  0x\(String(dInput, radix: 16))")
        print("   Output buffer: 0x\(String(dOutput, radix: 16))")

        // Step 5: Demonstrate synchronous vs async execution
        print("\n5. Comparing synchronous vs asynchronous execution...")

        // Prepare input data
        let input: [Float] = (0..<elementCount).map { Float($0) }

        // Copy input to device
        input.withUnsafeBytes { raw in
            _ = trt_cuda_memcpy_htod(dInput, raw.baseAddress, raw.count)
        }

        // --- Synchronous execution ---
        print("\n   --- Synchronous Execution ---")
        let syncStart = ContinuousClock.now

        for _ in 0..<numIterations {
            try await context.enqueueDevice(
                inputs: ["input": (address: dInput, length: byteCount)],
                outputs: ["output": (address: dOutput, length: byteCount)],
                synchronously: true  // Block until complete
            )
        }

        let syncDuration = ContinuousClock.now - syncStart
        print("   Sync time: \(syncDuration)")
        print("   Avg latency: \(syncDuration / numIterations)")

        // --- Asynchronous execution with event synchronization ---
        print("\n   --- Asynchronous Execution ---")
        let asyncStart = ContinuousClock.now

        // Create event for completion tracking
        let event = try TensorRTSystem.CUDAEvent()

        var cpuWorkDone = 0

        for i in 0..<numIterations {
            // Enqueue GPU work (non-blocking)
            try await context.enqueueDevice(
                inputs: ["input": (address: dInput, length: byteCount)],
                outputs: ["output": (address: dOutput, length: byteCount)],
                synchronously: false  // Don't block
            )

            // Record event after this operation
            try await context.recordEvent(event)

            // Do CPU work while GPU is executing
            cpuWorkDone += simulateCPUWork(iteration: i)

            // Wait for GPU only when needed (e.g., before using results)
            if (i + 1) % 10 == 0 {
                try event.synchronize()
            }
        }

        // Final sync
        try event.synchronize()

        let asyncDuration = ContinuousClock.now - asyncStart
        print("   Async time: \(asyncDuration)")
        print("   Avg latency: \(asyncDuration / numIterations)")
        print("   CPU work completed: \(cpuWorkDone) units")

        // Step 6: Demonstrate event-based polling
        print("\n6. Event-based polling demo...")

        try await context.enqueueDevice(
            inputs: ["input": (address: dInput, length: byteCount)],
            outputs: ["output": (address: dOutput, length: byteCount)],
            synchronously: false
        )
        try await context.recordEvent(event)

        var pollCount = 0
        while !(try event.isReady()) {
            pollCount += 1
            // Do other work...
        }
        print("   Polled \(pollCount) times before GPU completed")

        // Step 7: Verify results
        print("\n7. Verifying results...")
        var output = [Float](repeating: 0, count: elementCount)
        output.withUnsafeMutableBytes { raw in
            _ = trt_cuda_memcpy_dtoh(raw.baseAddress, dOutput, raw.count)
        }

        let match = input == output
        print("   Input == Output: \(match ? "YES" : "NO")")

        // Step 8: Summary
        print("\n8. Performance Summary:")
        let syncThroughput = Double(numIterations) / durationToSeconds(syncDuration)
        let asyncThroughput = Double(numIterations) / durationToSeconds(asyncDuration)
        let speedup = durationToSeconds(syncDuration) / durationToSeconds(asyncDuration)

        print("   Synchronous throughput:  \(formatDouble(syncThroughput, decimals: 1)) inferences/sec")
        print("   Asynchronous throughput: \(formatDouble(asyncThroughput, decimals: 1)) inferences/sec")
        print("   Speedup from async:      \(formatDouble(speedup, decimals: 2))x")
        print("   CPU work overlapped:     \(cpuWorkDone) units")

        print("\n=== Async Inference Complete ===")

#else
        print("This example requires TensorRTNative (Linux with TensorRT)")
#endif
    }

    /// Simulates CPU work that can overlap with GPU execution
    static func simulateCPUWork(iteration: Int) -> Int {
        // Simulate some computation
        var result = 0
        for i in 0..<1000 {
            result += i * iteration
        }
        return result > 0 ? 1 : 0
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
