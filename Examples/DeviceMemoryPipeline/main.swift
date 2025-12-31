// DeviceMemoryPipeline - Keep tensors on GPU across multiple inferences
//
// This example demonstrates:
// 1. Allocating persistent device memory
// 2. Running multiple inferences without H2D/D2H copies
// 3. Chaining operations on the GPU
// 4. Measuring the speedup from avoiding memory transfers
//
// Run with: swift run DeviceMemoryPipeline

import TensorRTLLM
import FoundationEssentials

#if canImport(TensorRTLLMNative)
import TensorRTLLMNative
#endif

@main
struct DeviceMemoryPipeline {
    static func main() async throws {
        print("=== Device Memory Pipeline Example ===\n")

#if canImport(TensorRTLLMNative)
        // Configuration
        let elementCount = 4096  // Large enough to see transfer overhead
        let pipelineStages = 5   // Number of inference stages
        let iterations = 100

        print("Configuration:")
        print("  Elements per tensor: \(elementCount)")
        print("  Pipeline stages: \(pipelineStages)")
        print("  Iterations: \(iterations)")

        // Step 1: Build engine
        print("\n1. Building TensorRT engine...")
        let plan = try TensorRTLLMSystem.buildIdentityEnginePlan(elementCount: elementCount)
        let engine = try TensorRTLLMRuntime().deserializeEngine(from: plan)
        let context = try engine.makeExecutionContext()
        print("   Engine ready")

        // Step 2: Allocate device memory for pipeline
        print("\n2. Allocating device memory pipeline...")
        let byteCount = elementCount * MemoryLayout<Float>.stride

        // Allocate buffers for multi-stage pipeline
        var deviceBuffers: [UInt64] = []
        for i in 0..<(pipelineStages + 1) {
            var buffer: UInt64 = 0
            guard trt_cuda_malloc(byteCount, &buffer) == 0 else {
                throw TensorRTLLMError.runtimeUnavailable("Failed to allocate buffer \(i)")
            }
            deviceBuffers.append(buffer)
            print("   Buffer \(i): 0x\(String(buffer, radix: 16))")
        }

        defer {
            for buffer in deviceBuffers {
                _ = trt_cuda_free(buffer)
            }
        }

        // Step 3: Prepare input data
        print("\n3. Preparing input data...")
        let input: [Float] = (0..<elementCount).map { Float($0) * 0.001 }

        // Step 4: Benchmark HOST memory pipeline (with H2D/D2H each stage)
        print("\n4. Benchmarking HOST memory pipeline...")
        print("   (Transfers data to/from CPU at each stage)")

        let hostStart = ContinuousClock.now
        var hostResult = input

        for _ in 0..<iterations {
            var currentData = hostResult

            for _ in 0..<pipelineStages {
                // Host -> Device
                currentData.withUnsafeBytes { raw in
                    _ = trt_cuda_memcpy_htod(deviceBuffers[0], raw.baseAddress, raw.count)
                }

                // Execute
                try await context.enqueueDevice(
                    inputs: ["input": (address: deviceBuffers[0], length: byteCount)],
                    outputs: ["output": (address: deviceBuffers[1], length: byteCount)],
                    synchronously: true
                )

                // Device -> Host
                currentData.withUnsafeMutableBytes { raw in
                    _ = trt_cuda_memcpy_dtoh(raw.baseAddress, deviceBuffers[1], raw.count)
                }
            }

            hostResult = currentData
        }

        let hostDuration = ContinuousClock.now - hostStart
        print("   Host pipeline time: \(hostDuration)")
        print("   Avg per iteration: \(hostDuration / iterations)")

        // Step 5: Benchmark DEVICE memory pipeline (data stays on GPU)
        print("\n5. Benchmarking DEVICE memory pipeline...")
        print("   (Data stays on GPU between stages)")

        let deviceStart = ContinuousClock.now

        for iter in 0..<iterations {
            // Initial H2D transfer (only once per iteration)
            input.withUnsafeBytes { raw in
                _ = trt_cuda_memcpy_htod(deviceBuffers[0], raw.baseAddress, raw.count)
            }

            // Pipeline: output of stage N becomes input to stage N+1
            for stage in 0..<pipelineStages {
                let inputBuffer = deviceBuffers[stage % 2]
                let outputBuffer = deviceBuffers[(stage + 1) % 2]

                try await context.enqueueDevice(
                    inputs: ["input": (address: inputBuffer, length: byteCount)],
                    outputs: ["output": (address: outputBuffer, length: byteCount)],
                    synchronously: true
                )
            }

            // Final D2H transfer (only once per iteration)
            if iter == iterations - 1 {
                hostResult.withUnsafeMutableBytes { raw in
                    let finalBuffer = deviceBuffers[pipelineStages % 2]
                    _ = trt_cuda_memcpy_dtoh(raw.baseAddress, finalBuffer, raw.count)
                }
            }
        }

        let deviceDuration = ContinuousClock.now - deviceStart
        print("   Device pipeline time: \(deviceDuration)")
        print("   Avg per iteration: \(deviceDuration / iterations)")

        // Step 6: Advanced - Double buffering for overlapped transfers
        print("\n6. Double-buffered pipeline demo...")

        var stream: UInt64 = 0
        guard trt_cuda_stream_create(&stream) == 0 else {
            throw TensorRTLLMError.runtimeUnavailable("Failed to create stream")
        }
        defer { _ = trt_cuda_stream_destroy(stream) }

        let asyncContext = try engine.makeExecutionContext(queue: .external(streamIdentifier: stream))
        let event = try TensorRTLLMSystem.CUDAEvent()

        let doubleBufferStart = ContinuousClock.now

        // Ping-pong between two buffer pairs
        for iter in 0..<iterations {
            let pingIdx = (iter % 2) * 2
            let pongIdx = pingIdx + 1

            // Async H2D
            input.withUnsafeBytes { raw in
                _ = trt_cuda_memcpy_htod(deviceBuffers[pingIdx], raw.baseAddress, raw.count)
            }

            // Chain pipeline stages
            var currentIn = deviceBuffers[pingIdx]
            var currentOut = deviceBuffers[pongIdx]

            for _ in 0..<pipelineStages {
                try await asyncContext.enqueueDevice(
                    inputs: ["input": (address: currentIn, length: byteCount)],
                    outputs: ["output": (address: currentOut, length: byteCount)],
                    synchronously: false
                )
                swap(&currentIn, &currentOut)
            }

            try await asyncContext.recordEvent(event)

            // Overlap: prepare next batch while GPU executes
            // (In real use, you'd prepare next input here)

            // Sync before reading results
            if (iter + 1) % 10 == 0 {
                try event.synchronize()
            }
        }

        try event.synchronize()
        let doubleBufferDuration = ContinuousClock.now - doubleBufferStart
        print("   Double-buffer time: \(doubleBufferDuration)")

        // Step 7: Verify correctness
        print("\n7. Verifying results...")
        let match = hostResult == input  // Identity network, should match
        print("   Output matches input: \(match ? "YES" : "NO")")

        // Step 8: Summary
        print("\n8. Performance Summary:")
        let hostThroughput = Double(iterations * pipelineStages) / durationToSeconds(hostDuration)
        let deviceThroughput = Double(iterations * pipelineStages) / durationToSeconds(deviceDuration)
        let speedup = durationToSeconds(hostDuration) / durationToSeconds(deviceDuration)

        print("   ┌─────────────────────┬─────────────────────┬─────────────────┐")
        print("   │ Pipeline Mode       │ Total Time          │ Stages/sec      │")
        print("   ├─────────────────────┼─────────────────────┼─────────────────┤")
        print("   │ Host (H2D/D2H each) │ \(formatDuration(hostDuration).padding(toLength: 19, withPad: " ", startingAt: 0)) │ \(formatDouble(hostThroughput, decimals: 0).padding(toLength: 15, withPad: " ", startingAt: 0)) │")
        print("   │ Device (GPU only)   │ \(formatDuration(deviceDuration).padding(toLength: 19, withPad: " ", startingAt: 0)) │ \(formatDouble(deviceThroughput, decimals: 0).padding(toLength: 15, withPad: " ", startingAt: 0)) │")
        print("   │ Double-buffered     │ \(formatDuration(doubleBufferDuration).padding(toLength: 19, withPad: " ", startingAt: 0)) │ -               │")
        print("   └─────────────────────┴─────────────────────┴─────────────────┘")
        print("\n   Speedup from device memory: \(formatDouble(speedup, decimals: 2))x")

        // Memory transfer analysis
        let transfersPerIterHost = pipelineStages * 2  // H2D + D2H per stage
        let transfersPerIterDevice = 2  // Just initial H2D and final D2H
        let transferReduction = Double(transfersPerIterHost) / Double(transfersPerIterDevice)
        print("   Memory transfers reduced: \(transfersPerIterHost) -> \(transfersPerIterDevice) per iteration (\(formatDouble(transferReduction, decimals: 1))x)")

        print("\n=== Device Memory Pipeline Complete ===")

#else
        print("This example requires TensorRTLLMNative (Linux with TensorRT)")
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
        if seconds < 0.001 {
            return "\(formatDouble(seconds * 1_000_000, decimals: 1)) µs"
        } else if seconds < 1 {
            return "\(formatDouble(seconds * 1000, decimals: 2)) ms"
        } else {
            return "\(formatDouble(seconds, decimals: 3)) s"
        }
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
