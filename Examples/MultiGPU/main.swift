// MultiGPU - Distribute inference across multiple GPUs
//
// This example demonstrates:
// 1. Querying available CUDA devices
// 2. Creating engines on specific GPUs
// 3. Load balancing across multiple GPUs
// 4. Parallel inference on multiple devices
//
// Run with: swift run MultiGPU

import TensorRTLLM
import FoundationEssentials

#if canImport(TensorRTLLMNative)
import TensorRTLLMNative
#endif

@main
struct MultiGPU {
    static func main() async throws {
        print("=== Multi-GPU Example ===\n")

#if canImport(TensorRTLLMNative)
        // Step 1: Query available GPUs
        print("1. Querying CUDA devices...")
        let deviceCount = try TensorRTLLMSystem.cudaDeviceCount()
        print("   Available GPUs: \(deviceCount)")

        if deviceCount < 1 {
            print("   No GPUs available. Exiting.")
            return
        }

        // Step 2: Build engine plan (shared across GPUs)
        print("\n2. Building shared engine plan...")
        let elementCount = 2048
        let plan = try TensorRTLLMSystem.buildIdentityEnginePlan(elementCount: elementCount)
        print("   Engine plan size: \(plan.count) bytes")

        // Step 3: Create engines on each GPU
        print("\n3. Creating engines on each GPU...")
        let runtime = TensorRTLLMRuntime()

        var engines: [(gpu: Int, engine: Engine, context: ExecutionContext)] = []

        for gpuIndex in 0..<deviceCount {
            let config = EngineLoadConfiguration(device: DeviceSelection(gpu: gpuIndex))
            let engine = try runtime.deserializeEngine(from: plan, configuration: config)
            let context = try engine.makeExecutionContext()

            engines.append((gpuIndex, engine, context))
            print("   GPU \(gpuIndex): Engine loaded and context created")
        }

        // Step 4: Single-GPU baseline
        print("\n4. Single-GPU baseline benchmark...")
        let numBatches = 100
        let input: [Float] = (0..<elementCount).map { Float($0) }

        let singleGPUStart = ContinuousClock.now
        let ctx0 = engines[0].context

        for _ in 0..<numBatches {
            var output: [Float] = []
            try await ctx0.enqueueF32(
                inputName: "input",
                input: input,
                outputName: "output",
                output: &output
            )
        }

        let singleGPUDuration = ContinuousClock.now - singleGPUStart
        print("   Single GPU (\(numBatches) batches): \(singleGPUDuration)")
        print("   Avg latency: \(singleGPUDuration / numBatches)")

        // Step 5: Multi-GPU round-robin (if multiple GPUs available)
        if deviceCount > 1 {
            print("\n5. Multi-GPU round-robin benchmark...")

            let multiGPUStart = ContinuousClock.now

            for i in 0..<numBatches {
                let gpuIndex = i % deviceCount
                let ctx = engines[gpuIndex].context

                var output: [Float] = []
                try await ctx.enqueueF32(
                    inputName: "input",
                    input: input,
                    outputName: "output",
                    output: &output
                )
            }

            let multiGPUDuration = ContinuousClock.now - multiGPUStart
            print("   Multi-GPU round-robin (\(numBatches) batches): \(multiGPUDuration)")
            print("   Avg latency: \(multiGPUDuration / numBatches)")
            print("   Speedup: \(formatDouble(durationToSeconds(singleGPUDuration) / durationToSeconds(multiGPUDuration), decimals: 2))x")

            // Step 6: Simulate parallel inference
            print("\n6. Parallel inference simulation...")
            print("   Distributing \(numBatches) batches across \(deviceCount) GPUs")

            let batchesPerGPU = numBatches / deviceCount
            print("   Batches per GPU: \(batchesPerGPU)")

            // In a real scenario, you'd use Swift concurrency to run these in parallel
            // Here we simulate by timing sequential execution per GPU
            var gpuTimes: [(gpu: Int, duration: Duration)] = []

            for (gpuIndex, _, ctx) in engines {
                let gpuStart = ContinuousClock.now

                for _ in 0..<batchesPerGPU {
                    var output: [Float] = []
                    try await ctx.enqueueF32(
                        inputName: "input",
                        input: input,
                        outputName: "output",
                        output: &output
                    )
                }

                let gpuDuration = ContinuousClock.now - gpuStart
                gpuTimes.append((gpuIndex, gpuDuration))
                print("   GPU \(gpuIndex): \(batchesPerGPU) batches in \(gpuDuration)")
            }

            // In true parallel, total time would be max(gpu_times)
            let maxGPUTime = gpuTimes.map { durationToSeconds($0.duration) }.max() ?? 0
            print("   Parallel execution time (theoretical): \(formatDouble(maxGPUTime * 1000, decimals: 2)) ms")

        } else {
            print("\n5. Skipping multi-GPU tests (only 1 GPU available)")
            print("   To test multi-GPU, run on a system with multiple NVIDIA GPUs")
        }

        // Step 7: Device selection demo
        print("\n7. Device selection patterns:")

        print("   a) Explicit GPU selection:")
        for gpuIndex in 0..<min(deviceCount, 2) {
            print("      DeviceSelection(gpu: \(gpuIndex)) -> GPU \(gpuIndex)")
        }

        print("\n   b) Round-robin load balancing:")
        print("      for i in 0..<batches { gpu = i % deviceCount }")

        print("\n   c) Least-loaded selection (conceptual):")
        print("      Track queue depth per GPU, route to least busy")

        // Step 8: GPU memory info (simulated since we can't query directly)
        print("\n8. GPU Configuration Summary:")
        print("   ┌─────────┬─────────────────┬─────────────────┐")
        print("   │ GPU     │ Engine Loaded   │ Status          │")
        print("   ├─────────┼─────────────────┼─────────────────┤")
        for (gpuIndex, _, _) in engines {
            print("   │ GPU \(gpuIndex)   │ Yes             │ Ready           │")
        }
        print("   └─────────┴─────────────────┴─────────────────┘")

        // Step 9: Best practices
        print("\n9. Multi-GPU Best Practices:")
        print("   - Build engine once, load on each GPU")
        print("   - Use DeviceSelection to target specific GPUs")
        print("   - Implement load balancing for optimal utilization")
        print("   - Consider data parallelism for batch processing")
        print("   - Monitor GPU memory to avoid OOM")
        print("   - Use CUDA streams for async multi-GPU operations")

        print("\n=== Multi-GPU Example Complete ===")

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
}
