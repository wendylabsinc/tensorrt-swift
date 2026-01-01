// HelloTensorRT - Minimal TensorRT Swift Example
//
// This example demonstrates the basics:
// 1. Probing the TensorRT runtime version
// 2. Building a simple identity engine
// 3. Running inference on the GPU
//
// Run with: ./scripts/swiftw run HelloTensorRT
import TensorRT
@main
struct HelloTensorRT {
    static func main() async throws {
        print("=== HelloTensorRT ===\n")

        // Step 1: Probe TensorRT version (dynamic dlopen)
        print("1. Probing TensorRT runtime...")
        let probedVersion = try TensorRTRuntimeProbe.inferRuntimeVersion()
        print("   Probed version: \(probedVersion)")

        // Step 2: Get linked library version
        print("\n2. Querying linked TensorRT version...")
        let linkedVersion = try TensorRTSystem.linkedRuntimeVersion()
        print("   Linked version: \(linkedVersion)")

        // Step 3: Check CUDA device availability
        print("\n3. Checking CUDA devices...")
        let deviceCount = try TensorRTSystem.cudaDeviceCount()
        print("   CUDA devices available: \(deviceCount)")

        guard deviceCount > 0 else {
            print("   No CUDA devices found. Exiting.")
            return
        }

        // Step 4: Build a minimal identity engine
        print("\n4. Building identity engine (8 elements)...")
        let elementCount = 8
        let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: elementCount)
        print("   Engine plan size: \(plan.count) bytes")

        // Step 5: Deserialize and inspect the engine
        print("\n5. Deserializing engine...")
        let runtime = TensorRTRuntime()
        let engine = try runtime.deserializeEngine(from: plan)

        print("   Inputs:")
        for input in engine.description.inputs {
            print("     - \(input.name): \(input.descriptor.shape.dimensions) (\(input.descriptor.dataType))")
        }
        print("   Outputs:")
        for output in engine.description.outputs {
            print("     - \(output.name): \(output.descriptor.shape.dimensions) (\(output.descriptor.dataType))")
        }

        // Step 6: Run inference using the low-level API
        print("\n6. Running inference (low-level API)...")
        let input: [Float] = (0..<elementCount).map { Float($0) * 1.5 }
        print("   Input:  \(input)")

        let output = try TensorRTSystem.runIdentityPlanF32(plan: plan, input: input)
        print("   Output: \(output)")

        let match = input == output
        print("   Input == Output: \(match ? "YES" : "NO")")

        // Step 7: Run inference using ExecutionContext
        print("\n7. Running inference (ExecutionContext API)...")
        let context = try engine.makeExecutionContext()

        var contextOutput: [Float] = []
        try await context.enqueueF32(
            inputName: "input",
            input: input,
            outputName: "output",
            output: &contextOutput
        )
        print("   Input:  \(input)")
        print("   Output: \(contextOutput)")

        let contextMatch = input == contextOutput
        print("   Input == Output: \(contextMatch ? "YES" : "NO")")

        // Summary
        print("\n=== Summary ===")
        print("TensorRT version: \(linkedVersion.major).\(linkedVersion.minor).\(linkedVersion.patch)")
        print("CUDA devices: \(deviceCount)")
        print("Identity engine test: \(match && contextMatch ? "PASSED" : "FAILED")")
        print("\nHello, TensorRT!")
    }
}
