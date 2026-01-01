// ONNXInference - Load and run an ONNX model with TensorRT
//
// This example demonstrates:
// 1. Building a TensorRT engine from an ONNX file
// 2. Inspecting the engine's input/output bindings
// 3. Running inference with the high-level API
//
// Run with: ./scripts/swiftw run ONNXInference [path/to/model.onnx]
//
// If no ONNX file is provided, a minimal identity model is created for demonstration.
import TensorRT
import FoundationEssentials

@main
struct ONNXInference {
    static func main() async throws {
        print("=== ONNX Inference Example ===\n")

        // Check for command-line argument or use embedded demo model
        let onnxURL: URL
        let isDemo: Bool

        if CommandLine.arguments.count > 1 {
            onnxURL = URL(fileURLWithPath: CommandLine.arguments[1])
            isDemo = false
            print("Using ONNX model: \(onnxURL.path)")
        } else {
            // Create a temporary demo ONNX model (identity with shape [1, 8])
            onnxURL = try createDemoONNXModel()
            isDemo = true
            print("No ONNX file provided. Using embedded demo model.")
            print("Usage: ./scripts/swiftw run ONNXInference <path/to/model.onnx>\n")
        }

        defer {
            if isDemo {
                try? FileManager.default.removeItem(at: onnxURL.deletingLastPathComponent())
            }
        }

        // Step 1: Build the TensorRT engine from ONNX
        print("1. Building TensorRT engine from ONNX...")
        let buildStart = ContinuousClock.now

        let runtime = TensorRTRuntime()
        let engine = try runtime.buildEngine(
            onnxURL: onnxURL,
            options: EngineBuildOptions(
                precision: [.fp32],
                workspaceSizeBytes: 256 * 1024 * 1024  // 256 MB workspace
            )
        )

        let buildDuration = ContinuousClock.now - buildStart
        print("   Build time: \(buildDuration)")
        print("   Plan size: \(engine.serialized?.count ?? 0) bytes")

        // Step 2: Inspect the engine
        print("\n2. Engine inspection:")
        print("   Inputs (\(engine.description.inputs.count)):")
        for input in engine.description.inputs {
            let shape = input.descriptor.shape
            print("     - \(input.name)")
            print("       Shape: \(shape.dimensions)")
            print("       Type: \(input.descriptor.dataType)")
            print("       Dynamic: \(shape.isDynamic)")
        }

        print("   Outputs (\(engine.description.outputs.count)):")
        for output in engine.description.outputs {
            let shape = output.descriptor.shape
            print("     - \(output.name)")
            print("       Shape: \(shape.dimensions)")
            print("       Type: \(output.descriptor.dataType)")
        }

        // Step 3: Prepare input data
        print("\n3. Preparing input data...")
        guard let inputBinding = engine.description.inputs.first else {
            print("   Error: No input bindings found")
            return
        }

        let inputDesc = inputBinding.descriptor
        let elementCount = inputDesc.shape.elementCount

        guard elementCount > 0 else {
            print("   Error: Input has dynamic shape. Use DynamicBatching example instead.")
            return
        }

        // Create sample input data
        let inputFloats: [Float] = (0..<elementCount).map { Float($0) }
        print("   Input tensor: \(inputDesc.name)")
        print("   Shape: \(inputDesc.shape.dimensions)")
        print("   Elements: \(elementCount)")
        if elementCount <= 16 {
            print("   Values: \(inputFloats)")
        } else {
            print("   Values: [\(inputFloats.prefix(4).map { String($0) }.joined(separator: ", ")), ..., \(inputFloats.suffix(4).map { String($0) }.joined(separator: ", "))]")
        }

        // Step 4: Create execution context and run inference
        print("\n4. Running inference...")
        let context = try engine.makeExecutionContext()

        let inputData = inputFloats.withUnsafeBufferPointer { buffer in
            Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
        }

        let batch = InferenceBatch(inputs: [
            inputDesc.name: TensorValue(descriptor: inputDesc, storage: .host(inputData))
        ])

        let inferenceStart = ContinuousClock.now
        let result = try await context.enqueue(batch, synchronously: true)
        let inferenceDuration = ContinuousClock.now - inferenceStart

        print("   Inference time: \(inferenceDuration)")

        // Step 5: Process outputs
        print("\n5. Output results:")
        for (name, value) in result.outputs {
            print("   Output: \(name)")
            print("   Shape: \(value.descriptor.shape.dimensions)")

            if case .host(let data) = value.storage {
                let outputCount = data.count / MemoryLayout<Float>.stride
                var outputFloats = [Float](repeating: 0, count: outputCount)
                outputFloats.withUnsafeMutableBytes { outBytes in
                    data.withUnsafeBytes { inBytes in
                        outBytes.copyBytes(from: inBytes)
                    }
                }

                if outputCount <= 16 {
                    print("   Values: \(outputFloats)")
                } else {
                    print("   Values: [\(outputFloats.prefix(4).map { String($0) }.joined(separator: ", ")), ..., \(outputFloats.suffix(4).map { String($0) }.joined(separator: ", "))]")
                }

                // For identity model, verify output matches input
                if isDemo {
                    let match = outputFloats == inputFloats
                    print("   Verification (identity): \(match ? "PASSED" : "FAILED")")
                }
            }
        }

        // Step 6: Run multiple inferences to measure throughput
        print("\n6. Throughput test (10 iterations)...")
        let iterations = 10
        let throughputStart = ContinuousClock.now

        for _ in 0..<iterations {
            _ = try await context.enqueue(batch, synchronously: true)
        }

        let throughputDuration = ContinuousClock.now - throughputStart
        let avgLatency = throughputDuration / iterations
        let throughput = Double(iterations) / (Double(throughputDuration.components.seconds) + Double(throughputDuration.components.attoseconds) / 1e18)

        print("   Total time: \(throughputDuration)")
        print("   Avg latency: \(avgLatency)")
        print("   Throughput: \(formatDouble(throughput, decimals: 1)) inferences/sec")

        print("\n=== Done ===")
    }

    /// Creates a minimal ONNX identity model for demonstration
    static func createDemoONNXModel() throws -> URL {
        // Minimal ONNX identity model (opset 13): input [1,8] float -> output [1,8] float
        let onnxBase64 = "CAc6XQoZCgVpbnB1dBIGb3V0cHV0IghJZGVudGl0eRINSWRlbnRpdHlHcmFwaFoXCgVpbnB1dBIOCgwIARIICgIIAQoCCAhiGAoGb3V0cHV0Eg4KDAgBEggKAggBCgIICEIECgAQDQ=="

        guard let onnxData = Data(base64Encoded: onnxBase64) else {
            throw TensorRTError.runtimeUnavailable("Failed to decode embedded ONNX model")
        }

        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tensorrt-swift-onnx-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)

        let onnxURL = tmpDir.appendingPathComponent("identity.onnx")
        try onnxData.write(to: onnxURL)

        return onnxURL
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
