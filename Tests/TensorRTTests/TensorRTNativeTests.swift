import Testing
import FoundationEssentials
@testable import TensorRT

#if canImport(TensorRTNative)
import TensorRTNative

private let staticOnnxBase64 = "CAc6XQoZCgVpbnB1dBIGb3V0cHV0IghJZGVudGl0eRINSWRlbnRpdHlHcmFwaFoXCgVpbnB1dBIOCgwIARIICgIIAQoCCAhiGAoGb3V0cHV0Eg4KDAgBEggKAggBCgIICEIECgAQDQ=="
private let dynamicOnnxBase64 = "CAc6VAoZCgVpbnB1dBIGb3V0cHV0IghJZGVudGl0eRIQRHluSWRlbnRpdHlHcmFwaFoRCgVpbnB1dBIICgYIARICCgBiEgoGb3V0cHV0EggKBggBEgIKAEIECgAQDQ=="

@Test("TensorRT linked version") func tensorRTLinkedVersion() async throws {
    let version = try TensorRTSystem.linkedRuntimeVersion()
    #expect(version.major > 0)
}

@Test("Engine device selection is respected") func tensorRTDeviceSelectionRespected() async throws {
    let deviceCount = try TensorRTSystem.cudaDeviceCount()
    #expect(deviceCount >= 1)
    let chosen = (deviceCount >= 2) ? 1 : 0

    let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: 8)
    let runtime = TensorRTRuntime()
    let engine = try runtime.deserializeEngine(from: plan, configuration: EngineLoadConfiguration(device: DeviceSelection(gpu: chosen)))
    let context = try engine.makeExecutionContext()

    let nativeIdx = try await context.nativeDeviceIndexForTesting()
    #expect(nativeIdx == Int32(chosen))

    let input: [Float] = (0..<8).map(Float.init)
    let inputData = input.withUnsafeBufferPointer { buffer in
        Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
    }

    let inputDescriptor = engine.description.inputs[0].descriptor
    let batch = InferenceBatch(inputs: ["input": TensorValue(descriptor: inputDescriptor, storage: .host(inputData))])
    let result = try await context.enqueue(batch, synchronously: true)
    guard let outputValue = result.outputs["output"] else {
        throw TensorRTError.invalidBinding("Missing output tensor")
    }
    guard case .host(let outData) = outputValue.storage else {
        throw TensorRTError.notImplemented("Expected host output from identity inference")
    }

    var output = [Float](repeating: 0, count: input.count)
    output.withUnsafeMutableBytes { outBytes in
        outData.withUnsafeBytes { inBytes in
            outBytes.copyBytes(from: inBytes.prefix(outBytes.count))
        }
    }
    #expect(output == input)
}

@Test("Build engine from ONNX and execute") func tensorRTBuildFromONNX() async throws {
    // A minimal ONNX identity model (opset 13) with input/output [1,8] float.
    guard let onnxData = Data(base64Encoded: staticOnnxBase64) else {
        throw TensorRTError.runtimeUnavailable("Failed to decode embedded ONNX fixture.")
    }

    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("tensorrt-swift-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tmpURL, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmpURL) }

    let onnxURL = tmpURL.appendingPathComponent("identity.onnx")
    try onnxData.write(to: onnxURL)

    let runtime = TensorRTRuntime()
    let engine = try runtime.buildEngine(onnxURL: onnxURL, options: EngineBuildOptions(precision: [.fp32]))
    #expect(engine.serialized != nil)
    #expect(engine.description.inputs.count == 1)
    #expect(engine.description.outputs.count == 1)

    let input: [Float] = (0..<8).map(Float.init)
    let inputData = input.withUnsafeBufferPointer { buffer in
        Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
    }

    let context = try engine.makeExecutionContext()
    let inputDescriptor = engine.description.inputs[0].descriptor

    let batch = InferenceBatch(
        inputs: [
            inputDescriptor.name: TensorValue(descriptor: inputDescriptor, storage: .host(inputData)),
        ]
    )

    let result = try await context.enqueue(batch, synchronously: true)
    guard let outputValue = result.outputs[engine.description.outputs[0].name] else {
        throw TensorRTError.invalidBinding("Missing output tensor")
    }

    let outputData: Data
    switch outputValue.storage {
    case .host(let data):
        outputData = data
    default:
        throw TensorRTError.notImplemented("Expected host output from ONNX identity inference")
    }

    var output = [Float](repeating: 0, count: input.count)
    output.withUnsafeMutableBytes { outBytes in
        outputData.withUnsafeBytes { inBytes in
            outBytes.copyBytes(from: inBytes.prefix(outBytes.count))
        }
    }

    #expect(output == input)
}

@Test("ONNX engine save/load round-trip") func tensorRTONNXEngineSaveLoad() async throws {
    // A minimal ONNX identity model (opset 13) with input/output [1,8] float.
    guard let onnxData = Data(base64Encoded: staticOnnxBase64) else {
        throw TensorRTError.runtimeUnavailable("Failed to decode embedded ONNX fixture.")
    }

    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("tensorrt-swift-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tmpURL, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmpURL) }

    let onnxURL = tmpURL.appendingPathComponent("identity.onnx")
    try onnxData.write(to: onnxURL)

    let runtime = TensorRTRuntime()
    let engine = try runtime.buildEngine(onnxURL: onnxURL, options: EngineBuildOptions(precision: [.fp32]))
    let planURL = tmpURL.appendingPathComponent("identity.plan")
    try engine.save(to: planURL)

    let loaded = try Engine.load(from: planURL)
    #expect(loaded.serialized != nil)
    #expect(loaded.description.inputs.count == 1)
    #expect(loaded.description.outputs.count == 1)

    let input: [Float] = (0..<8).map(Float.init)
    let inputData = input.withUnsafeBufferPointer { buffer in
        Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
    }

    let context = try loaded.makeExecutionContext()
    let inputDescriptor = loaded.description.inputs[0].descriptor
    let batch = InferenceBatch(inputs: ["input": TensorValue(descriptor: inputDescriptor, storage: .host(inputData))])
    let result = try await context.enqueue(batch, synchronously: true)
    guard let outputValue = result.outputs["output"] else {
        throw TensorRTError.invalidBinding("Missing output tensor")
    }

    let outputData: Data
    switch outputValue.storage {
    case .host(let data):
        outputData = data
    default:
        throw TensorRTError.notImplemented("Expected host output from loaded engine")
    }

    var output = [Float](repeating: 0, count: input.count)
    output.withUnsafeMutableBytes { outBytes in
        outputData.withUnsafeBytes { inBytes in
            outBytes.copyBytes(from: inBytes.prefix(outBytes.count))
        }
    }

    #expect(output == input)
}

@Test("Build dynamic ONNX with shape hints and execute") func tensorRTDynamicONNXShapeHints() async throws {
    // A minimal ONNX identity model (opset 13) with input/output [dynamic] float.
    guard let onnxData = Data(base64Encoded: dynamicOnnxBase64) else {
        throw TensorRTError.runtimeUnavailable("Failed to decode embedded ONNX dynamic fixture.")
    }

    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("tensorrt-swift-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tmpURL, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmpURL) }

    let onnxURL = tmpURL.appendingPathComponent("dynamic_identity.onnx")
    try onnxData.write(to: onnxURL)

    let runtime = TensorRTRuntime()
    let shapeHints: [String: TensorShape] = ["input": TensorShape([8])]
    let engine = try runtime.buildEngine(
        onnxURL: onnxURL,
        options: EngineBuildOptions(precision: [.fp32], shapeHints: shapeHints)
    )
    #expect(engine.serialized != nil)
    #expect(engine.description.inputs.count == 1)
    #expect(engine.description.outputs.count == 1)
    #expect(engine.description.profileNames.count >= 1)

    let context = try engine.makeExecutionContext()
    let inputDescriptor = engine.description.inputs[0].descriptor
    try await context.reshape(bindings: ["input": TensorShape([8])])

    let input: [Float] = (0..<8).map(Float.init)
    let inputData = input.withUnsafeBufferPointer { buffer in
        Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
    }

    let batch = InferenceBatch(inputs: ["input": TensorValue(descriptor: inputDescriptor, storage: .host(inputData))])
    let result = try await context.enqueue(batch, synchronously: true)
    guard let outputValue = result.outputs["output"] else {
        throw TensorRTError.invalidBinding("Missing output tensor")
    }

    let outputData: Data
    switch outputValue.storage {
    case .host(let data):
        outputData = data
    default:
        throw TensorRTError.notImplemented("Expected host output from dynamic ONNX inference")
    }

    var output = [Float](repeating: 0, count: input.count)
    output.withUnsafeMutableBytes { outBytes in
        outputData.withUnsafeBytes { inBytes in
            outBytes.copyBytes(from: inBytes.prefix(outBytes.count))
        }
    }

    #expect(output == input)
}

@Test("Dynamic ONNX without profiles or hints fails with guidance") func tensorRTDynamicONNXMissingProfilesFails() async throws {
    // A minimal ONNX identity model (opset 13) with input/output [dynamic] float.
    guard let onnxData = Data(base64Encoded: dynamicOnnxBase64) else {
        throw TensorRTError.runtimeUnavailable("Failed to decode embedded ONNX dynamic fixture.")
    }

    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("tensorrt-swift-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tmpURL, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmpURL) }

    let onnxURL = tmpURL.appendingPathComponent("dynamic_identity.onnx")
    try onnxData.write(to: onnxURL)

    let runtime = TensorRTRuntime()
    do {
        _ = try runtime.buildEngine(onnxURL: onnxURL, options: EngineBuildOptions(precision: [.fp32]))
        #expect(Bool(false), "Expected buildEngine to fail without profiles or shapeHints.")
    } catch let error as TensorRTError {
        switch error {
        case .runtimeUnavailable(let reason):
            #expect(reason.contains("shapeHints"))
            #expect(reason.contains("profiles"))
        default:
            #expect(Bool(false), "Expected runtimeUnavailable, got \(error)")
        }
    } catch {
        #expect(Bool(false), "Expected TensorRTError, got \(error)")
    }
}

@Test("Build FP32 and FP16 ONNX engines") func tensorRTONNXPrecisionBuilds() async throws {
    // A minimal ONNX identity model (opset 13) with input/output [1,8] float.
    guard let onnxData = Data(base64Encoded: staticOnnxBase64) else {
        throw TensorRTError.runtimeUnavailable("Failed to decode embedded ONNX fixture.")
    }

    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("tensorrt-swift-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tmpURL, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmpURL) }

    let onnxURL = tmpURL.appendingPathComponent("identity.onnx")
    try onnxData.write(to: onnxURL)

    let runtime = TensorRTRuntime()
    let fp32Engine = try runtime.buildEngine(onnxURL: onnxURL, options: EngineBuildOptions(precision: [.fp32]))
    let fp16Engine = try runtime.buildEngine(onnxURL: onnxURL, options: EngineBuildOptions(precision: [.fp16]))

    #expect(fp32Engine.serialized != nil)
    #expect(fp16Engine.serialized != nil)

    func run(engine: Engine, input: [Float]) async throws -> [Float] {
        let inputData = input.withUnsafeBufferPointer { buffer in
            Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
        }
        let context = try engine.makeExecutionContext()
        let inputDescriptor = engine.description.inputs[0].descriptor
        let batch = InferenceBatch(
            inputs: [
                inputDescriptor.name: TensorValue(descriptor: inputDescriptor, storage: .host(inputData)),
            ]
        )
        let result = try await context.enqueue(batch, synchronously: true)
        guard let outputValue = result.outputs[engine.description.outputs[0].name] else {
            throw TensorRTError.invalidBinding("Missing output tensor")
        }

        let outputData: Data
        switch outputValue.storage {
        case .host(let data):
            outputData = data
        default:
            throw TensorRTError.notImplemented("Expected host output from ONNX identity inference")
        }

        var output = [Float](repeating: 0, count: input.count)
        output.withUnsafeMutableBytes { outBytes in
            outputData.withUnsafeBytes { inBytes in
                outBytes.copyBytes(from: inBytes.prefix(outBytes.count))
            }
        }
        return output
    }

    let expected = (0..<8).map(Float.init)
    #expect(try await run(engine: fp32Engine, input: expected) == expected)
    #expect(try await run(engine: fp16Engine, input: expected) == expected)
}

@Test("Multi-GPU ONNX load/execute") func tensorRTMultiGPUONNX() async throws {
    let deviceCount = try TensorRTSystem.cudaDeviceCount()
    guard deviceCount >= 2 else { return }

    let props0 = try TensorRTSystem.deviceProperties(device: 0)
    let props1 = try TensorRTSystem.deviceProperties(device: 1)
    guard props0.computeCapability == props1.computeCapability else { return }

    // A minimal ONNX identity model (opset 13) with input/output [1,8] float.
    guard let onnxData = Data(base64Encoded: staticOnnxBase64) else {
        throw TensorRTError.runtimeUnavailable("Failed to decode embedded ONNX fixture.")
    }

    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("tensorrt-swift-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tmpURL, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmpURL) }

    let onnxURL = tmpURL.appendingPathComponent("identity.onnx")
    try onnxData.write(to: onnxURL)

    let runtime = TensorRTRuntime()
    let built = try runtime.buildEngine(onnxURL: onnxURL, options: EngineBuildOptions(precision: [.fp32]))
    guard let plan = built.serialized else {
        throw TensorRTError.invalidBinding("Expected serialized plan data from ONNX build.")
    }

    let engine0 = try runtime.deserializeEngine(from: plan, configuration: EngineLoadConfiguration(device: DeviceSelection(gpu: 0)))
    let engine1 = try runtime.deserializeEngine(from: plan, configuration: EngineLoadConfiguration(device: DeviceSelection(gpu: 1)))

    let context0 = try engine0.makeExecutionContext()
    let context1 = try engine1.makeExecutionContext()

    let idx0 = try await context0.nativeDeviceIndexForTesting()
    let idx1 = try await context1.nativeDeviceIndexForTesting()
    #expect(idx0 == 0)
    #expect(idx1 == 1)

    let input: [Float] = (0..<8).map(Float.init)
    let inputData = input.withUnsafeBufferPointer { buffer in
        Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
    }

    func run(context: ExecutionContext, engine: Engine) async throws -> [Float] {
        let inputDescriptor = engine.description.inputs[0].descriptor
        let outputName = engine.description.outputs[0].name
        let batch = InferenceBatch(inputs: ["input": TensorValue(descriptor: inputDescriptor, storage: .host(inputData))])
        let result = try await context.enqueue(batch, synchronously: true)
        guard let out = result.outputs[outputName] else {
            throw TensorRTError.invalidBinding("Missing output tensor")
        }
        guard case .host(let outData) = out.storage else {
            throw TensorRTError.notImplemented("Expected host output from ONNX inference")
        }

        var output = [Float](repeating: 0, count: input.count)
        output.withUnsafeMutableBytes { outBytes in
            outData.withUnsafeBytes { inBytes in
                outBytes.copyBytes(from: inBytes.prefix(outBytes.count))
            }
        }
        return output
    }

    #expect(try await run(context: context0, engine: engine0) == input)
    #expect(try await run(context: context1, engine: engine1) == input)
}

@Test("Build dynamic ONNX with profiles and switch at runtime") func tensorRTDynamicONNXProfiles() async throws {
    // A minimal ONNX identity model (opset 13) with input/output [dynamic] float.
    guard let onnxData = Data(base64Encoded: dynamicOnnxBase64) else {
        throw TensorRTError.runtimeUnavailable("Failed to decode embedded ONNX dynamic fixture.")
    }

    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("tensorrt-swift-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tmpURL, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmpURL) }

    let onnxURL = tmpURL.appendingPathComponent("dynamic_identity.onnx")
    try onnxData.write(to: onnxURL)

    let p0 = OptimizationProfile(
        name: "0",
        axes: [:],
        bindingRanges: [
            "input": .init(min: TensorShape([1]), optimal: TensorShape([8]), max: TensorShape([16])),
        ]
    )
    let p1 = OptimizationProfile(
        name: "1",
        axes: [:],
        bindingRanges: [
            "input": .init(min: TensorShape([32]), optimal: TensorShape([32]), max: TensorShape([64])),
        ]
    )

    let runtime = TensorRTRuntime()
    let engine = try runtime.buildEngine(
        onnxURL: onnxURL,
        options: EngineBuildOptions(precision: [.fp32], profiles: [p0, p1])
    )
    #expect(engine.serialized != nil)
    #expect(engine.description.profileNames.count >= 2)
    #expect(engine.description.inputs.count == 1)
    #expect(engine.description.outputs.count == 1)
    #expect(engine.description.inputs[0].descriptor.shape.isDynamic == true)

    let context = try engine.makeExecutionContext()
    let inputDescriptor = engine.description.inputs[0].descriptor

    func run(count: Int) async throws -> [Float] {
        let input: [Float] = (0..<count).map(Float.init)
        let inputData = input.withUnsafeBufferPointer { buffer in
            Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
        }
        let batch = InferenceBatch(inputs: ["input": TensorValue(descriptor: inputDescriptor, storage: .host(inputData))])
        let result = try await context.enqueue(batch, synchronously: true)
        guard let out = result.outputs["output"] else { throw TensorRTError.invalidBinding("Missing output") }
        guard case .host(let outData) = out.storage else { throw TensorRTError.notImplemented("Expected host output") }

        var output = [Float](repeating: 0, count: count)
        output.withUnsafeMutableBytes { outBytes in
            outData.withUnsafeBytes { inBytes in
                outBytes.copyBytes(from: inBytes.prefix(outBytes.count))
            }
        }
        return output
    }

    try await context.setOptimizationProfile(named: "0")
    try await context.reshape(bindings: ["input": TensorShape([8])])
    #expect(try await run(count: 8) == (0..<8).map(Float.init))

    try await context.setOptimizationProfile(named: "1")
    try await context.reshape(bindings: ["input": TensorShape([32])])
    #expect(try await run(count: 32) == (0..<32).map(Float.init))
}

@Test("Multiple contexts keep independent optimization profiles") func tensorRTProfileIsolationAcrossContexts() async throws {
    // A minimal ONNX identity model (opset 13) with input/output [dynamic] float.
    guard let onnxData = Data(base64Encoded: dynamicOnnxBase64) else {
        throw TensorRTError.runtimeUnavailable("Failed to decode embedded ONNX dynamic fixture.")
    }

    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("tensorrt-swift-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tmpURL, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmpURL) }

    let onnxURL = tmpURL.appendingPathComponent("dynamic_identity.onnx")
    try onnxData.write(to: onnxURL)

    let p0 = OptimizationProfile(
        name: "0",
        axes: [:],
        bindingRanges: [
            "input": .init(min: TensorShape([1]), optimal: TensorShape([8]), max: TensorShape([16])),
        ]
    )
    let p1 = OptimizationProfile(
        name: "1",
        axes: [:],
        bindingRanges: [
            "input": .init(min: TensorShape([32]), optimal: TensorShape([32]), max: TensorShape([64])),
        ]
    )

    let runtime = TensorRTRuntime()
    let engine = try runtime.buildEngine(
        onnxURL: onnxURL,
        options: EngineBuildOptions(precision: [.fp32], profiles: [p0, p1])
    )
    #expect(engine.description.profileNames.count >= 2)

    let contextA = try engine.makeExecutionContext()
    let contextB = try engine.makeExecutionContext()
    let inputDescriptor = engine.description.inputs[0].descriptor
    let outputName = engine.description.outputs[0].name

    func run(context: ExecutionContext, count: Int) async throws -> [Float] {
        let input: [Float] = (0..<count).map(Float.init)
        let inputData = input.withUnsafeBufferPointer { buffer in
            Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
        }
        let batch = InferenceBatch(inputs: ["input": TensorValue(descriptor: inputDescriptor, storage: .host(inputData))])
        let result = try await context.enqueue(batch, synchronously: true)
        guard let out = result.outputs[outputName] else { throw TensorRTError.invalidBinding("Missing output") }
        guard case .host(let outData) = out.storage else { throw TensorRTError.notImplemented("Expected host output") }

        var output = [Float](repeating: 0, count: count)
        output.withUnsafeMutableBytes { outBytes in
            outData.withUnsafeBytes { inBytes in
                outBytes.copyBytes(from: inBytes.prefix(outBytes.count))
            }
        }
        return output
    }

    try await contextA.setOptimizationProfile(named: "0")
    try await contextA.reshape(bindings: ["input": TensorShape([8])])

    try await contextB.setOptimizationProfile(named: "1")
    try await contextB.reshape(bindings: ["input": TensorShape([32])])

    #expect(try await run(context: contextA, count: 8) == (0..<8).map(Float.init))
    #expect(try await run(context: contextB, count: 32) == (0..<32).map(Float.init))

    try await contextA.reshape(bindings: ["input": TensorShape([16])])
    #expect(try await run(context: contextA, count: 16) == (0..<16).map(Float.init))
    #expect(try await run(context: contextB, count: 32) == (0..<32).map(Float.init))
}

@Test("TensorRT runtime create/destroy") func tensorRTRuntimeLifecycle() async throws {
    _ = try TensorRTSystem.Runtime()
    #expect(Bool(true))
}

@Test("TensorRT builder create/destroy") func tensorRTBuilderLifecycle() async throws {
    _ = try TensorRTSystem.Builder()
    #expect(Bool(true))
}

@Test("TensorRT plugins initialize") func tensorRTPluginInitialization() async throws {
    try TensorRTSystem.initializePlugins()
    #expect(Bool(true))
}

@Test("Loading missing plugin library fails") func tensorRTPluginLoadMissingLibrary() async throws {
    let missingPath = FileManager.default.temporaryDirectory
        .appendingPathComponent("tensorrt-swift-\(UUID().uuidString)")
        .appendingPathComponent("libmissing.so")
        .path

    do {
        try TensorRTSystem.loadPluginLibrary(missingPath)
        #expect(Bool(false), "Expected loadPluginLibrary to throw for missing path: \(missingPath)")
    } catch {
        #expect(Bool(true))
    }
}

@Test("Linked version matches probe when available") func tensorRTVersionConsistency() async throws {
    let linked = try TensorRTSystem.linkedRuntimeVersion()
    let probed = try TensorRTRuntimeProbe.inferRuntimeVersion()
    #expect(linked.major == probed.major)
    #expect(linked.minor == probed.minor)
}

@Test("Build, deserialize, and execute identity engine") func tensorRTIdentityEngineEndToEnd() async throws {
    let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: 8)

    let runtime = TensorRTRuntime()
    let engine = try runtime.deserializeEngine(from: plan)
    #expect(engine.description.inputs.count == 1)
    #expect(engine.description.outputs.count == 1)
    #expect(engine.description.inputs.first?.name == "input")
    #expect(engine.description.outputs.first?.name == "output")

    let input: [Float] = (0..<8).map(Float.init)
    let inputData = input.withUnsafeBufferPointer { buffer in
        Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
    }

    let context = try engine.makeExecutionContext()
    let inputDescriptor = engine.description.inputs[0].descriptor

    let batch = InferenceBatch(
        inputs: [
            "input": TensorValue(descriptor: inputDescriptor, storage: .host(inputData)),
        ]
    )

    let result = try await context.enqueue(batch, synchronously: true)
    guard let outputValue = result.outputs["output"] else {
        throw TensorRTError.invalidBinding("Missing output tensor")
    }

    let outputData: Data
    switch outputValue.storage {
    case .host(let data):
        outputData = data
    default:
        throw TensorRTError.notImplemented("Expected host output from identity inference")
    }

    var output = [Float](repeating: 0, count: input.count)
    output.withUnsafeMutableBytes { outBytes in
        outputData.withUnsafeBytes { inBytes in
            outBytes.copyBytes(from: inBytes.prefix(outBytes.count))
        }
    }

    #expect(output == input)
}

@Test("ExecutionContext reuses a persistent native context") func tensorRTPersistentContextReuse() async throws {
    let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: 8)
    let engine = try TensorRTRuntime().deserializeEngine(from: plan)
    let context = try engine.makeExecutionContext()

    func run(_ values: [Float]) async throws -> [Float] {
        let inputData = values.withUnsafeBufferPointer { buffer in
            Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
        }
        let inputDescriptor = engine.description.inputs[0].descriptor
        let batch = InferenceBatch(inputs: ["input": TensorValue(descriptor: inputDescriptor, storage: .host(inputData))])
        let result = try await context.enqueue(batch, synchronously: true)
        guard let outputValue = result.outputs["output"] else {
            throw TensorRTError.invalidBinding("Missing output tensor")
        }
        guard case .host(let outputData) = outputValue.storage else {
            throw TensorRTError.notImplemented("Expected host output from identity inference")
        }
        var output = [Float](repeating: 0, count: values.count)
        output.withUnsafeMutableBytes { outBytes in
            outputData.withUnsafeBytes { inBytes in
                outBytes.copyBytes(from: inBytes.prefix(outBytes.count))
            }
        }
        return output
    }

    let first = (0..<8).map(Float.init)
    let second = (0..<8).map { Float($0) * 2 }

    #expect(try await run(first) == first)
    #expect(try await run(second) == second)
}

@Test("Dynamic reshape updates output sizes") func tensorRTDynamicReshapeIdentity() async throws {
    let plan = try TensorRTSystem.buildDynamicIdentityEnginePlanF32(min: 1, opt: 8, max: 16)
    let engine = try TensorRTRuntime().deserializeEngine(from: plan)
    #expect(engine.description.inputs.first?.name == "input")
    #expect(engine.description.outputs.first?.name == "output")

    let context = try engine.makeExecutionContext()

    let inputDescriptor = engine.description.inputs[0].descriptor
    #expect(inputDescriptor.shape.isDynamic == true)

    // First shape.
    try await context.reshape(bindings: ["input": TensorShape([8])])
    let inputA: [Float] = (0..<8).map(Float.init)
    let inputDataA = inputA.withUnsafeBufferPointer { buffer in
        Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
    }
    let batchA = InferenceBatch(inputs: ["input": TensorValue(descriptor: inputDescriptor, storage: .host(inputDataA))])
    let resultA = try await context.enqueue(batchA, synchronously: true)
    guard let outA = resultA.outputs["output"] else { throw TensorRTError.invalidBinding("Missing output") }
    guard case .host(let outDataA) = outA.storage else { throw TensorRTError.notImplemented("Expected host output") }
    #expect(outDataA.count == inputDataA.count)

    // Second shape.
    try await context.reshape(bindings: ["input": TensorShape([16])])
    let inputB: [Float] = (0..<16).map(Float.init)
    let inputDataB = inputB.withUnsafeBufferPointer { buffer in
        Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
    }
    let batchB = InferenceBatch(inputs: ["input": TensorValue(descriptor: inputDescriptor, storage: .host(inputDataB))])
    let resultB = try await context.enqueue(batchB, synchronously: true)
    guard let outB = resultB.outputs["output"] else { throw TensorRTError.invalidBinding("Missing output") }
    guard case .host(let outDataB) = outB.storage else { throw TensorRTError.notImplemented("Expected host output") }
    #expect(outDataB.count == inputDataB.count)
}

@Test("Optimization profile switching enables different shape ranges") func tensorRTOptimizationProfileSwitching() async throws {
    let plan = try TensorRTSystem.buildDualProfileIdentityEnginePlanF32(
        profile0: (min: 1, opt: 8, max: 16),
        profile1: (min: 32, opt: 32, max: 64)
    )
    let engine = try TensorRTRuntime().deserializeEngine(from: plan)
    #expect(engine.description.profileNames.count >= 2)

    let context = try engine.makeExecutionContext()
    let inputDescriptor = engine.description.inputs[0].descriptor

    func run(count: Int) async throws -> [Float] {
        let input: [Float] = (0..<count).map(Float.init)
        let inputData = input.withUnsafeBufferPointer { buffer in
            Data(bytes: buffer.baseAddress!, count: buffer.count * MemoryLayout<Float>.stride)
        }
        let batch = InferenceBatch(inputs: ["input": TensorValue(descriptor: inputDescriptor, storage: .host(inputData))])
        let result = try await context.enqueue(batch, synchronously: true)
        guard let out = result.outputs["output"] else { throw TensorRTError.invalidBinding("Missing output") }
        guard case .host(let outData) = out.storage else { throw TensorRTError.notImplemented("Expected host output") }

        var output = [Float](repeating: 0, count: count)
        output.withUnsafeMutableBytes { outBytes in
            outData.withUnsafeBytes { inBytes in
                outBytes.copyBytes(from: inBytes.prefix(outBytes.count))
            }
        }
        return output
    }

    try await context.setOptimizationProfile(named: "0")
    try await context.reshape(bindings: ["input": TensorShape([8])])
    #expect(try await run(count: 8) == (0..<8).map(Float.init))

    try await context.setOptimizationProfile(named: "1")
    try await context.reshape(bindings: ["input": TensorShape([32])])
    #expect(try await run(count: 32) == (0..<32).map(Float.init))
}

@Test("Device buffer enqueue executes on GPU") func tensorRTDeviceBufferEnqueue() async throws {
    let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: 8)
    let engine = try TensorRTRuntime().deserializeEngine(from: plan)
    let context = try engine.makeExecutionContext()

    let input: [Float] = (0..<8).map(Float.init)
    let byteCount = input.count * MemoryLayout<Float>.stride

    var dIn: UInt64 = 0
    var dOut: UInt64 = 0
    #expect(trt_cuda_malloc(byteCount, &dIn) == 0)
    #expect(trt_cuda_malloc(byteCount, &dOut) == 0)
    defer {
        _ = trt_cuda_free(dOut)
        _ = trt_cuda_free(dIn)
    }

    input.withUnsafeBytes { raw in
        let status = trt_cuda_memcpy_htod(dIn, raw.baseAddress, raw.count)
        #expect(status == 0)
    }

    try await context.enqueueDevice(
        inputs: ["input": (address: dIn, length: byteCount)],
        outputs: ["output": (address: dOut, length: byteCount)],
        synchronously: true
    )

    var output = [Float](repeating: 0, count: input.count)
    output.withUnsafeMutableBytes { raw in
        let status = trt_cuda_memcpy_dtoh(raw.baseAddress, dOut, raw.count)
        #expect(status == 0)
    }

    #expect(output == input)
}

@Test("External CUDA stream executes asynchronously") func tensorRTExternalStreamAsyncEnqueue() async throws {
    let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: 8)
    let engine = try TensorRTRuntime().deserializeEngine(from: plan)

    var stream: UInt64 = 0
    #expect(trt_cuda_stream_create(&stream) == 0)
    defer { _ = trt_cuda_stream_destroy(stream) }

    let context = try engine.makeExecutionContext(queue: .external(streamIdentifier: stream))

    let input: [Float] = (0..<8).map(Float.init)
    let byteCount = input.count * MemoryLayout<Float>.stride

    var dIn: UInt64 = 0
    var dOut: UInt64 = 0
    #expect(trt_cuda_malloc(byteCount, &dIn) == 0)
    #expect(trt_cuda_malloc(byteCount, &dOut) == 0)
    defer {
        _ = trt_cuda_free(dOut)
        _ = trt_cuda_free(dIn)
    }

    input.withUnsafeBytes { raw in
        #expect(trt_cuda_memcpy_htod(dIn, raw.baseAddress, raw.count) == 0)
    }

    // Enqueue without synchronization; caller controls stream synchronization.
    try await context.enqueueDevice(
        inputs: ["input": (address: dIn, length: byteCount)],
        outputs: ["output": (address: dOut, length: byteCount)],
        synchronously: false
    )

    #expect(trt_cuda_stream_synchronize(stream) == 0)

    var output = [Float](repeating: 0, count: input.count)
    output.withUnsafeMutableBytes { raw in
        #expect(trt_cuda_memcpy_dtoh(raw.baseAddress, dOut, raw.count) == 0)
    }

    #expect(output == input)
}

@Test("CUDA event signals completion without stream sync") func tensorRTCudaEventCompletion() async throws {
    let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: 8)
    let engine = try TensorRTRuntime().deserializeEngine(from: plan)

    var stream: UInt64 = 0
    #expect(trt_cuda_stream_create(&stream) == 0)
    defer { _ = trt_cuda_stream_destroy(stream) }

    let context = try engine.makeExecutionContext(queue: .external(streamIdentifier: stream))

    let input: [Float] = (0..<8).map(Float.init)
    let byteCount = input.count * MemoryLayout<Float>.stride

    var dIn: UInt64 = 0
    var dOut: UInt64 = 0
    #expect(trt_cuda_malloc(byteCount, &dIn) == 0)
    #expect(trt_cuda_malloc(byteCount, &dOut) == 0)
    defer {
        _ = trt_cuda_free(dOut)
        _ = trt_cuda_free(dIn)
    }

    input.withUnsafeBytes { raw in
        #expect(trt_cuda_memcpy_htod(dIn, raw.baseAddress, raw.count) == 0)
    }

    try await context.enqueueDevice(
        inputs: ["input": (address: dIn, length: byteCount)],
        outputs: ["output": (address: dOut, length: byteCount)],
        synchronously: false
    )

    let event = try TensorRTSystem.CUDAEvent()
    try await context.recordEvent(event)
    _ = try event.isReady()

    try event.synchronize()

    var output = [Float](repeating: 0, count: input.count)
    output.withUnsafeMutableBytes { raw in
        #expect(trt_cuda_memcpy_dtoh(raw.baseAddress, dOut, raw.count) == 0)
    }

    #expect(output == input)
}

#endif
