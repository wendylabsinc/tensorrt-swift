import Testing
import FoundationEssentials
@testable import TensorRT

#if canImport(TensorRTNative)
import TensorRTNative

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

@Test("TensorRT runtime create/destroy") func tensorRTRuntimeLifecycle() async throws {
    _ = try TensorRTSystem.Runtime()
    #expect(true)
}

@Test("TensorRT builder create/destroy") func tensorRTBuilderLifecycle() async throws {
    _ = try TensorRTSystem.Builder()
    #expect(true)
}

@Test("TensorRT plugins initialize") func tensorRTPluginInitialization() async throws {
    try TensorRTSystem.initializePlugins()
    #expect(true)
}

@Test("Loading missing plugin library fails") func tensorRTPluginLoadMissingLibrary() async throws {
    let missingPath = FileManager.default.temporaryDirectory
        .appendingPathComponent("tensorrt-swift-\(UUID().uuidString)")
        .appendingPathComponent("libmissing.so")
        .path

    do {
        try TensorRTSystem.loadPluginLibrary(missingPath)
        #expect(false, "Expected loadPluginLibrary to throw for missing path: \(missingPath)")
    } catch {
        #expect(true)
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

    try input.withUnsafeBytes { raw in
        let status = trt_cuda_memcpy_htod(dIn, raw.baseAddress, raw.count)
        #expect(status == 0)
    }

    try await context.enqueueDevice(
        inputs: ["input": (address: dIn, length: byteCount)],
        outputs: ["output": (address: dOut, length: byteCount)],
        synchronously: true
    )

    var output = [Float](repeating: 0, count: input.count)
    try output.withUnsafeMutableBytes { raw in
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
