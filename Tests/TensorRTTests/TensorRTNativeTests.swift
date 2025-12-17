import Testing
import FoundationEssentials
@testable import TensorRT

#if canImport(TensorRTNative)

@Test("TensorRT linked version") func tensorRTLinkedVersion() async throws {
    let version = try TensorRTSystem.linkedRuntimeVersion()
    #expect(version.major > 0)
}

@Test("TensorRT runtime create/destroy") func tensorRTRuntimeLifecycle() async throws {
    _ = try TensorRTSystem.Runtime()
    #expect(true)
}

@Test("TensorRT builder create/destroy") func tensorRTBuilderLifecycle() async throws {
    _ = try TensorRTSystem.Builder()
    #expect(true)
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

#endif
