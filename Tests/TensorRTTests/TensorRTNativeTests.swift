import Testing
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
    let output = try TensorRTSystem.runIdentityPlanF32(plan: plan, input: input)
    #expect(output == input)
}

#endif
