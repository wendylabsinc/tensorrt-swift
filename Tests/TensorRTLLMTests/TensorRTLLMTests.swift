import Testing
@testable import TensorRTLLM

@Test("TensorRT-LLM runtime probe") func tensorRTLLMProbe() async throws {
    let version = try TensorRTLLMRuntimeProbe.inferRuntimeVersion()
    #expect(version.major > 0)
}
