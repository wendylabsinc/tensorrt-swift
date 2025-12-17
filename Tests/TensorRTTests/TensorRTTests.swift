import Testing
@testable import TensorRT

@Test("TensorRT runtime probe") func tensorRTProbe() async throws {
    let version = try TensorRTRuntimeProbe.inferRuntimeVersion()
    #expect(version.major > 0)
}
