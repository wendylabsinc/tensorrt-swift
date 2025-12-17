import FoundationEssentials

#if canImport(TensorRTNative)
import TensorRTNative
#endif

/// Swift-first API scaffolding for working with NVIDIA TensorRT engines on Jetson-class devices.
///
/// The API favors value types, Sendable actors, and concise builders so it can be used from
/// async contexts on Linux without leaning on `Foundation`. This file focuses purely on the
/// public surface; the underlying CUDA/TensorRT bindings are intentionally left unimplemented.
/// When wiring to TensorRT, prefer Swift 6.2 interop features (C++ interop, `Span`, and inline
/// arrays) to minimize copies and keep ABI boundaries thin.
///
/// - Important: This package is a work in progress and is expected to have breaking API changes.
public enum TensorRT {}

// MARK: - Errors

/// Errors emitted by the TensorRT Swift bindings.
public enum TensorRTError: LocalizedError, Sendable {
    case notImplemented(String)
    case invalidBinding(String)
    case unsupportedPrecision(String)
    case runtimeUnavailable(String)
    case shapeMismatch(expected: TensorShape, received: TensorShape)
    case missingOptimizationProfile(String)
    case calibrationFailed(String)
    case allocatorFailed(String)
    case profileUnavailable(String)
    case invalidShapeRange(String)

    public var errorDescription: String? {
        switch self {
        case .notImplemented(let feature):
            return "\(feature) is not implemented yet."
        case .invalidBinding(let name):
            return "No binding named \(name) exists in the current engine."
        case .unsupportedPrecision(let message):
            return "Precision selection failed: \(message)."
        case .runtimeUnavailable(let reason):
            return "TensorRT runtime is unavailable: \(reason)."
        case .shapeMismatch(let expected, let received):
            return "Shape mismatch. Expected \(expected.dimensions), received \(received.dimensions)."
        case .missingOptimizationProfile(let name):
            return "Optimization profile \(name) is not loaded for this context."
        case .calibrationFailed(let reason):
            return "Calibration failed: \(reason)."
        case .allocatorFailed(let reason):
            return "Allocator failed: \(reason)."
        case .profileUnavailable(let name):
            return "Requested profile \(name) is unavailable."
        case .invalidShapeRange(let details):
            return "Invalid shape range: \(details)."
        }
    }
}

// MARK: - Tensor descriptions

/// Describes a tensor's dimensions.
///
/// TensorRT shapes typically have rank ≤ 8, so this type stores dimensions inline (see `InlineArray`)
/// to avoid heap allocations in common paths like IO reflection and per-inference metadata.
///
/// Use non-positive values to represent dynamic axes (for example batch dimension `-1`).
public struct TensorShape: Hashable, Sendable {
    public static let maxRank = 8

    public var rank: Int
    public typealias InlineDims = [8 of Int32]
    public var dims: InlineDims

    /// Creates a tensor shape with explicit dimensions. Use non-positive values to represent
    /// dynamic axes (for example, `0` or `-1` for dynamic batch size).
    public init(_ dimensions: [Int]) {
        self.rank = min(Self.maxRank, dimensions.count)
        self.dims = InlineDims(repeating: 0)
        for index in 0..<rank {
            self.dims[index] = Int32(dimensions[index])
        }
    }

    public init(rank: Int, dims: InlineDims) {
        self.rank = min(Self.maxRank, max(0, rank))
        self.dims = dims
    }

    /// Total element count assuming dynamic axes are materialized.
    public var elementCount: Int {
        guard !isDynamic else { return 0 }
        var product = 1
        for index in 0..<rank {
            product *= Int(dims[index])
        }
        return product
    }

    /// Returns true when any axis is marked as dynamic.
    public var isDynamic: Bool {
        for index in 0..<rank where dims[index] <= 0 {
            return true
        }
        return false
    }

    public var dimensions: [Int] {
        (0..<rank).map { Int(dims[$0]) }
    }

    public static func == (lhs: TensorShape, rhs: TensorShape) -> Bool {
        guard lhs.rank == rhs.rank else { return false }
        for index in 0..<TensorShape.maxRank {
            if lhs.dims[index] != rhs.dims[index] { return false }
        }
        return true
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(rank)
        for index in 0..<TensorShape.maxRank {
            hasher.combine(dims[index])
        }
    }
}

public extension TensorShape {
    /// Bounds for a dynamic axis, used when creating optimization profiles.
    struct DynamicAxisRange: Hashable, Sendable {
        public var min: Int
        public var optimal: Int
        public var max: Int

        public init(min: Int, optimal: Int, max: Int) {
            self.min = min
            self.optimal = optimal
            self.max = max
        }
    }
}

/// Describes a tensor's explicit element strides.
///
/// Strides are stored inline (like `TensorShape`) to keep descriptors allocation-free.
public struct TensorStrides: Hashable, Sendable {
    public static let maxRank = TensorShape.maxRank

    public var rank: Int
    public typealias InlineStrides = [8 of Int32]
    public var strides: InlineStrides

    public init(_ values: [Int]) {
        self.rank = min(Self.maxRank, values.count)
        self.strides = InlineStrides(repeating: 0)
        for index in 0..<rank {
            self.strides[index] = Int32(values[index])
        }
    }

    public init(rank: Int, strides: InlineStrides) {
        self.rank = min(Self.maxRank, max(0, rank))
        self.strides = strides
    }

    public var values: [Int] {
        (0..<rank).map { Int(strides[$0]) }
    }

    public static func == (lhs: TensorStrides, rhs: TensorStrides) -> Bool {
        guard lhs.rank == rhs.rank else { return false }
        for index in 0..<TensorStrides.maxRank {
            if lhs.strides[index] != rhs.strides[index] { return false }
        }
        return true
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(rank)
        for index in 0..<TensorStrides.maxRank {
            hasher.combine(strides[index])
        }
    }
}

/// Supported tensor element types.
public enum TensorDataType: String, CaseIterable, Sendable {
    case float32
    case float16
    case bfloat16
    case int8
    case int4
    case int32
    case int64
    case boolean

    /// Byte width for a single element.
    public var byteCount: Int {
        switch self {
        case .float32: return 4
        case .float16, .bfloat16: return 2
        case .int8, .boolean, .int4: return 1
        case .int32: return 4
        case .int64: return 8
        }
    }

    /// True for floating-point types.
    public var isFloatingPoint: Bool {
        switch self {
        case .float16, .float32, .bfloat16:
            return true
        default:
            return false
        }
    }

    /// True for quantized integer formats.
    public var isQuantized: Bool {
        switch self {
        case .int8, .int4:
            return true
        default:
            return false
        }
    }
}

/// Memory ordering for tensor elements.
public enum TensorFormat: Hashable, Sendable {
    case linear
    case nchw
    case nhwc
    case chw2
    case chw32
    case strided(TensorStrides)
}

/// Where the tensor memory lives.
public enum MemoryLocation: Sendable {
    case host
    case device
    case pinnedHost
    case managed
}

/// Metadata describing a named tensor binding.
///
/// `TensorDescriptor` mirrors a TensorRT IO tensor/binding: name, shape, scalar element type, and
/// (optionally) layout/strides and quantization metadata.
public struct TensorDescriptor: Hashable, Sendable {
    public var name: String
    public var shape: TensorShape
    public var dataType: TensorDataType
    public var format: TensorFormat
    public var dynamicAxes: [Int: TensorShape.DynamicAxisRange]
    public var strides: TensorStrides?
    public var quantization: QuantizationParameters?

    /// Creates a descriptor for a TensorRT binding.
    /// - Parameters:
    ///   - name: Unique binding name as defined by the engine.
    ///   - shape: Rank and dimensions. Use non-positive values for dynamic axes.
    ///   - dataType: Scalar type stored in the tensor.
    ///   - format: Optional layout hint. Defaults to `.linear`.
    ///   - dynamicAxes: Optional ranges for dynamic axes by index.
    ///   - strides: Optional explicit element strides for strided layouts.
    ///   - quantization: Optional quantization parameters (scales, zero-points, per-channel axis).
    public init(
        name: String,
        shape: TensorShape,
        dataType: TensorDataType,
        format: TensorFormat = .linear,
        dynamicAxes: [Int: TensorShape.DynamicAxisRange] = [:],
        strides: TensorStrides? = nil,
        quantization: QuantizationParameters? = nil
    ) {
        self.name = name
        self.shape = shape
        self.dataType = dataType
        self.format = format
        self.dynamicAxes = dynamicAxes
        self.strides = strides
        self.quantization = quantization
    }
}

/// Quantization parameters attached to a binding.
public struct QuantizationParameters: Hashable, Sendable {
    public var scale: Float?
    public var zeroPoint: Int?
    public var perChannelScales: [Float]?
    public var axis: Int?
    public var calibration: CalibrationMethod?

    public init(
        scale: Float? = nil,
        zeroPoint: Int? = nil,
        perChannelScales: [Float]? = nil,
        axis: Int? = nil,
        calibration: CalibrationMethod? = nil
    ) {
        self.scale = scale
        self.zeroPoint = zeroPoint
        self.perChannelScales = perChannelScales
        self.axis = axis
        self.calibration = calibration
    }

    /// Calibration algorithms supported for INT8/INT4 flows.
    public enum CalibrationMethod: Hashable, Sendable {
        case entropy
        case minMax
        case percentile(Float)
    }
}

/// Provides batches for quantization calibration.
public protocol CalibrationDataProvider: Sendable {
    func nextBatch() async throws -> InferenceBatch?
}

/// Calibration runtime configuration that mirrors TensorRT calibrator behavior.
public struct CalibrationConfiguration: Sendable {
    public var method: QuantizationParameters.CalibrationMethod
    public var batchSize: Int
    public var maxBatches: Int
    public var cacheURL: URL?

    public init(
        method: QuantizationParameters.CalibrationMethod,
        batchSize: Int = 32,
        maxBatches: Int = 50,
        cacheURL: URL? = nil
    ) {
        self.method = method
        self.batchSize = batchSize
        self.maxBatches = maxBatches
        self.cacheURL = cacheURL
    }
}

/// A materialized binding with location and role information.
public struct TensorBinding: Identifiable, Hashable, Sendable {
    public enum Role: Sendable {
        case input
        case output
    }

    public var id: String { name }
    public var name: String { descriptor.name }
    public var descriptor: TensorDescriptor
    public var location: MemoryLocation
    public var role: Role

    public init(descriptor: TensorDescriptor, location: MemoryLocation = .device, role: Role) {
        self.descriptor = descriptor
        self.location = location
        self.role = role
    }
}

/// Value container passed to and from execution contexts.
///
/// `TensorValue` couples a `TensorDescriptor` (what this tensor *is*) with storage (where the bytes live).
///
/// - Note: Not all storage variants are wired to TensorRT yet. Today, end-to-end execution supports
///   host-backed payloads (`.host(Data)` and `.deferred`).
public struct TensorValue: Sendable {
    public enum Storage: Sendable {
        case host(Data)
        case deviceBuffer(address: UInt64, length: Int)
        case deferred(@Sendable () -> Data)
        /// Identifier for a buffer pulled from a pre-allocated pool (host/device pinned).
        case pooledBuffer(id: String, length: Int, location: MemoryLocation)
    }

    public var descriptor: TensorDescriptor
    public var storage: Storage

    /// Estimated payload size. When storage is deferred, this returns the declared size of the shape.
    public var estimatedByteCount: Int {
        descriptor.shape.elementCount * descriptor.dataType.byteCount
    }

    public init(descriptor: TensorDescriptor, storage: Storage) {
        self.descriptor = descriptor
        self.storage = storage
    }
}

/// Batch of named tensor inputs.
///
/// Input values are keyed by tensor/binding name.
public struct InferenceBatch: Sendable {
    public var inputs: [String: TensorValue]
    public var profileName: String?

    public init(inputs: [String: TensorValue], profileName: String? = nil) {
        self.inputs = inputs
        self.profileName = profileName
    }
}

/// Outputs and timing info returned from a single execution.
public struct InferenceResult: Sendable {
    public var outputs: [String: TensorValue]
    public var duration: Duration?
    public var metadata: [String: String]
    public var profileUsed: String?

    public init(outputs: [String: TensorValue], duration: Duration? = nil, metadata: [String: String] = [:], profileUsed: String? = nil) {
        self.outputs = outputs
        self.duration = duration
        self.metadata = metadata
        self.profileUsed = profileUsed
    }
}

// MARK: - Engine options

/// Precision set requested when building an engine.
public struct Precision: OptionSet, Sendable {
    public let rawValue: UInt8

    public init(rawValue: UInt8) {
        self.rawValue = rawValue
    }

    public static let fp32 = Precision(rawValue: 1 << 0)
    public static let fp16 = Precision(rawValue: 1 << 1)
    public static let int8 = Precision(rawValue: 1 << 2)
    public static let int4 = Precision(rawValue: 1 << 3)
    public static let bf16 = Precision(rawValue: 1 << 4)
}

/// Engine-level toggles, e.g. deterministic kernels or CUDA Graph capture.
public enum EngineFlag: Sendable {
    case deterministic
    case enableCUDAStreams
    case enableDLA
    case safeGPUFallback
    case enableTF32
    case captureCUDAGraphs
    case profilingVerbose
}

/// Device selection for heterogeneous Jetson systems.
public struct DeviceSelection: Sendable {
    public var gpu: Int?
    public var dlaCore: Int?
    public var enableMMAP: Bool
    public var streamPriority: Int?

    public init(gpu: Int? = 0, dlaCore: Int? = nil, enableMMAP: Bool = false, streamPriority: Int? = nil) {
        self.gpu = gpu
        self.dlaCore = dlaCore
        self.enableMMAP = enableMMAP
        self.streamPriority = streamPriority
    }
}

/// Range definitions for dynamic shapes used during engine building.
public struct OptimizationProfile: Hashable, Sendable {
    public var name: String
    public var axes: [String: TensorShape.DynamicAxisRange]
    public var bindingRanges: [String: BindingShapeRange]

    public init(name: String, axes: [String: TensorShape.DynamicAxisRange], bindingRanges: [String: BindingShapeRange] = [:]) {
        self.name = name
        self.axes = axes
        self.bindingRanges = bindingRanges
    }

    /// Binding-level shape ranges used during dynamic shape selection.
    public struct BindingShapeRange: Hashable, Sendable {
        public var min: TensorShape
        public var optimal: TensorShape
        public var max: TensorShape

        public init(min: TensorShape, optimal: TensorShape, max: TensorShape) {
            self.min = min
            self.optimal = optimal
            self.max = max
        }
    }
}

/// Options used when building a TensorRT engine from a network or ONNX file.
public struct EngineBuildOptions: Sendable {
    public var precision: Precision
    public var workspaceSizeBytes: Int?
    public var maxStreams: Int
    public var tacticSources: [String]
    public var profiles: [OptimizationProfile]
    public var flags: [EngineFlag]
    public var device: DeviceSelection
    public var calibrator: CalibrationDataProvider?
    public var calibration: CalibrationConfiguration?
    public var perBindingQuantization: [String: QuantizationParameters]
    public var allowRefit: Bool
    public var captureCUDAStreams: Bool
    public var enableTF32: Bool

    public init(
        precision: Precision = [.fp16, .fp32],
        workspaceSizeBytes: Int? = nil,
        maxStreams: Int = 1,
        tacticSources: [String] = [],
        profiles: [OptimizationProfile] = [],
        flags: [EngineFlag] = [.deterministic],
        device: DeviceSelection = DeviceSelection(),
        calibrator: CalibrationDataProvider? = nil,
        calibration: CalibrationConfiguration? = nil,
        perBindingQuantization: [String: QuantizationParameters] = [:],
        allowRefit: Bool = false,
        captureCUDAStreams: Bool = true,
        enableTF32: Bool = false
    ) {
        self.precision = precision
        self.workspaceSizeBytes = workspaceSizeBytes
        self.maxStreams = maxStreams
        self.tacticSources = tacticSources
        self.profiles = profiles
        self.flags = flags
        self.device = device
        self.calibrator = calibrator
        self.calibration = calibration
        self.perBindingQuantization = perBindingQuantization
        self.allowRefit = allowRefit
        self.captureCUDAStreams = captureCUDAStreams
        self.enableTF32 = enableTF32
    }
}

/// Options for deserializing an existing TensorRT engine.
public struct EngineLoadConfiguration: Sendable {
    public var profile: String?
    public var profileIndex: Int?
    public var device: DeviceSelection
    public var logger: Logger
    public var allowRefit: Bool

    public init(profile: String? = nil, profileIndex: Int? = nil, device: DeviceSelection = DeviceSelection(), logger: Logger = .standard, allowRefit: Bool = false) {
        self.profile = profile
        self.profileIndex = profileIndex
        self.device = device
        self.logger = logger
        self.allowRefit = allowRefit
    }
}

/// Summary of an engine's capabilities and IO surface.
public struct EngineDescription: Sendable {
    public var inputs: [TensorBinding]
    public var outputs: [TensorBinding]
    public var precision: Precision
    public var workspaceSizeBytes: Int?
    public var device: DeviceSelection
    public var profiles: [OptimizationProfile]
    public var metadata: [String: String]
    public var profileNames: [String]
    public var planSizeBytes: Int?
    public var computeCapability: String?
    public var tacticSources: [String]
    public var supportsRefit: Bool
    public var supportsCUDAStreamCapture: Bool
    public var supportsTF32: Bool

    public init(
        inputs: [TensorBinding],
        outputs: [TensorBinding],
        precision: Precision,
        workspaceSizeBytes: Int? = nil,
        device: DeviceSelection = DeviceSelection(),
        profiles: [OptimizationProfile] = [],
        metadata: [String: String] = [:],
        profileNames: [String] = [],
        planSizeBytes: Int? = nil,
        computeCapability: String? = nil,
        tacticSources: [String] = [],
        supportsRefit: Bool = false,
        supportsCUDAStreamCapture: Bool = true,
        supportsTF32: Bool = false
    ) {
        self.inputs = inputs
        self.outputs = outputs
        self.precision = precision
        self.workspaceSizeBytes = workspaceSizeBytes
        self.device = device
        self.profiles = profiles
        self.metadata = metadata
        self.profileNames = profileNames
        self.planSizeBytes = planSizeBytes
        self.computeCapability = computeCapability
        self.tacticSources = tacticSources
        self.supportsRefit = supportsRefit
        self.supportsCUDAStreamCapture = supportsCUDAStreamCapture
        self.supportsTF32 = supportsTF32
    }
}

// MARK: - Logging

/// Lightweight logger that mirrors TensorRT's verbosity settings.
public struct Logger: Sendable {
    public struct LogEntry: Sendable {
        public var severity: Severity
        public var message: String
        public var category: String
        public var timestamp: Date

        public init(severity: Severity, message: String, category: String = "TensorRT", timestamp: Date = Date()) {
            self.severity = severity
            self.message = message
            self.category = category
            self.timestamp = timestamp
        }
    }

    public enum Severity: String, CaseIterable, Sendable {
        case trace
        case info
        case warning
        case error
        case fatal
    }

    public enum Destination: Sendable {
        case standardOutput
        case standardError
        case custom(@Sendable (LogEntry) -> Void)
    }

    public var minimumSeverity: Severity
    public var destination: Destination

    public init(minimumSeverity: Severity = .info, destination: Destination = .standardError) {
        self.minimumSeverity = minimumSeverity
        self.destination = destination
    }

    public static let silent = Logger(minimumSeverity: .fatal)
    public static let standard = Logger()
}

// MARK: - Runtime and engine

/// Entry point for loading and building TensorRT engines.
///
/// Use `TensorRTRuntime` to:
/// - Deserialize serialized TensorRT plans (`.deserializeEngine(from:)`)
/// - Build engines from higher-level network definitions (not yet implemented)
public struct TensorRTRuntime: Sendable {
    public var logger: Logger
    private var native: TensorRTNativeInterface

    public init(logger: Logger = .standard, native: TensorRTNativeInterface = DefaultTensorRTNativeInterface()) {
        self.logger = logger
        self.native = native
    }

    /// Loads an engine from serialized bytes.
    /// - Parameters:
    ///   - data: The serialized engine buffer.
    ///   - configuration: Device and profile hints.
    /// - Returns: A ready-to-use engine description.
    public func deserializeEngine(
        from data: Data,
        configuration: EngineLoadConfiguration = EngineLoadConfiguration()
    ) throws -> Engine {
        try native.deserializeEngine(from: data, configuration: configuration)
    }

    /// Builds an engine directly from a high-level network definition.
    /// - Parameters:
    ///   - network: Graph definition created with ``NetworkBuilder``.
    ///   - options: Precision, workspace, and device options.
    public func buildEngine(
        from network: NetworkDefinition,
        options: EngineBuildOptions = EngineBuildOptions()
    ) throws -> Engine {
        try native.buildEngine(from: network, options: options)
    }

    /// Builds an engine from an ONNX model file. The ONNX parser integration is deferred to the
    /// underlying C bindings.
    /// - Parameters:
    ///   - onnxURL: Location of the model on disk.
    ///   - options: Precision, workspace, and optimization profiles.
    public func buildEngine(
        onnxURL: URL,
        options: EngineBuildOptions = EngineBuildOptions()
    ) throws -> Engine {
        try native.buildEngine(onnxURL: onnxURL, options: options)
    }
}

/// Represents a compiled TensorRT engine.
///
/// `Engine` is a lightweight value wrapper around the engine's IO surface (`EngineDescription`) and,
/// optionally, a serialized plan (`serialized`) and/or an opaque native handle (`nativeHandle`).
///
/// Today, `ExecutionContext` executes using `serialized` plan bytes (host buffers) via the native shim.
public struct Engine: Sendable {
    public var description: EngineDescription
    public var serialized: Data?
    public var nativeHandle: EngineHandle?

    public init(description: EngineDescription, serialized: Data? = nil, nativeHandle: EngineHandle? = nil) {
        self.description = description
        self.serialized = serialized
        self.nativeHandle = nativeHandle
    }

    /// Creates an execution context for the engine.
    /// - Parameters:
    ///   - queue: Queue selection for stream capture or reuse.
    ///   - allocator: Memory strategy for host/device buffers.
    public func makeExecutionContext(
        queue: ExecutionQueue = .automatic,
        allocator: MemoryAllocator = .default
    ) throws -> ExecutionContext {
        ExecutionContext(engine: self, queue: queue, allocator: allocator)
    }
}

/// Opaque handle to a native TensorRT engine.
public struct EngineHandle: Sendable, Hashable {
    public var rawValue: UInt64

    public init(rawValue: UInt64) {
        self.rawValue = rawValue
    }
}

/// Queue selection for execution. Maps onto CUDA streams or CUDA Graph captures.
public enum ExecutionQueue: Sendable {
    case automatic
    case external(streamIdentifier: UInt64)
    case capturedGraph(streamIdentifier: UInt64)
}

/// Memory allocation strategy for bindings.
public struct MemoryAllocator: Sendable {
    public enum Strategy: Sendable {
        case pageableHost
        case pinnedHost
        case device
    }

    public var strategy: Strategy
    public var alignment: Int
    public var poolIdentifier: String?
    public var maxSizeBytes: Int?
    public var reclaimAfterUse: Bool
    public var poolGrowthFactor: Double

    public init(
        strategy: Strategy = .pinnedHost,
        alignment: Int = 256,
        poolIdentifier: String? = nil,
        maxSizeBytes: Int? = nil,
        reclaimAfterUse: Bool = true,
        poolGrowthFactor: Double = 1.5
    ) {
        self.strategy = strategy
        self.alignment = alignment
        self.poolIdentifier = poolIdentifier
        self.maxSizeBytes = maxSizeBytes
        self.reclaimAfterUse = reclaimAfterUse
        self.poolGrowthFactor = poolGrowthFactor
    }

    public static let `default` = MemoryAllocator()
}

/// Stateful executor that owns resources for repeated inference calls.
///
/// This protocol exists to allow swapping in a mock executor in tests.
public protocol ExecutionContexting: Sendable {
    func enqueue(_ batch: InferenceBatch, synchronously: Bool) async throws -> InferenceResult
    func setOptimizationProfile(_ profile: OptimizationProfile) async throws
    func setOptimizationProfile(named name: String) async throws
    func reshape(bindings: [String: TensorShape]) async throws
    func warmup(iterations: Int) async throws -> WarmupSummary
}

public actor ExecutionContext: ExecutionContexting {
    public let engine: Engine
    public let queue: ExecutionQueue
    public let allocator: MemoryAllocator
    private var nativeContextHandle: NativeContextHandle?
    private var inputShapes: [String: TensorShape] = [:]

    public init(engine: Engine, queue: ExecutionQueue, allocator: MemoryAllocator = .default) {
        self.engine = engine
        self.queue = queue
        self.allocator = allocator
    }

    private final class NativeContextHandle: @unchecked Sendable {
#if canImport(TensorRTNative)
        let raw: UInt

        init(raw: UInt) {
            self.raw = raw
        }

        deinit {
            trt_context_destroy(raw)
        }
#endif
    }

#if canImport(TensorRTNative)
    private func getOrCreateNativeContext(plan: Data) throws -> UInt {
        if let existing = nativeContextHandle?.raw {
            return existing
        }

        let handle: UInt = plan.withUnsafeBytes { bytes in
            switch queue {
            case .automatic:
                return trt_context_create(bytes.baseAddress, bytes.count)
            case .external(let streamIdentifier):
                return trt_context_create_with_stream(bytes.baseAddress, bytes.count, streamIdentifier, 0)
            case .capturedGraph(let streamIdentifier):
                // TODO: expose CUDA graph capture semantics; for now this uses the provided stream.
                return trt_context_create_with_stream(bytes.baseAddress, bytes.count, streamIdentifier, 0)
            }
        }
        guard handle != 0 else {
            throw TensorRTError.runtimeUnavailable("Failed to create persistent TensorRT execution context.")
        }
        nativeContextHandle = NativeContextHandle(raw: handle)
        return handle
    }
#endif

#if canImport(TensorRTNative)
    private func applyInputShapesIfNeeded(ctx: UInt) throws {
        guard !inputShapes.isEmpty else { return }
        for (name, shape) in inputShapes {
            var dims = [Int32](repeating: 0, count: TensorShape.maxRank)
            for i in 0..<shape.rank {
                dims[i] = shape.dims[i]
            }
            let status = trt_context_set_input_shape(ctx, name, dims, Int32(shape.rank))
            guard status == 0 else {
                throw TensorRTError.runtimeUnavailable("Failed to set input shape for \(name) (status \(status)).")
            }
        }
    }

    private func resolvedTensorShape(ctx: UInt, name: String) throws -> TensorShape {
        var nbDims: Int32 = 0
        var dims = [Int32](repeating: 0, count: TensorShape.maxRank)
        let status = trt_context_get_tensor_shape(ctx, name, &dims, Int32(TensorShape.maxRank), &nbDims)
        guard status == 0 else {
            throw TensorRTError.runtimeUnavailable("Failed to get tensor shape for \(name) (status \(status)).")
        }
        let count = min(TensorShape.maxRank, max(0, Int(nbDims)))
        var inline = TensorShape.InlineDims(repeating: 0)
        for i in 0..<count {
            inline[i] = dims[i]
        }
        return TensorShape(rank: count, dims: inline)
    }
#endif

    /// Schedules a batch for execution. When `synchronously` is true, this awaits completion before returning.
    public func enqueue(
        _ batch: InferenceBatch,
        synchronously: Bool = true
    ) async throws -> InferenceResult {
#if canImport(TensorRTNative)
        guard let plan = engine.serialized else {
            throw TensorRTError.invalidBinding("Engine does not contain serialized plan data.")
        }

        let inputBindings = engine.description.inputs
        let outputBindings = engine.description.outputs

        // Keep input Data alive for the duration of the native call.
        let inputPairs: [(name: String, data: Data)] = try inputBindings.map { binding in
            guard let value = batch.inputs[binding.name] else {
                throw TensorRTError.invalidBinding("Missing required input \(binding.name)")
            }
            guard value.descriptor.name == binding.name else {
                throw TensorRTError.invalidBinding("Mismatched input descriptor for \(binding.name)")
            }

            switch value.storage {
            case .host(let data):
                return (binding.name, data)
            case .deferred(let thunk):
                return (binding.name, thunk())
            default:
                throw TensorRTError.notImplemented("Only host-backed TensorValue inputs are supported in enqueue() for now.")
            }
        }

        let ctx = try getOrCreateNativeContext(plan: plan)

        // For dynamic inputs, require a concrete shape (via reshape()) so we can validate sizes and
        // allow TensorRT to resolve output shapes.
        for binding in inputBindings {
            if binding.descriptor.shape.isDynamic, inputShapes[binding.name] == nil {
                throw TensorRTError.invalidShapeRange("Input \(binding.name) has dynamic shape; call reshape(bindings:) first.")
            }
            if let shape = inputShapes[binding.name], !shape.isDynamic {
                let expectedBytes = shape.elementCount * binding.descriptor.dataType.byteCount
                if let pair = inputPairs.first(where: { $0.name == binding.name }), pair.data.count != expectedBytes {
                    throw TensorRTError.runtimeUnavailable("Input \(binding.name) expected \(expectedBytes) bytes for shape \(shape.dimensions), got \(pair.data.count).")
                }
            }
        }

        try applyInputShapesIfNeeded(ctx: ctx)

        // Prepare output buffers.
        let outputPairs: [(name: String, data: Data)] = try outputBindings.map { binding in
            var desc = binding.descriptor
            if desc.shape.isDynamic {
                // Ask TensorRT for the resolved output shape after input shapes have been set.
                let resolved = try resolvedTensorShape(ctx: ctx, name: binding.name)
                guard !resolved.isDynamic else {
                    throw TensorRTError.invalidShapeRange("Output \(binding.name) is still dynamic after reshape; resolved \(resolved.dimensions).")
                }
                desc.shape = resolved
            }
            let size = desc.shape.elementCount * desc.dataType.byteCount
            return (binding.name, Data(count: size))
        }

        let inputCount = Int32(inputPairs.count)
        let outputCount = Int32(outputPairs.count)

        func withCStringPointers<R>(_ strings: [String], _ body: ([UnsafePointer<CChar>]) throws -> R) rethrows -> R {
            var allocations: [UnsafeMutablePointer<CChar>] = []
            allocations.reserveCapacity(strings.count)
            defer { allocations.forEach { $0.deallocate() } }

            let pointers: [UnsafePointer<CChar>] = strings.map { string in
                let utf8 = Array(string.utf8CString)
                let ptr = UnsafeMutablePointer<CChar>.allocate(capacity: utf8.count)
                ptr.initialize(from: utf8, count: utf8.count)
                allocations.append(ptr)
                return UnsafePointer(ptr)
            }

            return try body(pointers)
        }

        func withInputPointers<R>(
            _ pairs: [(name: String, data: Data)],
            _ index: Int,
            _ pointers: inout [UnsafeRawPointer?],
            _ body: ([UnsafeRawPointer?]) throws -> R
        ) rethrows -> R {
            if index == pairs.count {
                return try body(pointers)
            }
            return try pairs[index].data.withUnsafeBytes { raw in
                pointers.append(raw.baseAddress)
                defer { pointers.removeLast() }
                return try withInputPointers(pairs, index + 1, &pointers, body)
            }
        }

        func withOutputPointers<R>(
            _ index: Int,
            _ pairs: inout [(name: String, data: Data)],
            _ pointers: inout [UnsafeMutableRawPointer?],
            _ body: (inout [(name: String, data: Data)], [UnsafeMutableRawPointer?]) throws -> R
        ) rethrows -> R {
            if index == pairs.count {
                return try body(&pairs, pointers)
            }
            var data = pairs[index].data
            let result = try data.withUnsafeMutableBytes { raw in
                pointers.append(raw.baseAddress)
                defer { pointers.removeLast() }
                return try withOutputPointers(index + 1, &pairs, &pointers, body)
            }
            pairs[index].data = data
            return result
        }

        func execute(plan: Data, inputPairs: [(name: String, data: Data)], outputPairs: inout [(name: String, data: Data)]) throws -> Int32 {
            try withCStringPointers(inputPairs.map(\.name) + outputPairs.map(\.name)) { namePtrs in
                var inputPointers: [UnsafeRawPointer?] = []
                inputPointers.reserveCapacity(inputPairs.count)

                return try withInputPointers(inputPairs, 0, &inputPointers) { inPtrs in
                    var outputPointers: [UnsafeMutableRawPointer?] = []
                    outputPointers.reserveCapacity(outputPairs.count)

                    return try withOutputPointers(0, &outputPairs, &outputPointers) { pairs, outPtrs in
                        var inputs: [trt_named_buffer] = []
                        inputs.reserveCapacity(inputPairs.count)
                        for i in 0..<inputPairs.count {
                            inputs.append(trt_named_buffer(name: namePtrs[i], data: inPtrs[i], size: inputPairs[i].data.count))
                        }

                        var outputs: [trt_named_mutable_buffer] = []
                        outputs.reserveCapacity(pairs.count)
                        let outputNameOffset = inputPairs.count
                        for i in 0..<pairs.count {
                            outputs.append(trt_named_mutable_buffer(name: namePtrs[outputNameOffset + i], data: outPtrs[i], size: pairs[i].data.count))
                        }

                        return trt_context_execute_host(ctx, inputs, inputCount, outputs, outputCount)
                    }
                }
            }
        }

        var mutableOutputs = outputPairs
        let status = try execute(plan: plan, inputPairs: inputPairs, outputPairs: &mutableOutputs)
        guard status == 0 else {
            throw TensorRTError.runtimeUnavailable("TensorRT enqueue failed (status \(status)).")
        }

        let outputs: [String: TensorValue] = try outputBindings.reduce(into: [:]) { dict, binding in
            guard let item = mutableOutputs.first(where: { $0.name == binding.name }) else { return }
            var desc = binding.descriptor
            if desc.shape.isDynamic {
                desc.shape = try resolvedTensorShape(ctx: ctx, name: binding.name)
            }
            dict[binding.name] = TensorValue(descriptor: desc, storage: .host(item.data))
        }

        return InferenceResult(outputs: outputs, duration: nil, metadata: [:], profileUsed: batch.profileName)
#else
        throw TensorRTError.notImplemented("Inference enqueue requires TensorRTNative on Linux")
#endif
    }

    /// Enqueues a single Float32 input and writes a single Float32 output into a caller-provided buffer.
    ///
    /// This convenience API is useful for simple single-input/single-output engines and avoids wrapping
    /// payloads in `TensorValue` / `Data` when you already have `[Float]` buffers.
    public func enqueueF32(
        inputName: String,
        input: [Float],
        outputName: String,
        output: inout [Float],
        synchronously: Bool = true
    ) async throws {
#if canImport(TensorRTNative)
        guard let plan = engine.serialized else {
            throw TensorRTError.invalidBinding("Engine does not contain serialized plan data.")
        }

        guard let inputBinding = engine.description.inputs.first(where: { $0.name == inputName }) else {
            throw TensorRTError.invalidBinding("No input binding named \(inputName).")
        }
        guard let outputBinding = engine.description.outputs.first(where: { $0.name == outputName }) else {
            throw TensorRTError.invalidBinding("No output binding named \(outputName).")
        }

        guard inputBinding.descriptor.dataType == .float32 else {
            throw TensorRTError.unsupportedPrecision("enqueueF32 requires Float32 input for \(inputName).")
        }
        guard outputBinding.descriptor.dataType == .float32 else {
            throw TensorRTError.unsupportedPrecision("enqueueF32 requires Float32 output for \(outputName).")
        }

        let expectedInputElements = inputBinding.descriptor.shape.elementCount
        guard expectedInputElements > 0 else {
            throw TensorRTError.invalidShapeRange("Input \(inputName) has dynamic shape; reshape() is not implemented yet.")
        }
        guard input.count == expectedInputElements else {
            throw TensorRTError.shapeMismatch(
                expected: inputBinding.descriptor.shape,
                received: TensorShape([input.count])
            )
        }

        guard !outputBinding.descriptor.shape.isDynamic else {
            throw TensorRTError.invalidShapeRange("Output \(outputName) has dynamic shape; reshape() is not implemented yet.")
        }
        let expectedOutputElements = outputBinding.descriptor.shape.elementCount
        if output.count != expectedOutputElements {
            output = Array(repeating: 0, count: expectedOutputElements)
        }

        func withCString<R>(_ string: String, _ body: (UnsafePointer<CChar>) throws -> R) rethrows -> R {
            let utf8 = Array(string.utf8CString)
            return try utf8.withUnsafeBufferPointer { buffer in
                try body(buffer.baseAddress!)
            }
        }

        try withCString(inputName) { inputNamePtr in
            try withCString(outputName) { outputNamePtr in
                try input.withUnsafeBufferPointer { inBuf in
                    var outArray = output
                    let status: Int32 = try outArray.withUnsafeMutableBufferPointer { outBuf in
                        guard let inBase = inBuf.baseAddress else {
                            throw TensorRTError.invalidBinding("Input span has no base address.")
                        }
                        guard let outBase = outBuf.baseAddress else {
                            throw TensorRTError.invalidBinding("Output buffer has no base address.")
                        }

                        var inputBuffer = trt_named_buffer(
                            name: inputNamePtr,
                            data: UnsafeRawPointer(inBase),
                            size: inBuf.count * MemoryLayout<Float>.stride
                        )
                        var outputBuffer = trt_named_mutable_buffer(
                            name: outputNamePtr,
                            data: UnsafeMutableRawPointer(outBase),
                            size: outBuf.count * MemoryLayout<Float>.stride
                        )

                        let ctx = try getOrCreateNativeContext(plan: plan)
                        return withUnsafePointer(to: &inputBuffer) { inputPtr in
                            withUnsafePointer(to: &outputBuffer) { outputPtr in
                                trt_context_execute_host(ctx, inputPtr, 1, outputPtr, 1)
                            }
                        }
                    }
                    output = outArray
                    guard status == 0 else {
                        throw TensorRTError.runtimeUnavailable("TensorRT enqueue failed (status \(status)).")
                    }
                }
            }
        }
#else
        throw TensorRTError.notImplemented("enqueueF32 requires TensorRTNative on Linux")
#endif
    }

    /// Enqueues a single byte-addressed input and writes a single output into a caller-provided byte buffer.
    ///
    /// This is a low-level helper for engines with non-`Float32` IO (or callers that already have
    /// pre-packed bytes).
    public func enqueueBytes(
        inputName: String,
        input: [UInt8],
        outputName: String,
        output: inout [UInt8],
        synchronously: Bool = true
    ) async throws {
#if canImport(TensorRTNative)
        guard let plan = engine.serialized else {
            throw TensorRTError.invalidBinding("Engine does not contain serialized plan data.")
        }

        guard let inputBinding = engine.description.inputs.first(where: { $0.name == inputName }) else {
            throw TensorRTError.invalidBinding("No input binding named \(inputName).")
        }
        guard let outputBinding = engine.description.outputs.first(where: { $0.name == outputName }) else {
            throw TensorRTError.invalidBinding("No output binding named \(outputName).")
        }

        let expectedInputBytes = inputBinding.descriptor.shape.elementCount * inputBinding.descriptor.dataType.byteCount
        guard expectedInputBytes > 0 else {
            throw TensorRTError.invalidShapeRange("Input \(inputName) has dynamic shape; reshape() is not implemented yet.")
        }
        guard input.count == expectedInputBytes else {
            throw TensorRTError.invalidBinding("Input \(inputName) expected \(expectedInputBytes) bytes, got \(input.count).")
        }

        let expectedOutputBytes = outputBinding.descriptor.shape.elementCount * outputBinding.descriptor.dataType.byteCount
        guard expectedOutputBytes > 0 else {
            throw TensorRTError.invalidShapeRange("Output \(outputName) has dynamic shape; reshape() is not implemented yet.")
        }
        if output.count != expectedOutputBytes {
            output = Array(repeating: 0, count: expectedOutputBytes)
        }

        func withCString<R>(_ string: String, _ body: (UnsafePointer<CChar>) throws -> R) rethrows -> R {
            let utf8 = Array(string.utf8CString)
            return try utf8.withUnsafeBufferPointer { buffer in
                try body(buffer.baseAddress!)
            }
        }

        try withCString(inputName) { inputNamePtr in
            try withCString(outputName) { outputNamePtr in
                try input.withUnsafeBufferPointer { inBuf in
                    var outArray = output
                    let status: Int32 = try outArray.withUnsafeMutableBufferPointer { outBuf in
                        guard let inBase = inBuf.baseAddress else {
                            throw TensorRTError.invalidBinding("Input span has no base address.")
                        }
                        guard let outBase = outBuf.baseAddress else {
                            throw TensorRTError.invalidBinding("Output buffer has no base address.")
                        }

                        var inputBuffer = trt_named_buffer(
                            name: inputNamePtr,
                            data: UnsafeRawPointer(inBase),
                            size: inBuf.count
                        )
                        var outputBuffer = trt_named_mutable_buffer(
                            name: outputNamePtr,
                            data: UnsafeMutableRawPointer(outBase),
                            size: outBuf.count
                        )

                        let ctx = try getOrCreateNativeContext(plan: plan)
                        return withUnsafePointer(to: &inputBuffer) { inputPtr in
                            withUnsafePointer(to: &outputBuffer) { outputPtr in
                                trt_context_execute_host(ctx, inputPtr, 1, outputPtr, 1)
                            }
                        }
                    }
                    output = outArray
                    guard status == 0 else {
                        throw TensorRTError.runtimeUnavailable("TensorRT enqueue failed (status \(status)).")
                    }
                }
            }
        }
#else
        throw TensorRTError.notImplemented("enqueueBytes requires TensorRTNative on Linux")
#endif
    }

    /// Enqueues using device-resident buffers (CUDA device pointers).
    ///
    /// This is the lowest-level execution API and is intended for performance-sensitive pipelines where
    /// inputs/outputs already live on the GPU. The provided `address` values must be valid CUDA device
    /// pointers for the active device.
    public func enqueueDevice(
        inputs: [String: (address: UInt64, length: Int)],
        outputs: [String: (address: UInt64, length: Int)],
        synchronously: Bool = true
    ) async throws {
#if canImport(TensorRTNative)
        guard let plan = engine.serialized else {
            throw TensorRTError.invalidBinding("Engine does not contain serialized plan data.")
        }
        let ctx = try getOrCreateNativeContext(plan: plan)
        try applyInputShapesIfNeeded(ctx: ctx)

        func withCStringPointers<R>(_ strings: [String], _ body: ([UnsafePointer<CChar>]) throws -> R) rethrows -> R {
            var allocations: [UnsafeMutablePointer<CChar>] = []
            allocations.reserveCapacity(strings.count)
            defer { allocations.forEach { $0.deallocate() } }

            let pointers: [UnsafePointer<CChar>] = strings.map { string in
                let utf8 = Array(string.utf8CString)
                let ptr = UnsafeMutablePointer<CChar>.allocate(capacity: utf8.count)
                ptr.initialize(from: utf8, count: utf8.count)
                allocations.append(ptr)
                return UnsafePointer(ptr)
            }
            return try body(pointers)
        }

        let inputNames = Array(inputs.keys)
        let outputNames = Array(outputs.keys)
        try withCStringPointers(inputNames + outputNames) { namePtrs in
            var inputBuffers: [trt_named_buffer] = []
            inputBuffers.reserveCapacity(inputNames.count)
            for (i, name) in inputNames.enumerated() {
                guard let spec = inputs[name] else { continue }
                guard spec.address != 0, spec.length > 0 else {
                    throw TensorRTError.invalidBinding("Invalid device input buffer for \(name).")
                }
                let ptr = UnsafeRawPointer(bitPattern: UInt(spec.address))
                inputBuffers.append(trt_named_buffer(name: namePtrs[i], data: ptr, size: spec.length))
            }

            var outputBuffers: [trt_named_mutable_buffer] = []
            outputBuffers.reserveCapacity(outputNames.count)
            let offset = inputNames.count
            for (i, name) in outputNames.enumerated() {
                guard let spec = outputs[name] else { continue }
                guard spec.address != 0, spec.length > 0 else {
                    throw TensorRTError.invalidBinding("Invalid device output buffer for \(name).")
                }
                let ptr = UnsafeMutableRawPointer(bitPattern: UInt(spec.address))
                outputBuffers.append(trt_named_mutable_buffer(name: namePtrs[offset + i], data: ptr, size: spec.length))
            }

            let status = trt_context_execute_device(
                ctx,
                inputBuffers,
                Int32(inputBuffers.count),
                outputBuffers,
                Int32(outputBuffers.count),
                synchronously ? 1 : 0
            )
            guard status == 0 else {
                throw TensorRTError.runtimeUnavailable("TensorRT device enqueue failed (status \(status)).")
            }
        }
#else
        throw TensorRTError.notImplemented("enqueueDevice requires TensorRTNative on Linux")
#endif
    }

    /// Records a CUDA event on this context's underlying stream.
    ///
    /// Use this to wait for completion without synchronizing the whole stream. This is most useful
    /// with `ExecutionQueue.external` and `enqueueDevice(..., synchronously: false)`.
    public func recordEvent(_ event: TensorRTSystem.CUDAEvent) async throws {
#if canImport(TensorRTNative)
        guard let plan = engine.serialized else {
            throw TensorRTError.invalidBinding("Engine does not contain serialized plan data.")
        }
        let ctx = try getOrCreateNativeContext(plan: plan)
        let status = trt_context_record_event(ctx, event.rawValue)
        guard status == 0 else {
            throw TensorRTError.runtimeUnavailable("Failed to record event on context stream (status \(status)).")
        }
#else
        throw TensorRTError.notImplemented("recordEvent requires TensorRTNative on Linux")
#endif
    }

    /// Selects the active optimization profile for the context.
    public func setOptimizationProfile(_ profile: OptimizationProfile) async throws {
        throw TensorRTError.notImplemented("Optimization profile switching")
    }

    /// Selects the active optimization profile by name.
    public func setOptimizationProfile(named name: String) async throws {
        throw TensorRTError.notImplemented("Optimization profile switching by name")
    }

    /// Reshapes input and output bindings for dynamic shapes.
    public func reshape(bindings: [String: TensorShape]) async throws {
#if canImport(TensorRTNative)
        for (name, shape) in bindings {
            guard engine.description.inputs.contains(where: { $0.name == name }) else {
                throw TensorRTError.invalidBinding("reshape() expects input binding names; \(name) is not an input.")
            }
            guard shape.rank > 0, shape.rank <= TensorShape.maxRank else {
                throw TensorRTError.invalidShapeRange("Invalid rank \(shape.rank) for \(name).")
            }
            guard !shape.isDynamic else {
                throw TensorRTError.invalidShapeRange("reshape() requires concrete (non-dynamic) shapes; got \(shape.dimensions) for \(name).")
            }
        }

        inputShapes = bindings

        // If the native context already exists, apply immediately so output shape queries work.
        if let plan = engine.serialized, let ctx = nativeContextHandle?.raw {
            _ = plan // keep intent explicit; context is bound to the plan already.
            try applyInputShapesIfNeeded(ctx: ctx)
        }
#else
        throw TensorRTError.notImplemented("Runtime reshape requires TensorRTNative on Linux")
#endif
    }

    /// Allocates and pins buffers ahead of time to reduce per-inference overhead.
    public func preparePersistentBuffers(for bindings: [TensorBinding]) async throws -> PersistentBufferPlan {
        PersistentBufferPlan.plan(for: bindings, allocator: allocator)
    }

    /// Runs warm-up passes for timing stabilization. Returns a summary of measured latencies.
    public func warmup(iterations: Int = 1) async throws -> WarmupSummary {
        throw TensorRTError.notImplemented("Warm-up execution")
    }

    /// Releases device resources associated with the context.
    public func tearDown() async {
    }
}

/// Description of buffers allocated once and reused across inference calls.
public struct PersistentBufferPlan: Sendable {
    public var buffers: [String: PersistentBuffer]

    public init(buffers: [String: PersistentBuffer]) {
        self.buffers = buffers
    }

    /// Convenience constructor that derives buffer sizes from descriptors and allocator strategy.
    public static func plan(for bindings: [TensorBinding], allocator: MemoryAllocator) -> PersistentBufferPlan {
        let entries = bindings.reduce(into: [String: PersistentBuffer]()) { dict, binding in
            let size = binding.descriptor.shape.elementCount * binding.descriptor.dataType.byteCount
            let location: MemoryLocation
            switch allocator.strategy {
            case .device:
                location = .device
            case .pinnedHost:
                location = .pinnedHost
            case .pageableHost:
                location = .host
            }
            let id = allocator.poolIdentifier.map { "\($0)-\(binding.name)" } ?? binding.name
            dict[binding.name] = PersistentBuffer(id: id, length: size, location: location)
        }
        return PersistentBufferPlan(buffers: entries)
    }
}

/// A reusable buffer with location metadata.
public struct PersistentBuffer: Sendable {
    public var id: String
    public var length: Int
    public var location: MemoryLocation
    public var alignment: Int?
    public var reuse: Bool

    public init(id: String, length: Int, location: MemoryLocation, alignment: Int? = nil, reuse: Bool = true) {
        self.id = id
        self.length = length
        self.location = location
        self.alignment = alignment
        self.reuse = reuse
    }
}

/// Simple latency summary from warm-up or profiling passes.
public struct WarmupSummary: Sendable {
    public var samples: [Duration]
    public var average: Duration?
    public var minimum: Duration?
    public var maximum: Duration?

    public init(samples: [Duration], average: Duration? = nil, minimum: Duration? = nil, maximum: Duration? = nil) {
        self.samples = samples
        self.average = average
        self.minimum = minimum
        self.maximum = maximum
    }
}

// MARK: - Scheduling

/// Request metadata for scheduling inference with QoS hints.
public struct InferenceRequest: Sendable {
    public enum QoS: Sendable {
        case latencySensitive
        case throughput
    }

    public var batch: InferenceBatch
    public var deadline: Date?
    public var qos: QoS

    public init(batch: InferenceBatch, deadline: Date? = nil, qos: QoS = .latencySensitive) {
        self.batch = batch
        self.deadline = deadline
        self.qos = qos
    }
}

/// A lightweight scheduler that multiplexes requests onto an execution context.
public actor InferenceScheduler {
    private let context: ExecutionContexting

    public init(context: ExecutionContexting) {
        self.context = context
    }

    /// Submits a request to the underlying context. Implementations may reorder or batch requests
    /// depending on QoS and deadlines.
    public func submit(_ request: InferenceRequest) async throws -> InferenceResult {
        throw TensorRTError.notImplemented("Inference scheduling")
    }
}

// MARK: - Testing hooks

/// Simple mock context that routes requests to a user-provided handler, useful for tests without TensorRT present.
public actor MockExecutionContext: ExecutionContexting {
    public var handler: @Sendable (InferenceBatch) async throws -> InferenceResult

    public init(handler: @escaping @Sendable (InferenceBatch) async throws -> InferenceResult) {
        self.handler = handler
    }

    public func enqueue(_ batch: InferenceBatch, synchronously: Bool = true) async throws -> InferenceResult {
        try await handler(batch)
    }

    public func setOptimizationProfile(_ profile: OptimizationProfile) async throws {}

    public func setOptimizationProfile(named name: String) async throws {}

    public func reshape(bindings: [String: TensorShape]) async throws {}

    public func warmup(iterations: Int = 1) async throws -> WarmupSummary {
        WarmupSummary(samples: [])
    }
}

// MARK: - Native interface scaffolding

/// Abstracts the C++ TensorRT interaction so the Swift surface remains testable and portable.
public protocol TensorRTNativeInterface: Sendable {
    func deserializeEngine(from data: Data, configuration: EngineLoadConfiguration) throws -> Engine
    func buildEngine(from network: NetworkDefinition, options: EngineBuildOptions) throws -> Engine
    func buildEngine(onnxURL: URL, options: EngineBuildOptions) throws -> Engine
}

/// Default implementation that should be backed by Swift 6.2 C++ interop and Span-friendly buffer views.
public struct DefaultTensorRTNativeInterface: TensorRTNativeInterface {
    public init() {}

    public func deserializeEngine(from data: Data, configuration: EngineLoadConfiguration) throws -> Engine {
#if canImport(TensorRTNative)
        let handle: UInt = data.withUnsafeBytes { bytes in
            trt_deserialize_engine(bytes.baseAddress, bytes.count)
        }

        guard handle != 0 else {
            throw TensorRTError.runtimeUnavailable("Failed to deserialize engine via libnvinfer.")
        }
        defer { trt_destroy_engine(handle) }

        var ioCount: Int32 = 0
        guard trt_engine_get_io_count(handle, &ioCount) == 0, ioCount > 0 else {
            throw TensorRTError.runtimeUnavailable("Engine IO introspection failed.")
        }

        var inputs: [TensorBinding] = []
        var outputs: [TensorBinding] = []
        inputs.reserveCapacity(Int(ioCount))
        outputs.reserveCapacity(Int(ioCount))

        func mapDataType(_ raw: Int32) throws -> TensorDataType {
            // nvinfer1::DataType values (stable across TensorRT versions):
            // kFLOAT=0, kHALF=1, kINT8=2, kINT32=3, kBOOL=4, kUINT8=5, kFP8=6, kBF16=7, kINT64=8, kINT4=9.
            switch raw {
            case 0: return .float32
            case 1: return .float16
            case 2: return .int8
            case 3: return .int32
            case 4: return .boolean
            case 7: return .bfloat16
            case 8: return .int64
            case 9: return .int4
            default:
                throw TensorRTError.unsupportedPrecision("Unsupported TensorRT DataType raw value \(raw)")
            }
        }

        for index in 0..<ioCount {
            var desc = trt_io_tensor_desc()
            guard trt_engine_get_io_desc(handle, index, &desc) == 0 else {
                throw TensorRTError.runtimeUnavailable("Engine IO desc lookup failed at index \(index).")
            }

            let name = withUnsafePointer(to: &desc.name) { ptr in
                ptr.withMemoryRebound(to: CChar.self, capacity: Int(TRT_MAX_NAME)) { cStr in
                    String(cString: cStr)
                }
            }

            let nbDims = Int(desc.nbDims)
            let shape: TensorShape = withUnsafeBytes(of: &desc.dims) { raw in
                let buffer = raw.bindMemory(to: Int32.self)
                var inline = TensorShape.InlineDims(repeating: 0)
                let count = min(TensorShape.maxRank, max(0, nbDims))
                for index in 0..<count {
                    inline[index] = buffer[index]
                }
                return TensorShape(rank: count, dims: inline)
            }
            let dtype = try mapDataType(desc.dataType)

            let descriptor = TensorDescriptor(name: name, shape: shape, dataType: dtype, format: .linear)
            let role: TensorBinding.Role = (desc.isInput == 1) ? .input : .output
            let binding = TensorBinding(descriptor: descriptor, location: .device, role: role)
            switch role {
            case .input: inputs.append(binding)
            case .output: outputs.append(binding)
            }
        }

        let description = EngineDescription(
            inputs: inputs,
            outputs: outputs,
            precision: [.fp32],
            workspaceSizeBytes: nil,
            device: configuration.device,
            profiles: [],
            metadata: [:],
            profileNames: [],
            planSizeBytes: data.count,
            computeCapability: nil,
            tacticSources: [],
            supportsRefit: false,
            supportsCUDAStreamCapture: true,
            supportsTF32: false
        )

        return Engine(description: description, serialized: data, nativeHandle: nil)
#else
        throw TensorRTError.notImplemented("Native engine deserialization (TensorRTNative module unavailable)")
#endif
    }

    public func buildEngine(from network: NetworkDefinition, options: EngineBuildOptions) throws -> Engine {
        throw TensorRTError.notImplemented("Native network-driven build")
    }

    public func buildEngine(onnxURL: URL, options: EngineBuildOptions) throws -> Engine {
        throw TensorRTError.notImplemented("Native ONNX build")
    }
}

// MARK: - Network definition

/// Graph-level network description that can be built into an engine.
public struct NetworkDefinition: Sendable {
    public var name: String
    public var inputs: [TensorBinding]
    public var layers: [Layer]
    public var outputs: [TensorBinding]
    public var metadata: [String: String]

    public init(name: String, inputs: [TensorBinding], layers: [Layer], outputs: [TensorBinding], metadata: [String: String] = [:]) {
        self.name = name
        self.inputs = inputs
        self.layers = layers
        self.outputs = outputs
        self.metadata = metadata
    }
}

/// Helper for constructing a `NetworkDefinition` incrementally.
public struct NetworkBuilder: Sendable {
    public private(set) var definition: NetworkDefinition
    public private(set) var validationIssues: [NetworkValidationIssue] = []

    public init(name: String = "Network") {
        self.definition = NetworkDefinition(name: name, inputs: [], layers: [], outputs: [])
    }

    /// Attaches metadata to the network (e.g., authorship, versioning, provenance).
    public mutating func setMetadata(_ metadata: [String: String]) {
        definition.metadata = metadata
    }

    /// Adds an input binding with a descriptor.
    @discardableResult
    public mutating func addInput(
        name: String,
        shape: TensorShape,
        dataType: TensorDataType,
        format: TensorFormat = .linear
    ) -> TensorBinding {
        let descriptor = TensorDescriptor(name: name, shape: shape, dataType: dataType, format: format)
        if definition.inputs.contains(where: { $0.name == name }) {
            validationIssues.append(NetworkValidationIssue(kind: .duplicateBinding(name), message: "Duplicate input binding \(name)"))
        }
        let binding = TensorBinding(descriptor: descriptor, role: .input)
        definition.inputs.append(binding)
        return binding
    }

    /// Adds a convolution layer definition.
    @discardableResult
    public mutating func addConvolution(
        name: String,
        input: TensorBinding,
        parameters: Layer.Convolution
    ) throws -> TensorBinding {
        let output = TensorBinding(descriptor: TensorDescriptor(name: name, shape: input.descriptor.shape, dataType: input.descriptor.dataType), role: .output)
        let layer = Layer(name: name, kind: .convolution(parameters, input: input, output: output))
        definition.layers.append(layer)
        return output
    }

    /// Adds an activation layer definition.
    @discardableResult
    public mutating func addActivation(
        name: String,
        input: TensorBinding,
        activation: Layer.Activation
    ) throws -> TensorBinding {
        let output = TensorBinding(descriptor: TensorDescriptor(name: name, shape: input.descriptor.shape, dataType: input.descriptor.dataType), role: .output)
        let layer = Layer(name: name, kind: .activation(activation, input: input, output: output))
        definition.layers.append(layer)
        return output
    }

    /// Adds a pooling layer definition.
    @discardableResult
    public mutating func addPooling(
        name: String,
        input: TensorBinding,
        parameters: Layer.Pooling
    ) throws -> TensorBinding {
        let output = TensorBinding(descriptor: TensorDescriptor(name: name, shape: input.descriptor.shape, dataType: input.descriptor.dataType), role: .output)
        let layer = Layer(name: name, kind: .pooling(parameters, input: input, output: output))
        definition.layers.append(layer)
        return output
    }

    /// Adds a normalization layer definition.
    @discardableResult
    public mutating func addNormalization(
        name: String,
        input: TensorBinding,
        parameters: Layer.Normalization
    ) throws -> TensorBinding {
        let output = TensorBinding(descriptor: TensorDescriptor(name: name, shape: input.descriptor.shape, dataType: input.descriptor.dataType), role: .output)
        let layer = Layer(name: name, kind: .normalization(parameters, input: input, output: output))
        definition.layers.append(layer)
        return output
    }

    /// Adds a reshape layer definition.
    @discardableResult
    public mutating func addReshape(
        name: String,
        input: TensorBinding,
        targetShape: TensorShape
    ) throws -> TensorBinding {
        let descriptor = TensorDescriptor(name: name, shape: targetShape, dataType: input.descriptor.dataType, format: input.descriptor.format)
        let output = TensorBinding(descriptor: descriptor, role: .output)
        let layer = Layer(name: name, kind: .reshape(Layer.Reshape(targetShape: targetShape), input: input, output: output))
        definition.layers.append(layer)
        return output
    }

    /// Adds a transpose layer definition.
    @discardableResult
    public mutating func addTranspose(
        name: String,
        input: TensorBinding,
        permutation: [Int]
    ) throws -> TensorBinding {
        let output = TensorBinding(descriptor: TensorDescriptor(name: name, shape: input.descriptor.shape, dataType: input.descriptor.dataType), role: .output)
        let layer = Layer(name: name, kind: .transpose(Layer.Transpose(permutation: permutation), input: input, output: output))
        definition.layers.append(layer)
        return output
    }

    /// Adds a concat layer definition.
    @discardableResult
    public mutating func addConcat(
        name: String,
        inputs: [TensorBinding],
        axis: Int
    ) throws -> TensorBinding {
        guard let first = inputs.first else {
            throw TensorRTError.invalidBinding("Concat requires at least one input")
        }
        let output = TensorBinding(descriptor: TensorDescriptor(name: name, shape: first.descriptor.shape, dataType: first.descriptor.dataType), role: .output)
        let layer = Layer(name: name, kind: .concat(Layer.Concat(axis: axis), inputs: inputs, output: output))
        definition.layers.append(layer)
        return output
    }

    /// Adds a slice layer definition.
    @discardableResult
    public mutating func addSlice(
        name: String,
        input: TensorBinding,
        start: [Int],
        size: [Int],
        stride: [Int]
    ) throws -> TensorBinding {
        let descriptor = TensorDescriptor(name: name, shape: input.descriptor.shape, dataType: input.descriptor.dataType, format: input.descriptor.format)
        let output = TensorBinding(descriptor: descriptor, role: .output)
        let slice = Layer.Slice(start: start, size: size, stride: stride)
        let layer = Layer(name: name, kind: .slice(slice, input: input, output: output))
        definition.layers.append(layer)
        return output
    }

    /// Adds a softmax layer definition.
    @discardableResult
    public mutating func addSoftmax(
        name: String,
        input: TensorBinding,
        axis: Int
    ) throws -> TensorBinding {
        let output = TensorBinding(descriptor: TensorDescriptor(name: name, shape: input.descriptor.shape, dataType: input.descriptor.dataType), role: .output)
        let layer = Layer(name: name, kind: .softmax(Layer.Softmax(axis: axis), input: input, output: output))
        definition.layers.append(layer)
        return output
    }

    /// Adds a self-attention style layer definition.
    @discardableResult
    public mutating func addAttention(
        name: String,
        inputs: [TensorBinding],
        parameters: Layer.Attention
    ) throws -> TensorBinding {
        guard let first = inputs.first else {
            throw TensorRTError.invalidBinding("Attention requires at least one input")
        }
        let output = TensorBinding(descriptor: TensorDescriptor(name: name, shape: first.descriptor.shape, dataType: first.descriptor.dataType), role: .output)
        let layer = Layer(name: name, kind: .attention(parameters, inputs: inputs, output: output))
        definition.layers.append(layer)
        return output
    }

    /// Adds a recurrent layer definition (e.g., LSTM/GRU).
    @discardableResult
    public mutating func addRecurrent(
        name: String,
        inputs: [TensorBinding],
        parameters: Layer.Recurrent
    ) throws -> [TensorBinding] {
        guard let first = inputs.first else {
            throw TensorRTError.invalidBinding("Recurrent layer requires at least one input")
        }
        let output = TensorBinding(descriptor: TensorDescriptor(name: name, shape: first.descriptor.shape, dataType: first.descriptor.dataType), role: .output)
        let hiddenState = TensorBinding(descriptor: TensorDescriptor(name: name + "_state", shape: first.descriptor.shape, dataType: first.descriptor.dataType), role: .output)
        let layer = Layer(name: name, kind: .recurrent(parameters, inputs: inputs, outputs: [output, hiddenState]))
        definition.layers.append(layer)
        return [output, hiddenState]
    }

    /// Marks a binding as an engine output.
    public mutating func markOutput(_ binding: TensorBinding) {
        if !definition.outputs.contains(binding) {
            definition.outputs.append(binding)
        }
    }

    /// Finalizes the builder into a definition.
    public func build() -> NetworkDefinition {
        definition
    }
}

/// Validation issue detected during network construction.
public struct NetworkValidationIssue: Sendable, Hashable {
    public enum Kind: Sendable, Hashable {
        case missingOutputs
        case duplicateBinding(String)
        case shapeMismatch(String)
        case unsupportedLayer(String)
    }

    public var kind: Kind
    public var message: String

    public init(kind: Kind, message: String) {
        self.kind = kind
        self.message = message
    }
}

/// Layer metadata used to build TensorRT graphs.
public struct Layer: Hashable, Sendable {
    public var name: String
    public var kind: Kind

    public init(name: String, kind: Kind) {
        self.name = name
        self.kind = kind
    }

    public enum Kind: Hashable, Sendable {
        case convolution(Convolution, input: TensorBinding, output: TensorBinding)
        case activation(Activation, input: TensorBinding, output: TensorBinding)
        case elementWise(ElementWise, inputs: [TensorBinding], output: TensorBinding)
        case matmul(Matmul, left: TensorBinding, right: TensorBinding, output: TensorBinding)
        case pooling(Pooling, input: TensorBinding, output: TensorBinding)
        case normalization(Normalization, input: TensorBinding, output: TensorBinding)
        case reshape(Reshape, input: TensorBinding, output: TensorBinding)
        case transpose(Transpose, input: TensorBinding, output: TensorBinding)
        case concat(Concat, inputs: [TensorBinding], output: TensorBinding)
        case slice(Slice, input: TensorBinding, output: TensorBinding)
        case softmax(Softmax, input: TensorBinding, output: TensorBinding)
        case attention(Attention, inputs: [TensorBinding], output: TensorBinding)
        case recurrent(Recurrent, inputs: [TensorBinding], outputs: [TensorBinding])
        case plugin(identifier: String, version: String, inputs: [TensorBinding], outputs: [TensorBinding], metadata: [String: String])
    }

    /// Convolution parameters mirroring TensorRT builder options.
    public struct Convolution: Hashable, Sendable {
        public var outputChannels: Int
        public var kernelShape: TensorShape
        public var stride: TensorShape
        public var padding: TensorShape
        public var dilation: TensorShape
        public var groups: Int
        public var bias: Bool

        public init(
            outputChannels: Int,
            kernelShape: TensorShape,
            stride: TensorShape = TensorShape([1, 1]),
            padding: TensorShape = TensorShape([0, 0]),
            dilation: TensorShape = TensorShape([1, 1]),
            groups: Int = 1,
            bias: Bool = true
        ) {
            self.outputChannels = outputChannels
            self.kernelShape = kernelShape
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.bias = bias
        }
    }

    /// Activation operator variants.
    public enum Activation: Hashable, Sendable {
        case relu
        case leakyReLU(alpha: Float)
        case sigmoid
        case tanh
        case swish
        case gelu
        case identity
        case custom(name: String, parameters: [String: String] = [:])
    }

    /// Element-wise operation metadata.
    public enum ElementWise: Hashable, Sendable {
        case add
        case subtract
        case multiply
        case divide
        case max
        case min
        case power(Float)
        case custom(name: String, parameters: [String: String] = [:])
    }

    /// Matrix multiplication metadata.
    public struct Matmul: Hashable, Sendable {
        public var transposedLeft: Bool
        public var transposedRight: Bool
        public var bias: Bool

        public init(transposedLeft: Bool = false, transposedRight: Bool = false, bias: Bool = false) {
            self.transposedLeft = transposedLeft
            self.transposedRight = transposedRight
            self.bias = bias
        }
    }

    /// Pooling metadata.
    public struct Pooling: Hashable, Sendable {
        public enum Mode: Hashable, Sendable {
            case max
            case average
        }

        public var mode: Mode
        public var window: TensorShape
        public var stride: TensorShape
        public var padding: TensorShape

        public init(mode: Mode, window: TensorShape, stride: TensorShape = TensorShape([1, 1]), padding: TensorShape = TensorShape([0, 0])) {
            self.mode = mode
            self.window = window
            self.stride = stride
            self.padding = padding
        }
    }

    /// Normalization metadata.
    public struct Normalization: Hashable, Sendable {
        public enum Kind: Hashable, Sendable {
            case batch(epsilon: Float)
            case layer(epsilon: Float)
            case instance(epsilon: Float)
        }

        public var kind: Kind

        public init(kind: Kind) {
            self.kind = kind
        }
    }

    /// Reshape metadata.
    public struct Reshape: Hashable, Sendable {
        public var targetShape: TensorShape

        public init(targetShape: TensorShape) {
            self.targetShape = targetShape
        }
    }

    /// Transpose metadata.
    public struct Transpose: Hashable, Sendable {
        public var permutation: [Int]

        public init(permutation: [Int]) {
            self.permutation = permutation
        }
    }

    /// Concatenation metadata.
    public struct Concat: Hashable, Sendable {
        public var axis: Int

        public init(axis: Int) {
            self.axis = axis
        }
    }

    /// Slice metadata.
    public struct Slice: Hashable, Sendable {
        public var start: [Int]
        public var size: [Int]
        public var stride: [Int]

        public init(start: [Int], size: [Int], stride: [Int]) {
            self.start = start
            self.size = size
            self.stride = stride
        }
    }

    /// Softmax metadata.
    public struct Softmax: Hashable, Sendable {
        public var axis: Int

        public init(axis: Int) {
            self.axis = axis
        }
    }

    /// Attention metadata.
    public struct Attention: Hashable, Sendable {
        public var numHeads: Int
        public var headDimension: Int
        public var dropout: Float

        public init(numHeads: Int, headDimension: Int, dropout: Float = 0) {
            self.numHeads = numHeads
            self.headDimension = headDimension
            self.dropout = dropout
        }
    }

    /// Simple recurrent network metadata.
    public struct Recurrent: Hashable, Sendable {
        public enum Kind: Hashable, Sendable {
            case lstm
            case gru
            case rnn
        }

        public var kind: Kind
        public var hiddenSize: Int
        public var numLayers: Int
        public var bidirectional: Bool

        public init(kind: Kind, hiddenSize: Int, numLayers: Int = 1, bidirectional: Bool = false) {
            self.kind = kind
            self.hiddenSize = hiddenSize
            self.numLayers = numLayers
            self.bidirectional = bidirectional
        }
    }
}
