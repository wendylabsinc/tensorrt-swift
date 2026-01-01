import TensorRT

// MARK: - Streaming Inference (LLM)

/// A single step in a streaming inference sequence.
///
/// For LLM use cases, each step typically contains a single token or a small batch of tokens.
public struct StreamingInferenceStep: Sendable {
    /// The inference result for this step.
    public var result: InferenceResult

    /// The step index (0-based) in the sequence.
    public var stepIndex: Int

    /// Whether this is the final step in the sequence.
    public var isFinal: Bool

    /// Optional token IDs decoded from this step (for LLM use cases).
    public var tokenIds: [Int]?

    /// Optional decoded text from this step (for LLM use cases).
    public var text: String?

    /// Cumulative duration from the start of streaming.
    public var cumulativeDuration: Duration?

    public init(
        result: InferenceResult,
        stepIndex: Int,
        isFinal: Bool = false,
        tokenIds: [Int]? = nil,
        text: String? = nil,
        cumulativeDuration: Duration? = nil
    ) {
        self.result = result
        self.stepIndex = stepIndex
        self.isFinal = isFinal
        self.tokenIds = tokenIds
        self.text = text
        self.cumulativeDuration = cumulativeDuration
    }
}

/// Configuration for streaming inference.
public struct StreamingConfiguration: Sendable {
    /// Maximum number of steps to generate.
    public var maxSteps: Int

    /// Name of the output tensor containing token logits/probabilities.
    public var outputTensorName: String?

    /// Optional stop condition that examines each step and returns true to stop.
    public var stopCondition: (@Sendable (StreamingInferenceStep) -> Bool)?

    /// Whether to yield intermediate results or only final results.
    public var yieldIntermediateResults: Bool

    public init(
        maxSteps: Int = 1024,
        outputTensorName: String? = nil,
        stopCondition: (@Sendable (StreamingInferenceStep) -> Bool)? = nil,
        yieldIntermediateResults: Bool = true
    ) {
        self.maxSteps = maxSteps
        self.outputTensorName = outputTensorName
        self.stopCondition = stopCondition
        self.yieldIntermediateResults = yieldIntermediateResults
    }
}

/// An `AsyncSequence` that yields inference results step-by-step.
///
/// This is the primary API for streaming inference, particularly useful for LLM token-by-token generation.
///
/// Example usage:
/// ```swift
/// let stream = context.stream(
///     initialBatch: batch,
///     configuration: StreamingConfiguration(maxSteps: 100)
/// ) { previousResult in
///     // Transform previous output into next input
///     return nextBatch
/// }
///
/// for try await step in stream {
///     print("Step \(step.stepIndex): \(step.text ?? "")")
///     if step.isFinal { break }
/// }
/// ```
public struct InferenceStream: AsyncSequence, Sendable {
    public typealias Element = StreamingInferenceStep

    private let context: ExecutionContext
    private let initialBatch: InferenceBatch
    private let configuration: StreamingConfiguration
    private let nextBatchGenerator: @Sendable (InferenceResult) async throws -> InferenceBatch?

    public init(
        context: ExecutionContext,
        initialBatch: InferenceBatch,
        configuration: StreamingConfiguration,
        nextBatchGenerator: @escaping @Sendable (InferenceResult) async throws -> InferenceBatch?
    ) {
        self.context = context
        self.initialBatch = initialBatch
        self.configuration = configuration
        self.nextBatchGenerator = nextBatchGenerator
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(
            context: context,
            initialBatch: initialBatch,
            configuration: configuration,
            nextBatchGenerator: nextBatchGenerator
        )
    }

    public struct AsyncIterator: AsyncIteratorProtocol {
        private let context: ExecutionContext
        private let configuration: StreamingConfiguration
        private let nextBatchGenerator: @Sendable (InferenceResult) async throws -> InferenceBatch?

        private var currentBatch: InferenceBatch?
        private var stepIndex: Int = 0
        private var isFinished: Bool = false
        private var startTime: ContinuousClock.Instant?
        private var lastResult: InferenceResult?

        init(
            context: ExecutionContext,
            initialBatch: InferenceBatch,
            configuration: StreamingConfiguration,
            nextBatchGenerator: @escaping @Sendable (InferenceResult) async throws -> InferenceBatch?
        ) {
            self.context = context
            self.currentBatch = initialBatch
            self.configuration = configuration
            self.nextBatchGenerator = nextBatchGenerator
        }

        public mutating func next() async throws -> StreamingInferenceStep? {
            guard !isFinished else { return nil }
            guard stepIndex < configuration.maxSteps else {
                isFinished = true
                return nil
            }
            guard let batch = currentBatch else {
                isFinished = true
                return nil
            }

            if startTime == nil {
                startTime = ContinuousClock.now
            }

            // Execute inference for current step
            let result = try await context.enqueue(batch, synchronously: true)
            lastResult = result

            let cumulativeDuration = startTime.map { ContinuousClock.now - $0 }

            // Try to generate next batch
            let nextBatch = try await nextBatchGenerator(result)
            let isFinal = nextBatch == nil || stepIndex + 1 >= configuration.maxSteps

            let step = StreamingInferenceStep(
                result: result,
                stepIndex: stepIndex,
                isFinal: isFinal,
                tokenIds: nil,
                text: nil,
                cumulativeDuration: cumulativeDuration
            )

            // Check stop condition
            if let stopCondition = configuration.stopCondition, stopCondition(step) {
                isFinished = true
                return StreamingInferenceStep(
                    result: result,
                    stepIndex: stepIndex,
                    isFinal: true,
                    tokenIds: step.tokenIds,
                    text: step.text,
                    cumulativeDuration: cumulativeDuration
                )
            }

            stepIndex += 1
            currentBatch = nextBatch

            if isFinal {
                isFinished = true
            }

            return step
        }
    }
}

/// Extension to ExecutionContext for streaming inference.
public extension ExecutionContext {
    /// Creates a streaming inference sequence for step-by-step generation.
    ///
    /// This is the primary API for LLM-style autoregressive generation where each step's
    /// output becomes part of the next step's input.
    ///
    /// Example:
    /// ```swift
    /// let stream = context.stream(
    ///     initialBatch: promptBatch,
    ///     configuration: .init(maxSteps: 100)
    /// ) { previousResult in
    ///     // Build next batch from previous output
    ///     return makeNextBatch(from: previousResult)
    /// }
    ///
    /// for try await step in stream {
    ///     if let tokens = step.tokenIds {
    ///         print(decode(tokens))
    ///     }
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - initialBatch: The first batch to process (e.g., the prompt for an LLM).
    ///   - configuration: Configuration for the streaming session.
    ///   - nextBatchGenerator: A closure that transforms the previous result into the next batch.
    ///                         Return `nil` to stop generation.
    /// - Returns: An `AsyncSequence` of `StreamingInferenceStep` values.
    func stream(
        initialBatch: InferenceBatch,
        configuration: StreamingConfiguration = StreamingConfiguration(),
        nextBatchGenerator: @escaping @Sendable (InferenceResult) async throws -> InferenceBatch?
    ) -> InferenceStream {
        InferenceStream(
            context: self,
            initialBatch: initialBatch,
            configuration: configuration,
            nextBatchGenerator: nextBatchGenerator
        )
    }

    /// Convenience method for simple repeated inference without batch transformation.
    ///
    /// Runs the same batch repeatedly for a fixed number of steps. Useful for benchmarking
    /// or when the model doesn't require autoregressive input transformation.
    ///
    /// - Parameters:
    ///   - batch: The batch to run repeatedly.
    ///   - steps: Number of inference steps to run.
    /// - Returns: An `AsyncSequence` of `StreamingInferenceStep` values.
    func streamRepeated(
        batch: InferenceBatch,
        steps: Int
    ) -> InferenceStream {
        stream(
            initialBatch: batch,
            configuration: StreamingConfiguration(maxSteps: steps)
        ) { _ in batch }
    }
}

/// Convenience initializers for common streaming patterns.
public extension StreamingConfiguration {
    /// Creates a configuration that stops when a specific token is generated.
    ///
    /// - Parameters:
    ///   - stopTokenId: The token ID that signals end of generation (e.g., EOS token).
    ///   - maxSteps: Maximum steps before forced stop.
    ///   - outputTensorName: Name of the output tensor containing token IDs.
    static func stoppingAt(
        tokenId stopTokenId: Int,
        maxSteps: Int = 1024,
        outputTensorName: String = "output"
    ) -> StreamingConfiguration {
        StreamingConfiguration(
            maxSteps: maxSteps,
            outputTensorName: outputTensorName,
            stopCondition: { step in
                step.tokenIds?.contains(stopTokenId) ?? false
            }
        )
    }

    /// Creates a configuration that stops when any of the specified tokens is generated.
    ///
    /// - Parameters:
    ///   - stopTokenIds: Token IDs that signal end of generation.
    ///   - maxSteps: Maximum steps before forced stop.
    ///   - outputTensorName: Name of the output tensor containing token IDs.
    static func stoppingAt(
        tokenIds stopTokenIds: Set<Int>,
        maxSteps: Int = 1024,
        outputTensorName: String = "output"
    ) -> StreamingConfiguration {
        StreamingConfiguration(
            maxSteps: maxSteps,
            outputTensorName: outputTensorName,
            stopCondition: { step in
                guard let tokens = step.tokenIds else { return false }
                return !stopTokenIds.isDisjoint(with: tokens)
            }
        )
    }
}
