// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "TensorRT",
    products: [
        .library(
            name: "TensorRT",
            targets: ["TensorRT"]
        ),
        .library(
            name: "TensorRTLLM",
            targets: ["TensorRTLLM"]
        ),
        // Beginner Examples
        .executable(name: "HelloTensorRT", targets: ["HelloTensorRT"]),
        .executable(name: "ONNXInference", targets: ["ONNXInference"]),
        .executable(name: "BatchProcessing", targets: ["BatchProcessing"]),
        // Intermediate Examples
        .executable(name: "DynamicBatching", targets: ["DynamicBatching"]),
        .executable(name: "MultiProfile", targets: ["MultiProfile"]),
        .executable(name: "AsyncInference", targets: ["AsyncInference"]),
        .executable(name: "ImageClassifier", targets: ["ImageClassifier"]),
        .executable(name: "DeviceMemoryPipeline", targets: ["DeviceMemoryPipeline"]),
        // LLM Examples
        .executable(name: "StreamingLLM", targets: ["StreamingLLM"]),
        // Advanced Examples
        .executable(name: "MultiGPU", targets: ["MultiGPU"]),
        .executable(name: "CUDAEventPipelining", targets: ["CUDAEventPipelining"]),
        .executable(name: "BenchmarkSuite", targets: ["BenchmarkSuite"]),
        .executable(name: "FP16Quantization", targets: ["FP16Quantization"]),
        // Real-World Examples
        .executable(name: "TextEmbedding", targets: ["TextEmbedding"]),
        .executable(name: "ObjectDetection", targets: ["ObjectDetection"]),
        .executable(name: "WhisperTranscription", targets: ["WhisperTranscription"]),
        .executable(name: "VisionTransformer", targets: ["VisionTransformer"]),
    ],
    targets: [
        .target(
            name: "TensorRTNative",
            publicHeadersPath: "include",
            cxxSettings: [
                .unsafeFlags(["-std=c++17"], .when(platforms: [.linux])),
                .unsafeFlags(["-I/usr/local/cuda/include"], .when(platforms: [.linux])),
            ],
            linkerSettings: [
                .linkedLibrary("nvinfer", .when(platforms: [.linux])),
                .linkedLibrary("nvinfer_plugin", .when(platforms: [.linux])),
                .linkedLibrary("nvonnxparser", .when(platforms: [.linux])),
                .linkedLibrary("cuda", .when(platforms: [.linux])),
                .linkedLibrary("dl", .when(platforms: [.linux])),
            ]
        ),
        .target(
            name: "TensorRT",
            dependencies: [
                .target(name: "TensorRTNative", condition: .when(platforms: [.linux])),
            ]
        ),
        .target(
            name: "TensorRTLLM",
            dependencies: [
                "TensorRT",
            ]
        ),
        .testTarget(
            name: "TensorRTTests",
            dependencies: ["TensorRT"]
        ),

        // MARK: - Beginner Examples
        .executableTarget(
            name: "HelloTensorRT",
            dependencies: ["TensorRT"],
            path: "Examples/HelloTensorRT"
        ),
        .executableTarget(
            name: "ONNXInference",
            dependencies: ["TensorRT"],
            path: "Examples/ONNXInference"
        ),
        .executableTarget(
            name: "BatchProcessing",
            dependencies: ["TensorRT"],
            path: "Examples/BatchProcessing"
        ),

        // MARK: - Intermediate Examples
        .executableTarget(
            name: "DynamicBatching",
            dependencies: ["TensorRT"],
            path: "Examples/DynamicBatching"
        ),
        .executableTarget(
            name: "MultiProfile",
            dependencies: ["TensorRT"],
            path: "Examples/MultiProfile"
        ),
        .executableTarget(
            name: "AsyncInference",
            dependencies: ["TensorRT"],
            path: "Examples/AsyncInference"
        ),
        .executableTarget(
            name: "ImageClassifier",
            dependencies: ["TensorRT"],
            path: "Examples/ImageClassifier"
        ),
        .executableTarget(
            name: "DeviceMemoryPipeline",
            dependencies: ["TensorRT"],
            path: "Examples/DeviceMemoryPipeline"
        ),

        // MARK: - LLM Examples
        .executableTarget(
            name: "StreamingLLM",
            dependencies: ["TensorRTLLM"],
            path: "ExamplesLLM/StreamingLLM"
        ),

        // MARK: - Advanced Examples
        .executableTarget(
            name: "MultiGPU",
            dependencies: ["TensorRT"],
            path: "Examples/MultiGPU"
        ),
        .executableTarget(
            name: "CUDAEventPipelining",
            dependencies: ["TensorRT"],
            path: "Examples/CUDAEventPipelining"
        ),
        .executableTarget(
            name: "BenchmarkSuite",
            dependencies: ["TensorRT"],
            path: "Examples/BenchmarkSuite"
        ),
        .executableTarget(
            name: "FP16Quantization",
            dependencies: ["TensorRT"],
            path: "Examples/FP16Quantization"
        ),

        // MARK: - Real-World Examples
        .executableTarget(
            name: "TextEmbedding",
            dependencies: ["TensorRT"],
            path: "Examples/TextEmbedding"
        ),
        .executableTarget(
            name: "ObjectDetection",
            dependencies: ["TensorRT"],
            path: "Examples/ObjectDetection"
        ),
        .executableTarget(
            name: "WhisperTranscription",
            dependencies: ["TensorRT"],
            path: "Examples/WhisperTranscription"
        ),
        .executableTarget(
            name: "VisionTransformer",
            dependencies: ["TensorRT"],
            path: "Examples/VisionTransformer"
        ),
    ],
    swiftLanguageModes: [.v6]
)
