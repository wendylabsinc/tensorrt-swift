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
    ],
    targets: [
        .target(
            name: "TensorRT"
        ),
        .testTarget(
            name: "TensorRTTests",
            dependencies: ["TensorRT"]
        ),
    ],
    swiftLanguageVersions: [.v6]
)
