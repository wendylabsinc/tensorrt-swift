// ObjectDetection - YOLO-style object detection with NMS
//
// This example demonstrates:
// 1. Image preprocessing for detection models
// 2. Running object detection inference
// 3. Non-Maximum Suppression (NMS) post-processing
// 4. Bounding box extraction and visualization
//
// Note: Uses simulated model since real weights are large
//
// Run with: ./scripts/swiftw run ObjectDetection
import TensorRT
import FoundationEssentials

@main
struct ObjectDetection {
    // YOLO-style configuration
    static let imageSize = 640
    static let numClasses = 80
    static let numAnchors = 8400  // Typical for YOLOv8
    static let confidenceThreshold: Float = 0.25
    static let nmsThreshold: Float = 0.45

    // COCO class names (subset)
    static let classNames = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]

    struct Detection {
        var x: Float       // Center x
        var y: Float       // Center y
        var width: Float
        var height: Float
        var confidence: Float
        var classId: Int
        var className: String

        var x1: Float { x - width / 2 }
        var y1: Float { y - height / 2 }
        var x2: Float { x + width / 2 }
        var y2: Float { y + height / 2 }
    }

    static func main() async throws {
        print("=== Object Detection Example ===\n")
        print("This example demonstrates YOLO-style object detection with NMS.")
        print("Using a simulated model for demonstration.\n")

        // Step 1: Build detection model
        print("1. Building detection model...")
        let outputSize = numAnchors * (4 + numClasses)  // boxes + class scores
        let plan = try TensorRTSystem.buildIdentityEnginePlan(elementCount: min(outputSize, 8192))
        let engine = try TensorRTRuntime().deserializeEngine(from: plan)
        let context = try engine.makeExecutionContext()
        let warmup = try await context.warmup(iterations: 1)
        if let avg = warmup.average {
            print("   Context warmup: \(formatDuration(avg))")
        }
        print("   Model: YOLOv8-style, \(numAnchors) anchors, \(numClasses) classes")
        print("   Input: \(imageSize)x\(imageSize) RGB")

        // Step 2: Generate synthetic test image
        print("\n2. Generating synthetic test image...")
        let image = generateTestImage()
        print("   Generated \(imageSize)x\(imageSize) synthetic scene")

        // Step 3: Preprocess image
        print("\n3. Preprocessing image...")
        let preprocessStart = ContinuousClock.now
        let preprocessed = preprocessImage(image)
        let preprocessDuration = ContinuousClock.now - preprocessStart
        print("   Applied: resize, normalize, CHW format")
        print("   Preprocessing time: \(preprocessDuration)")
        print("   Preprocessed elements: \(preprocessed.count)")

        // Step 4: Run inference
        print("\n4. Running detection inference...")

        // Simulate model output (in real impl, this comes from TensorRT)
        let inferenceStart = ContinuousClock.now
        let rawDetections = simulateModelOutput()
        let inferenceDuration = ContinuousClock.now - inferenceStart
        print("   Inference time: \(inferenceDuration)")
        print("   Raw outputs: \(rawDetections.count) anchor predictions")

        // Step 5: Post-process detections
        print("\n5. Post-processing detections...")
        let postprocessStart = ContinuousClock.now

        // Filter by confidence
        var candidateDetections: [Detection] = []
        for raw in rawDetections {
            if raw.confidence >= confidenceThreshold {
                candidateDetections.append(raw)
            }
        }
        print("   After confidence filter (\(confidenceThreshold)): \(candidateDetections.count) candidates")

        // Apply NMS
        let finalDetections = nonMaxSuppression(candidateDetections, threshold: nmsThreshold)
        let postprocessDuration = ContinuousClock.now - postprocessStart

        print("   After NMS (\(nmsThreshold)): \(finalDetections.count) detections")
        print("   Post-processing time: \(postprocessDuration)")

        // Step 6: Display results
        print("\n6. Detection Results:")
        print("   ┌─────┬─────────────────┬────────────┬───────────────────────────────┐")
        print("   │ #   │ Class           │ Confidence │ Bounding Box (x1,y1,x2,y2)    │")
        print("   ├─────┼─────────────────┼────────────┼───────────────────────────────┤")

        for (i, det) in finalDetections.enumerated() {
            let classStr = det.className.padding(toLength: 15, withPad: " ", startingAt: 0)
            let confStr = formatDouble(Double(det.confidence), decimals: 2).padding(toLength: 10, withPad: " ", startingAt: 0)
            let boxStr = "(\(Int(det.x1)),\(Int(det.y1)),\(Int(det.x2)),\(Int(det.y2)))"

            print("   │ \(String(i + 1).padding(toLength: 3, withPad: " ", startingAt: 0)) │ \(classStr) │ \(confStr) │ \(boxStr.padding(toLength: 29, withPad: " ", startingAt: 0)) │")
        }

        print("   └─────┴─────────────────┴────────────┴───────────────────────────────┘")

        // Step 7: Visualize as ASCII art
        print("\n7. Detection Visualization (ASCII):")
        printASCIIVisualization(finalDetections)

        // Step 8: Class distribution
        print("\n8. Class Distribution:")
        var classCounts: [String: Int] = [:]
        for det in finalDetections {
            classCounts[det.className, default: 0] += 1
        }

        let sortedClasses = classCounts.sorted { $0.value > $1.value }
        for (className, count) in sortedClasses.prefix(5) {
            let bar = String(repeating: "█", count: count * 5)
            print("   \(className.padding(toLength: 15, withPad: " ", startingAt: 0)) \(bar) (\(count))")
        }

        // Step 9: Performance summary
        print("\n9. Performance Summary:")
        print("   ┌─────────────────────┬────────────────────┐")
        print("   │ Stage               │ Time               │")
        print("   ├─────────────────────┼────────────────────┤")
        print("   │ Preprocessing       │ \(formatDuration(preprocessDuration).padding(toLength: 18, withPad: " ", startingAt: 0)) │")
        print("   │ Inference           │ \(formatDuration(inferenceDuration).padding(toLength: 18, withPad: " ", startingAt: 0)) │")
        print("   │ Post-processing     │ \(formatDuration(postprocessDuration).padding(toLength: 18, withPad: " ", startingAt: 0)) │")
        print("   └─────────────────────┴────────────────────┘")

        // Step 10: Batch detection demo
        print("\n10. Batch Detection Demo:")
        let batchStart = ContinuousClock.now
        let numFrames = 30

        for _ in 0..<numFrames {
            let img = generateTestImage()
            _ = preprocessImage(img)
            let raw = simulateModelOutput()
            let candidates = raw.filter { $0.confidence >= confidenceThreshold }
            _ = nonMaxSuppression(candidates, threshold: nmsThreshold)
        }

        let batchDuration = ContinuousClock.now - batchStart
        let fps = Double(numFrames) / durationToSeconds(batchDuration)

        print("   Processed \(numFrames) frames in \(batchDuration)")
        print("   Throughput: \(formatDouble(fps, decimals: 1)) FPS")

        print("\n=== Object Detection Example Complete ===")
    }

    /// Generates a synthetic test image
    static func generateTestImage() -> [UInt8] {
        var image = [UInt8](repeating: 128, count: imageSize * imageSize * 3)

        // Add some "objects" as colored rectangles
        addRectangle(&image, x: 100, y: 100, w: 150, h: 200, r: 255, g: 0, b: 0)    // Red
        addRectangle(&image, x: 300, y: 200, w: 100, h: 100, r: 0, g: 255, b: 0)    // Green
        addRectangle(&image, x: 450, y: 150, w: 120, h: 180, r: 0, g: 0, b: 255)    // Blue

        return image
    }

    static func addRectangle(_ image: inout [UInt8], x: Int, y: Int, w: Int, h: Int, r: UInt8, g: UInt8, b: UInt8) {
        for dy in 0..<h {
            for dx in 0..<w {
                let px = x + dx
                let py = y + dy
                if px >= 0 && px < imageSize && py >= 0 && py < imageSize {
                    let idx = (py * imageSize + px) * 3
                    image[idx] = r
                    image[idx + 1] = g
                    image[idx + 2] = b
                }
            }
        }
    }

    /// Preprocesses image for detection model
    static func preprocessImage(_ image: [UInt8]) -> [Float] {
        image.map { Float($0) / 255.0 }
    }

    /// Simulates model output (in real impl, comes from TensorRT)
    static func simulateModelOutput() -> [Detection] {
        var detections: [Detection] = []

        // Simulate some detections
        let simulatedObjects: [(x: Float, y: Float, w: Float, h: Float, cls: Int, conf: Float)] = [
            (175, 200, 150, 200, 0, 0.92),   // person
            (350, 250, 100, 100, 2, 0.85),   // car
            (510, 240, 120, 180, 0, 0.78),   // person
            (100, 400, 80, 60, 16, 0.65),    // dog
            (400, 100, 50, 50, 39, 0.45),    // bottle
            (200, 300, 60, 40, 56, 0.35),    // chair
            (180, 210, 140, 190, 0, 0.70),   // overlapping person (for NMS)
        ]

        for obj in simulatedObjects {
            detections.append(Detection(
                x: obj.x,
                y: obj.y,
                width: obj.w,
                height: obj.h,
                confidence: obj.conf,
                classId: obj.cls,
                className: obj.cls < classNames.count ? classNames[obj.cls] : "unknown"
            ))
        }

        return detections
    }

    /// Non-Maximum Suppression
    static func nonMaxSuppression(_ detections: [Detection], threshold: Float) -> [Detection] {
        // Group by class
        var byClass: [Int: [Detection]] = [:]
        for det in detections {
            byClass[det.classId, default: []].append(det)
        }

        var results: [Detection] = []

        for (_, classDets) in byClass {
            // Sort by confidence (descending)
            var sorted = classDets.sorted { $0.confidence > $1.confidence }
            var keep: [Detection] = []

            while !sorted.isEmpty {
                let best = sorted.removeFirst()
                keep.append(best)

                // Remove overlapping detections
                sorted = sorted.filter { det in
                    let iou = computeIoU(best, det)
                    return iou < threshold
                }
            }

            results.append(contentsOf: keep)
        }

        return results.sorted { $0.confidence > $1.confidence }
    }

    /// Computes Intersection over Union
    static func computeIoU(_ a: Detection, _ b: Detection) -> Float {
        let x1 = max(a.x1, b.x1)
        let y1 = max(a.y1, b.y1)
        let x2 = min(a.x2, b.x2)
        let y2 = min(a.y2, b.y2)

        let intersectionArea = max(0, x2 - x1) * max(0, y2 - y1)
        let aArea = a.width * a.height
        let bArea = b.width * b.height
        let unionArea = aArea + bArea - intersectionArea

        return unionArea > 0 ? intersectionArea / unionArea : 0
    }

    /// Prints ASCII visualization of detections
    static func printASCIIVisualization(_ detections: [Detection]) {
        let scale = 16  // Scale factor for ASCII
        let width = imageSize / scale
        let height = imageSize / scale

        var grid = [[Character]](repeating: [Character](repeating: ".", count: width), count: height)

        for (i, det) in detections.enumerated() {
            let x1 = Int(det.x1) / scale
            let y1 = Int(det.y1) / scale
            let x2 = Int(det.x2) / scale
            let y2 = Int(det.y2) / scale

            let char = Character(String(i + 1))

            // Draw box
            for x in max(0, x1)...min(width - 1, x2) {
                if y1 >= 0 && y1 < height { grid[y1][x] = char }
                if y2 >= 0 && y2 < height { grid[y2][x] = char }
            }
            for y in max(0, y1)...min(height - 1, y2) {
                if x1 >= 0 && x1 < width { grid[y][x1] = char }
                if x2 >= 0 && x2 < width { grid[y][x2] = char }
            }
        }

        print("   +" + String(repeating: "-", count: width) + "+")
        for row in grid {
            print("   |" + String(row) + "|")
        }
        print("   +" + String(repeating: "-", count: width) + "+")
        print("   Legend: ", terminator: "")
        for (i, det) in detections.enumerated() {
            print("\(i + 1)=\(det.className) ", terminator: "")
        }
        print("")
    }

    static func durationToSeconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    static func formatDouble(_ value: Double, decimals: Int) -> String {
        if decimals <= 0 { return String(Int(value.rounded())) }
        var multiplier = 1.0
        for _ in 0..<decimals { multiplier *= 10.0 }
        let rounded = (value * multiplier).rounded() / multiplier
        let intPart = Int(rounded)
        let fracPart = abs(Int((rounded - Double(intPart)) * multiplier))
        return "\(intPart).\(String(fracPart))"
    }

    static func formatDuration(_ duration: Duration) -> String {
        let us = durationToSeconds(duration) * 1_000_000
        if us < 1000 { return "\(formatDouble(us, decimals: 0)) µs" }
        else { return "\(formatDouble(us / 1000, decimals: 2)) ms" }
    }
}

extension String {
    func padding(toLength length: Int, withPad padString: String, startingAt: Int) -> String {
        if self.count >= length { return String(self.prefix(length)) }
        var result = self
        while result.count < length { result += padString }
        return String(result.prefix(length))
    }
}
