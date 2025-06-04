import AVFoundation
import CoreML
import CoreMedia
import UIKit
import Vision
import CoreImage

// YOLO 模型
var mlModel: MLModel = {
    guard let path = Bundle.main.path(forResource: "best", ofType: "mlmodelc") else {
        // 列出 Bundle 內容以調試
        let fileManager = FileManager.default
        if let bundlePath = Bundle.main.bundlePath as NSString? {
            do {
                let contents = try fileManager.contentsOfDirectory(atPath: bundlePath as String)
                print("Bundle 內容：\(contents)")
                let modelsPath = bundlePath.appendingPathComponent("Models")
                if fileManager.fileExists(atPath: modelsPath) {
                    let modelsContents = try fileManager.contentsOfDirectory(atPath: modelsPath)
                    print("Models/ 資料夾內容：\(modelsContents)")
                }
            } catch {
                print("無法列出 Bundle 內容：\(error)")
            }
        }
        fatalError("無法找到 best.mlmodelc 檔案，請確認檔案已正確添加到專案")
    }
    let modelURL = URL(fileURLWithPath: path)
    do {
        return try MLModel(contentsOf: modelURL, configuration: mlmodelConfig)
    } catch {
        fatalError("無法載入 best.mlmodelc 模型：\(error.localizedDescription)")
    }
}()

var mlmodelConfig: MLModelConfiguration = {
    let config = MLModelConfiguration()
    if #available(iOS 17.0, *) {
        config.setValue(1, forKey: "experimentalMLE5EngineUsage")
    }
    return config
}()

// MegaDescriptor 模型
var megaDescriptorModel: MLModel? = {
    do {
        if let path = Bundle.main.path(forResource: "MegaDescriptor", ofType: "mlmodelc") {
            let modelURL = URL(fileURLWithPath: path)
            let config = MLModelConfiguration()
            config.computeUnits = .all
            return try MLModel(contentsOf: modelURL, configuration: config)
        }
        return nil
    } catch {
        print("無法載入 MegaDescriptor 模型：\(error.localizedDescription)")
        return nil
    }
}()

struct DetectionResult {
    let boundingBox: CGRect
    let name: String
    let confidence: Float
}

class ViewController: UIViewController {
    @IBOutlet var videoPreview: UIView!
    @IBOutlet var View0: UIView!
    @IBOutlet var playButtonOutlet: UIBarButtonItem!
    @IBOutlet var pauseButtonOutlet: UIBarButtonItem!
    @IBOutlet weak var labelName: UILabel!
    @IBOutlet weak var labelFPS: UILabel!
    @IBOutlet weak var labelZoom: UILabel!
    @IBOutlet weak var labelVersion: UILabel!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var focus: UIImageView!
    @IBOutlet weak var toolBar: UIToolbar!

    // 新增UI元素
    private lazy var resultStackView: UIStackView = {
        let stack = UIStackView()
        stack.axis = .vertical
        stack.spacing = 8
        stack.alignment = .center
        stack.distribution = .fillEqually
        stack.isHidden = true
        return stack
    }()
    
    private lazy var resultLabels: [UILabel] = {
        return (0..<3).map { _ in
            let label = UILabel()
            label.textColor = .white
            label.textAlignment = .center
            label.font = .systemFont(ofSize: 16, weight: .medium)
            label.backgroundColor = UIColor.black.withAlphaComponent(0.6)
            label.layer.cornerRadius = 8
            label.clipsToBounds = true
            return label
        }
    }()

    // 新增屬性
    private var continuousResults: [(name: String, score: Float, confidence: Float)] = []
    private let requiredResults = 2  // 只需要2次結果
    private var isProcessing = false
    private var hasDetectedDonkey = false
    private var currentDonkeyBoundingBox: CGRect?
    
    private lazy var hintLabel: UILabel = {
        let label = UILabel()
        label.text = "Point camera at donkey"
        label.textColor = .white
        label.textAlignment = .center
        label.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        label.layer.cornerRadius = 8
        label.clipsToBounds = true
        return label
    }()
    
    private lazy var identifyButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Identify Donkey", for: .normal)
        button.setTitleColor(.white, for: .normal)
        button.backgroundColor = .systemBlue
        button.layer.cornerRadius = 8
        button.addTarget(self, action: #selector(identifyButtonTapped), for: .touchUpInside)
        button.isHidden = true
        return button
    }()

    let selection = UISelectionFeedbackGenerator()
    var detector = try! VNCoreMLModel(for: mlModel)
    var session: AVCaptureSession!
    var videoCapture: VideoCapture!
    var currentBuffer: CVPixelBuffer?
    var framesDone = 0
    var t0 = 0.0
    var t1 = 0.0
    var t2 = 0.0
    var t3 = CACurrentMediaTime()
    var t4 = 0.0
    var longSide: CGFloat = 3
    var shortSide: CGFloat = 4
    var frameSizeCaptured = false
    var features: [[Float]] = []
    var labels: [String] = []
    var featureVector: [Float]?

    let donkeyClassLabel = "donkey"

    lazy var visionRequest: VNCoreMLRequest = {
        let request = VNCoreMLRequest(
            model: detector,
            completionHandler: { [weak self] request, error in
                self?.processObservations(for: request, error: error)
            })
        request.imageCropAndScaleOption = .scaleFill
        return request
    }()

    // 添加特徵緩存
    private var featureCache: [String: [Float]] = [:]
    private let cacheQueue = DispatchQueue(label: "com.donkeyrecognition.featurecache")

    override func viewDidLoad() {
        super.viewDidLoad()
        // 設置 UserDefaults 中的 app_version（如果未設置）
        if UserDefaults.standard.string(forKey: "app_version") == nil {
            UserDefaults.standard.set("1.0.0", forKey: "app_version")
        }
        setLabels()
        setUpBoundingBoxViews()
        setUpOrientationChangeNotification()
        startVideo()
        setModel()
        
        // 設定 toolbar 高度
        toolBar.frame.size.height = 49
        
        // 新增驢子保護區網站按鈕到最右側
        let sanctuaryButton = UIBarButtonItem(image: UIImage(systemName: "globe"), style: .plain, target: self, action: #selector(openSanctuaryWebsite))
        toolBar.items?.append(UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil))
        toolBar.items?.append(sanctuaryButton)
        
        // 移除分享按鈕
        if let items = toolBar.items {
            toolBar.items = items.filter { $0.action != #selector(shareButton(_:)) }
        }

        // 異步載入特徵庫
        DispatchQueue.global(qos: .userInitiated).async {
            if let url = Bundle.main.url(forResource: "features_and_labels", withExtension: "json") {
                print("找到 features_and_labels.json：\(url.path)")
                do {
                    let data = try Data(contentsOf: url)
                    let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
                    
                    // 嘗試解析 features
                    if let rawFeatures = json?["features"] as? [[Any]] {
                        // 將 [[Any]] 轉換為 [[Float]]
                        self.features = rawFeatures.map { feature in
                            feature.compactMap { value in
                                if let floatValue = value as? Float {
                                    return floatValue
                                } else if let doubleValue = value as? Double {
                                    return Float(doubleValue)
                                } else if let intValue = value as? Int {
                                    return Float(intValue)
                                }
                                return nil
                            }
                        }
                    } else {
                        print("無法解析 features 數據，可能是格式不正確")
                        self.features = []
                    }

                    // 解析 labels
                    self.labels = (json?["labels"] as? [String]) ?? []
                    
                    print("成功載入 features_and_labels.json，features 數量：\(self.features.count)，labels 數量：\(self.labels.count)")
                    
                    // 檢查 features 和 labels 數量是否一致
                    if self.features.count != self.labels.count {
                        print("警告：features 和 labels 數量不一致，features: \(self.features.count), labels: \(self.labels.count)")
                    }
                    // 檢查第一個特徵向量的長度（應為 768）
                    if let firstFeature = self.features.first {
                        print("第一個特徵向量長度：\(firstFeature.count)")
                        if firstFeature.count != 768 {
                            print("警告：特徵向量長度不為 768，可能導致匹配失敗")
                        }
                    }
                } catch {
                    print("無法載入 features_and_labels.json 檔案，錯誤：\(error.localizedDescription)")
                }
            } else {
                print("無法找到 features_and_labels.json 檔案，請確認檔案已正確添加到專案")
            }
        }

        setupHintLabelAndButton()
        setupResultLabels()
    }

    override func viewWillTransition(to size: CGSize, with coordinator: any UIViewControllerTransitionCoordinator) {
        super.viewWillTransition(to: size, with: coordinator)
        self.videoCapture.previewLayer?.frame = CGRect(x: 0, y: 0, width: size.width, height: size.height)
    }

    private func setUpOrientationChangeNotification() {
        NotificationCenter.default.addObserver(
            self, selector: #selector(orientationDidChange),
            name: UIDevice.orientationDidChangeNotification, object: nil)
    }

    @objc func orientationDidChange() {
        videoCapture.updateVideoOrientation()
    }

    @IBAction func vibrate(_ sender: Any) {
        selection.selectionChanged()
    }

    func setModel() {
        do {
            detector = try VNCoreMLModel(for: mlModel)
            detector.featureProvider = ThresholdProvider()
            let request = VNCoreMLRequest(
                model: detector,
                completionHandler: { [weak self] request, error in
                    self?.processObservations(for: request, error: error)
                })
            request.imageCropAndScaleOption = .scaleFill
            visionRequest = request
            t2 = 0.0
            t3 = CACurrentMediaTime()
            t4 = 0.0
        } catch {
            print("無法初始化 VNCoreMLModel：\(error.localizedDescription)")
        }
    }

    @IBAction func takePhoto(_ sender: Any?) {
        let settings = AVCapturePhotoSettings()
        usleep(20_000)
        self.videoCapture.cameraOutput.capturePhoto(with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
    }

    @IBAction func logoButton(_ sender: Any) {
        selection.selectionChanged()
        if let link = URL(string: "https://www.ultralytics.com") {
            UIApplication.shared.open(link)
        }
    }

    func setLabels() {
        self.labelName?.text = "Ask ELVIS"
        self.labelVersion?.text = "Version " + (UserDefaults.standard.string(forKey: "app_version") ?? "Unknown")
    }

    @IBAction func playButton(_ sender: Any) {
        selection.selectionChanged()
        self.videoCapture.start()
        playButtonOutlet?.isEnabled = false
        pauseButtonOutlet?.isEnabled = true
    }

    @IBAction func pauseButton(_ sender: Any?) {
        selection.selectionChanged()
        self.videoCapture.stop()
        playButtonOutlet?.isEnabled = true
        pauseButtonOutlet?.isEnabled = false
    }

    @IBAction func shareButton(_ sender: Any) {
        selection.selectionChanged()
        let settings = AVCapturePhotoSettings()
        self.videoCapture.cameraOutput.capturePhoto(with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
    }

    let maxBoundingBoxViews = 10
    var boundingBoxViews = [BoundingBoxView]()

    func setUpBoundingBoxViews() {
        while boundingBoxViews.count < maxBoundingBoxViews {
            boundingBoxViews.append(BoundingBoxView())
        }
    }

    func startVideo() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self

        videoCapture.setUp(sessionPreset: .hd1280x720) { success in
            if success {
                if let previewLayer = self.videoCapture.previewLayer {
                    self.videoPreview?.layer.addSublayer(previewLayer)
                    self.videoCapture.previewLayer?.frame = self.videoPreview?.bounds ?? CGRect.zero
                }

                for box in self.boundingBoxViews {
                    box.addToLayer(self.videoPreview?.layer ?? CALayer())
                }

                self.videoCapture.start()
            } else {
                print("無法初始化視訊捕捉")
            }
        }
    }

    func predict(sampleBuffer: CMSampleBuffer) {
        if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            currentBuffer = pixelBuffer
            if !frameSizeCaptured {
                let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
                longSide = max(frameWidth, frameHeight)
                shortSide = min(frameWidth, frameHeight)
                frameSizeCaptured = true
                print("✅ 捕獲到幀尺寸：\(frameWidth)x\(frameHeight)")
            }

            let imageOrientation: CGImagePropertyOrientation
            switch UIDevice.current.orientation {
            case .portrait:
                imageOrientation = .up
            case .portraitUpsideDown:
                imageOrientation = .down
            case .landscapeLeft:
                imageOrientation = .up
            case .landscapeRight:
                imageOrientation = .up
            case .unknown:
                imageOrientation = .up
            default:
                imageOrientation = .up
            }

            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
            if UIDevice.current.orientation != .faceUp {
                t0 = CACurrentMediaTime()
                do {
                    try handler.perform([visionRequest])
                } catch {
                    print("❌ 視覺請求失敗：\(error.localizedDescription)")
                }
                t1 = CACurrentMediaTime() - t0
            }
        } else {
            print("❌ 無法從 sampleBuffer 獲取 pixelBuffer")
        }
    }

    func processObservations(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            if let error = error {
                print("❌ 請求錯誤：\(error.localizedDescription)")
            }
            
            if let results = request.results as? [VNRecognizedObjectObservation] {
                let donkeyResults = results.filter { observation in
                    observation.labels.contains { $0.identifier == self.donkeyClassLabel }
                }
                
                // 更新 UI 狀態
                self.hasDetectedDonkey = !donkeyResults.isEmpty
                self.hintLabel.isHidden = self.hasDetectedDonkey
                self.identifyButton.isHidden = !self.hasDetectedDonkey
                
                if let firstDonkey = donkeyResults.first {
                    self.currentDonkeyBoundingBox = firstDonkey.boundingBox
                    // 只顯示 YOLO 的檢測結果
                    self.show(predictions: [DetectionResult(boundingBox: firstDonkey.boundingBox, name: "Donkey", confidence: firstDonkey.confidence)])
                } else {
                    self.currentDonkeyBoundingBox = nil
                    self.show(predictions: [])
                }
            } else {
                self.hasDetectedDonkey = false
                self.hintLabel.isHidden = false
                self.identifyButton.isHidden = true
                self.currentDonkeyBoundingBox = nil
                self.show(predictions: [])
            }
            
            // 更新 FPS 顯示
            if self.t1 < 10.0 {
                self.t2 = self.t1 * 0.05 + self.t2 * 0.95
            }
            self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95
            self.labelFPS?.text = String(format: "%.1f FPS - %.1f ms", 1 / self.t4, self.t2 * 1000)
            self.t3 = CACurrentMediaTime()
        }
    }

    func show(predictions: [DetectionResult]) {
        let width = videoPreview?.bounds.width ?? 0
        let height = videoPreview?.bounds.height ?? 0

        if UIDevice.current.orientation == .portrait {
            var ratio: CGFloat = 1.0
            if videoCapture.captureSession.sessionPreset == .photo {
                ratio = (height / width) / (4.0 / 3.0)
            } else {
                ratio = (height / width) / (16.0 / 9.0)
            }

            for i in 0..<boundingBoxViews.count {
                if i < predictions.count {
                    let prediction = predictions[i]

                    var rect = prediction.boundingBox
                    switch UIDevice.current.orientation {
                    case .portraitUpsideDown:
                        rect = CGRect(
                            x: 1.0 - rect.origin.x - rect.width,
                            y: 1.0 - rect.origin.y - rect.height,
                            width: rect.width,
                            height: rect.height)
                    case .landscapeLeft, .landscapeRight, .unknown:
                        break
                    default:
                        break
                    }

                    if ratio >= 1 {
                        let offset = (1 - ratio) * (0.5 - rect.minX)
                        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
                        rect = rect.applying(transform)
                        rect.size.width *= ratio
                    } else {
                        let offset = (ratio - 1) * (0.5 - rect.maxY)
                        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
                        rect = rect.applying(transform)
                        ratio = (height / width) / (3.0 / 4.0)
                        rect.size.height /= ratio
                    }

                    rect = VNImageRectForNormalizedRect(rect, Int(width), Int(height))

                    let label = String(format: "%@ %.1f%%", prediction.name, prediction.confidence * 100)
                    boundingBoxViews[i].show(frame: rect, label: label)
                } else {
                    boundingBoxViews[i].hide()
                }
            }
        } else {
            let frameAspectRatio = longSide / shortSide
            let viewAspectRatio = width / height
            var scaleX: CGFloat = 1.0
            var scaleY: CGFloat = 1.0
            var offsetX: CGFloat = 0.0
            var offsetY: CGFloat = 0.0

            if frameAspectRatio > viewAspectRatio {
                scaleY = height / shortSide
                scaleX = scaleY
                offsetX = (longSide * scaleX - width) / 2
            } else {
                scaleX = width / longSide
                scaleY = scaleX
                offsetY = (shortSide * scaleY - height) / 2
            }

            for i in 0..<boundingBoxViews.count {
                if i < predictions.count {
                    let prediction = predictions[i]

                    var rect = prediction.boundingBox
                    rect.origin.x = rect.origin.x * longSide * scaleX - offsetX
                    rect.origin.y = height - (rect.origin.y * shortSide * scaleY - offsetY + rect.size.height * shortSide * scaleY)
                    rect.size.width *= longSide * scaleX
                    rect.size.height *= shortSide * scaleY

                    let label = String(format: "%@ %.1f%%", prediction.name, prediction.confidence * 100)
                    boundingBoxViews[i].show(frame: rect, label: label)
                } else {
                    boundingBoxViews[i].hide()
                }
            }
        }
    }

    let minimumZoom: CGFloat = 1.0
    let maximumZoom: CGFloat = 1.0
    var lastZoomFactor: CGFloat = 1.0

    @IBAction func pinch(_ pinch: UIPinchGestureRecognizer) {
        let device = videoCapture.captureDevice

        func minMaxZoom(_ factor: CGFloat) -> CGFloat {
            return min(min(max(factor, minimumZoom), maximumZoom), device.activeFormat.videoMaxZoomFactor)
        }

        func update(scale factor: CGFloat) {
            do {
                try device.lockForConfiguration()
                defer { device.unlockForConfiguration() }
                device.videoZoomFactor = factor
            } catch {
                print("\(error.localizedDescription)")
            }
        }

        let newScaleFactor = minMaxZoom(pinch.scale * lastZoomFactor)
        switch pinch.state {
        case .began, .changed:
            update(scale: newScaleFactor)
            self.labelZoom?.text = String(format: "%.2fx", newScaleFactor)
            self.labelZoom?.font = UIFont.preferredFont(forTextStyle: .title2)
        case .ended:
            lastZoomFactor = minMaxZoom(newScaleFactor)
            update(scale: lastZoomFactor)
            self.labelZoom?.font = UIFont.preferredFont(forTextStyle: .body)
        default:
            break
        }
    }

    func yoloToPixel(bbox: CGRect, imgWidth: CGFloat, imgHeight: CGFloat) -> CGRect {
        let xMin = bbox.origin.x * imgWidth
        let yMin = bbox.origin.y * imgHeight
        let width = bbox.width * imgWidth
        let height = bbox.height * imgHeight
        return CGRect(x: xMin, y: yMin, width: width, height: height)
    }

    func preprocessImage(_ image: CIImage, bbox: CGRect) -> CVPixelBuffer? {
        let croppedImage = image.cropped(to: bbox)

        let context = CIContext()
        let resizedImage = croppedImage.transformed(by: CGAffineTransform(scaleX: 224.0 / croppedImage.extent.width, y: 224.0 / croppedImage.extent.height))

        guard let cgImage = context.createCGImage(resizedImage, from: resizedImage.extent) else {
            print("Failed to create CGImage")
            return nil
        }

        let width = 224
        let height = 224
        
        // Create CVPixelBuffer attributes
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            print("Failed to create CVPixelBuffer")
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        
        // Create CGContext
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context2 = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            print("Failed to create CGContext")
            CVPixelBufferUnlockBaseAddress(buffer, [])
            return nil
        }
        
        // Set white background
        context2.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context2.fill(CGRect(x: 0, y: 0, width: width, height: height))
        
        // Draw image
        context2.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        CVPixelBufferUnlockBaseAddress(buffer, [])

        print("Successfully created preprocessed image buffer, size: \(width)x\(height)")
        return buffer
    }

    func extractFeatures(from pixelBuffer: CVPixelBuffer) -> [Float]? {
        print("Starting feature extraction...")
        
        guard let megaDescriptorModel = megaDescriptorModel else {
            print("❌ MegaDescriptor model not loaded")
            return nil
        }
        print("✅ MegaDescriptor model loaded")

        do {
            // Get model input description
            let modelDescription = megaDescriptorModel.modelDescription
            print("Model input description:")
            for input in modelDescription.inputDescriptionsByName {
                print("- Input name: \(input.key), type: \(input.value.type)")
            }
            
            // Get model output description
            print("Model output description:")
            for output in modelDescription.outputDescriptionsByName {
                print("- Output name: \(output.key), type: \(output.value.type)")
            }
            
            // Create MLFeatureProvider
            let inputFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
                "input": MLFeatureValue(pixelBuffer: pixelBuffer)
            ])
            
            // Execute prediction
            print("Starting model prediction...")
            let startTime = CACurrentMediaTime()
            let outputFeatures = try megaDescriptorModel.prediction(from: inputFeatureProvider)
            let endTime = CACurrentMediaTime()
            print("✅ Model prediction completed, time: \((endTime - startTime) * 1000)ms")
            
            // Try different output feature names
            let possibleOutputNames = ["output", "features", "embeddings", "descriptor"]
            var featureVector: [Float]?
            
            for name in possibleOutputNames {
                if let outputFeature = outputFeatures.featureValue(for: name) {
                    print("Found output feature: \(name)")
                    
                    switch outputFeature.type {
                    case .multiArray:
                        guard let multiArray = outputFeature.multiArrayValue else {
                            print("❌ Unable to get multi-array value")
                            continue
                        }
                        print("✅ Successfully got multi-array, shape: \(multiArray.shape)")
                        
                        // Convert to Float array
                        let count = multiArray.count
                        var vector = [Float](repeating: 0, count: count)
                        for i in 0..<count {
                            vector[i] = Float(multiArray[i].floatValue)
                        }
                        
                        print("✅ Successfully converted feature vector, length: \(vector.count)")
                        
                        // Check feature vector
                        if vector.count != 768 {
                            print("⚠️ Feature vector length mismatch: expected 768, got \(vector.count)")
                        }
                        
                        // Check if all zeros
                        let isAllZero = vector.allSatisfy { $0 == 0 }
                        if isAllZero {
                            print("❌ Feature vector is all zeros")
                            continue
                        }
                        
                        // Output statistics
                        let min = vector.min() ?? 0
                        let max = vector.max() ?? 0
                        let mean = vector.reduce(0, +) / Float(vector.count)
                        print("Feature vector statistics: min=\(min), max=\(max), mean=\(mean)")
                        
                        featureVector = vector
                        break
                        
                    case .dictionary:
                        print("❌ Output feature is dictionary type, expected multi-array")
                        continue
                        
                    default:
                        print("❌ Unsupported output feature type: \(outputFeature.type)")
                        continue
                    }
                }
            }
            
            if featureVector == nil {
                print("❌ Unable to get features from any possible output names")
                return nil
            }
            
            return featureVector
            
        } catch {
            print("❌ Feature extraction failed: \(error.localizedDescription)")
            print("Error details: \(error)")
            return nil
        }
    }

    func cosineSimilarity(_ vectorA: [Float], _ vectorB: [Float]) -> Float {
        guard vectorA.count == vectorB.count else {
            print("Feature vector length mismatch: vectorA length \(vectorA.count), vectorB length \(vectorB.count)")
            return 0.0
        }
        
        // Normalize vectors
        let normalizeVector = { (vector: [Float]) -> [Float] in
            let magnitude = sqrt(vector.map { $0 * $0 }.reduce(0, +))
            guard magnitude != 0 else { return vector }
            return vector.map { $0 / magnitude }
        }
        
        let normA = normalizeVector(vectorA)
        let normB = normalizeVector(vectorB)
        
        // Calculate cosine similarity
        let dotProduct = zip(normA, normB).map(*).reduce(0, +)
        
        // Add additional similarity checks
        let euclideanDistance = sqrt(zip(normA, normB).map { pow($0.0 - $0.1, 2) }.reduce(0, +))
        let euclideanSimilarity = 1.0 / (1.0 + euclideanDistance)
        
        // Combine cosine similarity and Euclidean distance similarity
        let combinedSimilarity = (dotProduct + euclideanSimilarity) / 2.0
        
        // Add additional validation steps
        let manhattanDistance = zip(normA, normB).map { abs($0.0 - $0.1) }.reduce(0, +)
        let manhattanSimilarity = 1.0 / (1.0 + manhattanDistance)
        
        // Final similarity is weighted average of three similarities
        return (combinedSimilarity * 0.5 + manhattanSimilarity * 0.5)
    }

    @objc func openSanctuaryWebsite() {
        selection.selectionChanged()
        if let url = URL(string: "https://www.thedonkeysanctuary.org.uk") {
            UIApplication.shared.open(url)
        }
    }

    private func setupHintLabelAndButton() {
        view.addSubview(hintLabel)
        view.addSubview(identifyButton)
        
        hintLabel.translatesAutoresizingMaskIntoConstraints = false
        identifyButton.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            hintLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            hintLabel.bottomAnchor.constraint(equalTo: toolBar.topAnchor, constant: -20),
            hintLabel.widthAnchor.constraint(lessThanOrEqualTo: view.widthAnchor, constant: -40),
            hintLabel.heightAnchor.constraint(equalToConstant: 40),
            
            identifyButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            identifyButton.bottomAnchor.constraint(equalTo: toolBar.topAnchor, constant: -20),
            identifyButton.widthAnchor.constraint(equalToConstant: 200),
            identifyButton.heightAnchor.constraint(equalToConstant: 40)
        ])
    }
    
    private func setupResultLabels() {
        view.addSubview(resultStackView)
        resultStackView.translatesAutoresizingMaskIntoConstraints = false
        
        // 設置約束
        NSLayoutConstraint.activate([
            resultStackView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            resultStackView.topAnchor.constraint(equalTo: labelName.bottomAnchor, constant: 20),
            resultStackView.widthAnchor.constraint(equalToConstant: 200)
        ])
        
        // 添加標籤到堆疊視圖
        resultLabels.forEach { resultStackView.addArrangedSubview($0) }
    }

    @objc private func identifyButtonTapped() {
        guard !isProcessing else { return }
        isProcessing = true
        continuousResults.removeAll()
        
        // 隱藏結果標籤
        resultStackView.isHidden = true
        
        // Show loading indicator
        activityIndicator.startAnimating()
        labelName.text = "Identifying..."
        
        // 開始識別過程
        processFrame()
    }
    
    private func processFrame() {
        guard let currentBuffer = currentBuffer,
              let bbox = currentDonkeyBoundingBox else {
            finishProcessing()
            return
        }
        
        // Process current frame
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            let startTime = CACurrentMediaTime()
            
            // Process frame once but perform multiple feature matches
            let ciImage = CIImage(cvPixelBuffer: currentBuffer)
            guard let croppedBuffer = self.preprocessImage(ciImage, bbox: bbox),
                  let feature = self.extractFeatures(from: croppedBuffer) else {
                self.finishProcessing()
                return
            }
            
            // Use parallel processing to speed up feature matching
            let group = DispatchGroup()
            let queue = DispatchQueue(label: "com.donkeyrecognition.featurematching", attributes: .concurrent)
            let chunkCount = 16
            let chunkSize = self.features.count / chunkCount
            var allMatches: [(name: String, score: Float)] = []
            let semaphore = DispatchSemaphore(value: 1)
            
            for chunk in 0..<chunkCount {
                group.enter()
                queue.async {
                    let start = chunk * chunkSize
                    let end = chunk == chunkCount - 1 ? self.features.count : (chunk + 1) * chunkSize
                    
                    var localMatches: [(name: String, score: Float)] = []
                    
                    for index in start..<end {
                        let score = self.cosineSimilarity(feature, self.features[index])
                        if score > 0.05 {
                            let donkeyName = String(self.labels[index].split(separator: "_").first ?? "Donkey")
                            localMatches.append((name: donkeyName, score: score))
                        }
                    }
                    
                    semaphore.wait()
                    allMatches.append(contentsOf: localMatches)
                    semaphore.signal()
                    
                    group.leave()
                }
            }
            
            group.wait()
            
            // Sort and group all matches
            var resultCounts: [String: (count: Int, maxScore: Float, totalScore: Float, minScore: Float)] = [:]
            for match in allMatches {
                if let existing = resultCounts[match.name] {
                    resultCounts[match.name] = (
                        count: existing.count + 1,
                        maxScore: max(existing.maxScore, match.score),
                        totalScore: existing.totalScore + match.score,
                        minScore: min(existing.minScore, match.score)
                    )
                } else {
                    resultCounts[match.name] = (count: 1, maxScore: match.score, totalScore: match.score, minScore: match.score)
                }
            }
            
            // Get top 3 results
            let topResults = resultCounts.sorted { a, b in
                let scoreA = (a.value.maxScore * 0.4 + a.value.totalScore / Float(a.value.count) * 0.4 + a.value.minScore * 0.2)
                let scoreB = (b.value.maxScore * 0.4 + b.value.totalScore / Float(b.value.count) * 0.4 + b.value.minScore * 0.2)
                return scoreA > scoreB
            }.prefix(3)
            
            let endTime = CACurrentMediaTime()
            print("\nTotal processing time: \((endTime - startTime) * 1000)ms")
            
            // Update UI
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.activityIndicator.stopAnimating()
                
                // Update top 3 results display
                self.resultStackView.isHidden = false
                for (index, result) in topResults.enumerated() {
                    let donkeyName = result.key
                    let maxScore = result.value.maxScore
                    let count = result.value.count
                    
                    // Calculate normalized confidence score (0-100%)
                    let normalizedScore = min(maxScore * 100, 100)
                    
                    self.resultLabels[index].text = String(format: "%d. %@ (%.1f%%)", 
                        index + 1, 
                        donkeyName, 
                        normalizedScore)
                }
                
                // Clear remaining labels
                for index in topResults.count..<self.resultLabels.count {
                    self.resultLabels[index].text = ""
                }
                
                self.isProcessing = false
                self.labelName.text = "Ask ELVIS"
            }
        }
    }
    
    private func finishProcessing() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.isProcessing = false
            self.activityIndicator.stopAnimating()
            self.labelName.text = "Ask ELVIS"
        }
    }
}

extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
        predict(sampleBuffer: sampleBuffer)
    }
}

extension ViewController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            print("error occurred : \(error.localizedDescription)")
            return
        }
        guard let dataImage = photo.fileDataRepresentation(),
              let dataProvider = CGDataProvider(data: dataImage as CFData),
              let cgImageRef = CGImage(
                  jpegDataProviderSource: dataProvider,
                  decode: nil,
                  shouldInterpolate: true,
                  intent: .defaultIntent
              ) else {
            print("無法創建 CGImage")
            return
        }

        var isCameraFront = false
        if let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput,
           currentInput.device.position == .front {
            isCameraFront = true
        }
        var orientation: CGImagePropertyOrientation = isCameraFront ? .leftMirrored : .right
        switch UIDevice.current.orientation {
        case .landscapeLeft:
            orientation = isCameraFront ? .downMirrored : .up
        case .landscapeRight:
            orientation = isCameraFront ? .upMirrored : .down
        default:
            break
        }
        var image = UIImage(cgImage: cgImageRef, scale: 0.5, orientation: .right)
        if let orientedCIImage = CIImage(image: image)?.oriented(orientation),
           let cgImage = CIContext().createCGImage(orientedCIImage, from: orientedCIImage.extent) {
            image = UIImage(cgImage: cgImage)
        }
        let imageView = UIImageView(image: image)
        imageView.contentMode = .scaleAspectFill
        imageView.frame = videoPreview?.frame ?? CGRect.zero
        let imageLayer = imageView.layer
        videoPreview?.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)

        let bounds = UIScreen.main.bounds
        UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
        self.View0?.drawHierarchy(in: bounds, afterScreenUpdates: true)
        guard let img = UIGraphicsGetImageFromCurrentImageContext() else {
            print("無法生成分享圖片")
            UIGraphicsEndImageContext()
            return
        }
        UIGraphicsEndImageContext()
        imageLayer.removeFromSuperlayer()
        let activityViewController = UIActivityViewController(
            activityItems: [img], applicationActivities: nil)
        activityViewController.popoverPresentationController?.sourceView = self.View0
        self.present(activityViewController, animated: true, completion: nil)
    }
}
