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
    private var isProcessing = false

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
            
            print("\n=== 開始處理觀察結果 ===")
            
            if let error = error {
                print("❌ 請求錯誤：\(error.localizedDescription)")
            }
            
            if let results = request.results as? [VNRecognizedObjectObservation] {
                print("✅ 成功獲取 YOLO 檢測結果，數量：\(results.count)")
                
                let donkeyResults = results.filter { observation in
                    observation.labels.contains { $0.identifier == self.donkeyClassLabel }
                }
                print("找到驢子數量：\(donkeyResults.count)")

                // 如果正在處理中，直接返回
                if self.isProcessing {
                    print("⚠️ 正在處理中，跳過此幀")
                    return
                }
                
                // 保存當前緩衝區的副本
                guard let currentBuffer = self.currentBuffer else {
                    print("❌ 當前緩衝區為空")
                    self.show(predictions: donkeyResults.map { DetectionResult(boundingBox: $0.boundingBox, name: "Donkey", confidence: $0.confidence) })
                    return
                }
                
                // 創建緩衝區的副本
                var copiedBuffer: CVPixelBuffer?
                let status = CVPixelBufferCreate(
                    kCFAllocatorDefault,
                    CVPixelBufferGetWidth(currentBuffer),
                    CVPixelBufferGetHeight(currentBuffer),
                    CVPixelBufferGetPixelFormatType(currentBuffer),
                    nil,
                    &copiedBuffer
                )
                
                guard status == kCVReturnSuccess, let copiedBuffer = copiedBuffer else {
                    print("❌ 無法創建緩衝區副本")
                    self.show(predictions: donkeyResults.map { DetectionResult(boundingBox: $0.boundingBox, name: "Donkey", confidence: $0.confidence) })
                    return
                }
                
                CVPixelBufferLockBaseAddress(currentBuffer, .readOnly)
                CVPixelBufferLockBaseAddress(copiedBuffer, [])
                
                let bytesPerRow = CVPixelBufferGetBytesPerRow(currentBuffer)
                let height = CVPixelBufferGetHeight(currentBuffer)
                
                if let sourceData = CVPixelBufferGetBaseAddress(currentBuffer),
                   let destData = CVPixelBufferGetBaseAddress(copiedBuffer) {
                    memcpy(destData, sourceData, bytesPerRow * height)
                }
                
                CVPixelBufferUnlockBaseAddress(currentBuffer, .readOnly)
                CVPixelBufferUnlockBaseAddress(copiedBuffer, [])
                
                print("✅ 成功創建緩衝區副本")
                
                let ciImage = CIImage(cvPixelBuffer: copiedBuffer)
                print("✅ 成功創建 CIImage")

                // 設置處理標誌
                self.isProcessing = true
                
                // 在後台線程處理特徵提取
                DispatchQueue.global(qos: .userInitiated).async {
                    var updatedResults: [DetectionResult] = []
                    
                    for (index, observation) in donkeyResults.enumerated() {
                        print("\n處理第 \(index + 1) 個驢子檢測結果")
                        
                        let imgWidth = CGFloat(CVPixelBufferGetWidth(copiedBuffer))
                        let imgHeight = CGFloat(CVPixelBufferGetHeight(copiedBuffer))
                        let bbox = self.yoloToPixel(bbox: observation.boundingBox, imgWidth: imgWidth, imgHeight: imgHeight)
                        print("圖像尺寸：\(imgWidth)x\(imgHeight)")
                        print("邊界框：\(bbox)")

                        // 檢查特徵庫是否已載入
                        guard !self.features.isEmpty, !self.labels.isEmpty else {
                            print("❌ 特徵庫尚未載入")
                            updatedResults.append(DetectionResult(boundingBox: observation.boundingBox, name: "Donkey", confidence: observation.confidence))
                            continue
                        }
                        print("✅ 特徵庫已載入，特徵數量：\(self.features.count)，標籤數量：\(self.labels.count)")

                        // 預處理圖像並提取特徵
                        guard let croppedBuffer = self.preprocessImage(ciImage, bbox: bbox) else {
                            print("❌ 圖像預處理失敗")
                            updatedResults.append(DetectionResult(boundingBox: observation.boundingBox, name: "Donkey", confidence: observation.confidence))
                            continue
                        }
                        print("✅ 圖像預處理成功")

                        // 生成緩存鍵
                        let cacheKey = "\(bbox.origin.x)_\(bbox.origin.y)_\(bbox.width)_\(bbox.height)"
                        
                        // 檢查緩存
                        var queryFeature: [Float]?
                        self.cacheQueue.sync {
                            queryFeature = self.featureCache[cacheKey]
                        }
                        
                        if queryFeature == nil {
                            queryFeature = self.extractFeatures(from: croppedBuffer)
                            if let feature = queryFeature {
                                self.cacheQueue.async {
                                    self.featureCache[cacheKey] = feature
                                    // 限制緩存大小
                                    if self.featureCache.count > 100 {
                                        self.featureCache.removeValue(forKey: self.featureCache.keys.first!)
                                    }
                                }
                            }
                        } else {
                            print("✅ 使用緩存的特徵")
                        }

                        guard let feature = queryFeature else {
                            print("❌ 特徵提取失敗")
                            updatedResults.append(DetectionResult(boundingBox: observation.boundingBox, name: "Donkey", confidence: observation.confidence))
                            continue
                        }
                        print("✅ 特徵提取成功，特徵向量長度：\(feature.count)")

                        // 特徵匹配
                        var bestScore: Float = -1.0
                        var bestIndex = 0
                        for (index, storedFeature) in self.features.enumerated() {
                            let score = self.cosineSimilarity(feature, storedFeature)
                            if score > bestScore {
                                bestScore = score
                                bestIndex = index
                            }
                        }
                        print("最佳匹配分數：\(bestScore)")
                        print("最佳匹配索引：\(bestIndex)")

                        // 使用特徵匹配的結果
                        let donkeyName = self.labels[bestIndex].split(separator: "_").first ?? "Donkey"
                        updatedResults.append(DetectionResult(boundingBox: observation.boundingBox, name: String(donkeyName), confidence: observation.confidence))
                        print("✅ 檢測到驢子：\(donkeyName), 置信度：\(observation.confidence * 100)%, 相似度：\(bestScore)")
                    }

                    // 在主線程更新 UI
                    DispatchQueue.main.async {
                        // 只在有結果時更新顯示
                        if !updatedResults.isEmpty {
                            self.show(predictions: updatedResults)
                        }
                        print("=== 處理完成 ===\n")
                        
                        // 重置處理標誌
                        self.isProcessing = false
                    }
                }
            } else {
                print("❌ 無法獲取 YOLO 檢測結果")
                // 只在有結果時更新顯示
                if let results = request.results as? [VNRecognizedObjectObservation], !results.isEmpty {
                    self.show(predictions: results.map { DetectionResult(boundingBox: $0.boundingBox, name: "Donkey", confidence: $0.confidence) })
                }
            }

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
        
        // 創建 CVPixelBuffer 的屬性
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
        
        // 創建 CGContext
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
        
        // 設置背景為白色
        context2.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context2.fill(CGRect(x: 0, y: 0, width: width, height: height))
        
        // 繪製圖像
        context2.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        CVPixelBufferUnlockBaseAddress(buffer, [])

        print("Successfully created preprocessed image buffer, size: \(width)x\(height)")
        return buffer
    }

    func extractFeatures(from pixelBuffer: CVPixelBuffer) -> [Float]? {
        print("開始特徵提取...")
        
        guard let megaDescriptorModel = megaDescriptorModel else {
            print("❌ MegaDescriptor 模型未載入")
            return nil
        }
        print("✅ MegaDescriptor 模型已載入")

        do {
            // 獲取模型的輸入描述
            let modelDescription = megaDescriptorModel.modelDescription
            print("模型輸入描述：")
            for input in modelDescription.inputDescriptionsByName {
                print("- 輸入名稱：\(input.key), 類型：\(input.value.type)")
            }
            
            // 獲取模型的輸出描述
            print("模型輸出描述：")
            for output in modelDescription.outputDescriptionsByName {
                print("- 輸出名稱：\(output.key), 類型：\(output.value.type)")
            }
            
            // 創建 MLFeatureProvider
            let inputFeatureProvider = try MLDictionaryFeatureProvider(dictionary: [
                "input": MLFeatureValue(pixelBuffer: pixelBuffer)
            ])
            
            // 執行預測
            print("開始執行模型預測...")
            let startTime = CACurrentMediaTime()
            let outputFeatures = try megaDescriptorModel.prediction(from: inputFeatureProvider)
            let endTime = CACurrentMediaTime()
            print("✅ 模型預測完成，耗時：\((endTime - startTime) * 1000)ms")
            
            // 嘗試不同的輸出特徵名稱
            let possibleOutputNames = ["output", "features", "embeddings", "descriptor"]
            var featureVector: [Float]?
            
            for name in possibleOutputNames {
                if let outputFeature = outputFeatures.featureValue(for: name) {
                    print("找到輸出特徵：\(name)")
                    
                    switch outputFeature.type {
                    case .multiArray:
                        guard let multiArray = outputFeature.multiArrayValue else {
                            print("❌ 無法獲取多維數組值")
                            continue
                        }
                        print("✅ 成功獲取多維數組，形狀：\(multiArray.shape)")
                        
                        // 轉換為 Float 數組
                        let count = multiArray.count
                        var vector = [Float](repeating: 0, count: count)
                        for i in 0..<count {
                            vector[i] = Float(multiArray[i].floatValue)
                        }
                        
                        print("✅ 成功轉換特徵向量，長度：\(vector.count)")
                        
                        // 檢查特徵向量
                        if vector.count != 768 {
                            print("⚠️ 特徵向量長度不匹配：期望 768，實際 \(vector.count)")
                        }
                        
                        // 檢查是否全為零
                        let isAllZero = vector.allSatisfy { $0 == 0 }
                        if isAllZero {
                            print("❌ 特徵向量全為零")
                            continue
                        }
                        
                        // 輸出統計信息
                        let min = vector.min() ?? 0
                        let max = vector.max() ?? 0
                        let mean = vector.reduce(0, +) / Float(vector.count)
                        print("特徵向量統計：最小值=\(min), 最大值=\(max), 平均值=\(mean)")
                        
                        featureVector = vector
                        break
                        
                    case .dictionary:
                        print("❌ 輸出特徵是字典類型，期望多維數組")
                        continue
                        
                    default:
                        print("❌ 不支持的輸出特徵類型：\(outputFeature.type)")
                        continue
                    }
                }
            }
            
            if featureVector == nil {
                print("❌ 無法從任何可能的輸出名稱獲取特徵")
                return nil
            }
            
            return featureVector
            
        } catch {
            print("❌ 特徵提取失敗：\(error.localizedDescription)")
            print("錯誤詳情：\(error)")
            return nil
        }
    }

    func cosineSimilarity(_ vectorA: [Float], _ vectorB: [Float]) -> Float {
        guard vectorA.count == vectorB.count else {
            print("特徵向量長度不匹配：vectorA 長度 \(vectorA.count)，vectorB 長度 \(vectorB.count)")
            return 0.0
        }
        let dotProduct = zip(vectorA, vectorB).map(*).reduce(0, +)
        let magnitudeA = sqrt(vectorA.map { $0 * $0 }.reduce(0, +))
        let magnitudeB = sqrt(vectorB.map { $0 * $0 }.reduce(0, +))
        guard magnitudeA != 0, magnitudeB != 0 else {
            print("特徵向量幅度為 0：magnitudeA \(magnitudeA)，magnitudeB \(magnitudeB)")
            return 0.0
        }
        return dotProduct / (magnitudeA * magnitudeB)
    }

    @objc func openSanctuaryWebsite() {
        selection.selectionChanged()
        if let url = URL(string: "https://www.thedonkeysanctuary.org.uk") {
            UIApplication.shared.open(url)
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
