import AVFoundation
import CoreML
import CoreMedia
import UIKit
import Vision
import CoreImage
import SafariServices  // 添加 SafariServices 框架

// YOLO model
var mlModel: MLModel = {
    guard let path = Bundle.main.path(forResource: "best", ofType: "mlmodelc") else {
        // List Bundle contents for debugging
        let fileManager = FileManager.default
        if let bundlePath = Bundle.main.bundlePath as NSString? {
            do {
                let contents = try fileManager.contentsOfDirectory(atPath: bundlePath as String)
                print("Bundle contents: \(contents)")
                let modelsPath = bundlePath.appendingPathComponent("Models")
                if fileManager.fileExists(atPath: modelsPath) {
                    let modelsContents = try fileManager.contentsOfDirectory(atPath: modelsPath)
                    print("Models/ folder contents: \(modelsContents)")
                }
            } catch {
                print("Unable to list Bundle contents: \(error)")
            }
        }
        fatalError("Cannot find best.mlmodelc file, please ensure it is correctly added to the project")
    }
    let modelURL = URL(fileURLWithPath: path)
    do {
        return try MLModel(contentsOf: modelURL, configuration: mlmodelConfig)
    } catch {
        fatalError("Unable to load best.mlmodelc model: \(error.localizedDescription)")
    }
}()

var mlmodelConfig: MLModelConfiguration = {
    let config = MLModelConfiguration()
    if #available(iOS 17.0, *) {
        config.setValue(1, forKey: "experimentalMLE5EngineUsage")
    }
    return config
}()

// MegaDescriptor model
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
        print("Unable to load MegaDescriptor model: \(error.localizedDescription)")
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
    @IBOutlet weak var labelZoom: UILabel!
    @IBOutlet weak var labelVersion: UILabel!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var focus: UIImageView!
    @IBOutlet weak var toolBar: UIToolbar!
    @IBOutlet weak var identifyButton: UIButton!

    // Add status label
    private lazy var statusLabel: UILabel = {
        let label = UILabel()
        label.textColor = .white
        label.textAlignment = .center
        label.font = .systemFont(ofSize: 15)  // 調整為標準正文大小
        label.numberOfLines = 2
        label.text = "Loading model..."
        return label
    }()

    // Add processing time label
    private lazy var processingTimeLabel: UILabel = {
        let label = UILabel()
        label.textColor = .white
        label.textAlignment = .center
        label.font = .systemFont(ofSize: 12)
        label.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        label.layer.cornerRadius = 4
        label.clipsToBounds = true
        label.text = "Processing: 0ms"
        return label
    }()

    // New UI elements
    private lazy var resultStackView: UIStackView = {
        let stack = UIStackView()
        stack.axis = .vertical
        stack.spacing = 8  // 調整為標準間距
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
            label.font = .systemFont(ofSize: 15)  // 調整為標準正文大小
            label.backgroundColor = UIColor.black.withAlphaComponent(0.6)
            label.layer.cornerRadius = 8
            label.clipsToBounds = true
            return label
        }
    }()

    // New properties
    private var continuousResults: [(name: String, score: Float, confidence: Float)] = []
    private let requiredResults = 2  // Only need 2 results
    private var isProcessing = false
    private var hasDetectedDonkey = false
    private var currentDonkeyBoundingBox: CGRect?
    private var currentDonkeyName: String?  // Store the currently recognized donkey name
    
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

    // Add feature cache
    private var featureCache: [String: [Float]] = [:]
    private let cacheQueue = DispatchQueue(label: "com.donkeyrecognition.featurecache")

    // 添加 hintContainer 屬性
    private var hintContainer: UIView?

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
        
        // 設置 identifyButton
        setupIdentifyButton()
        
        // 確保 hintContainer 初始顯示
        self.hintContainer?.isHidden = false
        self.identifyButton.isHidden = true
        
        // Set toolbar height and style
        toolBar.frame.size.height = 49  // 標準工具列高度
        toolBar.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        
        // Add status label to toolbar
        let statusItem = UIBarButtonItem(customView: statusLabel)
        statusLabel.frame = CGRect(x: 0, y: 0, width: 200, height: 44)  // 標準按鈕高度
        
        // 新增驢子保護區網站按鈕到最右側
        let sanctuaryButton = UIBarButtonItem(image: UIImage(systemName: "globe"), style: .plain, target: self, action: #selector(openSanctuaryWebsite))
        let donkeyInfoButton = UIBarButtonItem(image: UIImage(systemName: "magnifyingglass"), style: .plain, target: self, action: #selector(openDonkeyInfo))
        let saveButton = UIBarButtonItem(image: UIImage(systemName: "square.and.arrow.down"), style: .plain, target: self, action: #selector(saveToPhotoLibrary))
        
        // 設置按鈕顏色和大小
        let buttonColor = UIColor.white
        sanctuaryButton.tintColor = buttonColor
        donkeyInfoButton.tintColor = buttonColor
        saveButton.tintColor = buttonColor
        
        // 設置按鈕大小
        let buttonSize = CGSize(width: 44, height: 44)  // 最小點擊區域
        sanctuaryButton.customView?.frame.size = buttonSize
        donkeyInfoButton.customView?.frame.size = buttonSize
        saveButton.customView?.frame.size = buttonSize
        
        // 添加彈性空間和按鈕到工具列
        toolBar.items?.append(UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil))
        toolBar.items?.append(statusItem)
        toolBar.items?.append(UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil))
        toolBar.items?.append(saveButton)
        
        // 添加固定間距
        let spacing = UIBarButtonItem(barButtonSystemItem: .fixedSpace, target: nil, action: nil)
        spacing.width = 8
        toolBar.items?.append(spacing)
        
        toolBar.items?.append(sanctuaryButton)
        
        // 添加固定間距
        let spacing2 = UIBarButtonItem(barButtonSystemItem: .fixedSpace, target: nil, action: nil)
        spacing2.width = 8
        toolBar.items?.append(spacing2)
        
        toolBar.items?.append(donkeyInfoButton)
        
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
                        
                        // 更新狀態標籤
                        DispatchQueue.main.async {
                            self.statusLabel.text = "Model: ✅\nFeatures: \(firstFeature.count) (\(self.features.count))"
                        }
                    }
                } catch {
                    print("無法載入 features_and_labels.json 檔案，錯誤：\(error.localizedDescription)")
                    DispatchQueue.main.async {
                        self.statusLabel.text = "Model: ❌\nFeatures: Error"
                    }
                }
            } else {
                print("無法找到 features_and_labels.json 檔案，請確認檔案已正確添加到專案")
                DispatchQueue.main.async {
                    self.statusLabel.text = "Model: ❌\nFeatures: Not found"
                }
            }
        }

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
                self.updateStatusLabel(frameSize: "720.0x1280.0")
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
                // 過濾出信心度大於 40% 的驢子檢測結果
                let donkeyResults = results.filter { observation in
                    observation.labels.contains { label in
                        label.identifier == self.donkeyClassLabel && label.confidence > 0.4
                    }
                }
                
                // 更新 UI 狀態
                self.hasDetectedDonkey = !donkeyResults.isEmpty
                
                // 更新 hintLabel 和 identifyButton 的顯示狀態
                self.hintContainer?.isHidden = self.hasDetectedDonkey
                self.identifyButton.isHidden = !self.hasDetectedDonkey
                
                if !donkeyResults.isEmpty {
                    // 轉換所有檢測到的驢子為 DetectionResult
                    let predictions = donkeyResults.map { donkey in
                        DetectionResult(
                            boundingBox: donkey.boundingBox,
                            name: "Donkey",
                            confidence: donkey.confidence
                        )
                    }
                    self.show(predictions: predictions)
                    
                    // 保存第一個驢子的邊界框用於特徵提取
                    self.currentDonkeyBoundingBox = donkeyResults.first?.boundingBox
                } else {
                    self.currentDonkeyBoundingBox = nil
                    self.show(predictions: [])
                }
            } else {
                self.hasDetectedDonkey = false
                
                // 更新 hintLabel 和 identifyButton 的顯示狀態
                self.hintContainer?.isHidden = false
                self.identifyButton.isHidden = true
                
                self.currentDonkeyBoundingBox = nil
                self.show(predictions: [])
            }
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
                    
                    // Check box size
                    let boxArea = rect.width * rect.height
                    let screenArea = width * height
                    let areaRatio = boxArea / screenArea
                    
                    // Set box color and display logic
                    if areaRatio > 0.8 {  // Box too large, likely false detection
                        boundingBoxViews[i].hide()
                        self.hasDetectedDonkey = false
                        self.identifyButton.isHidden = true
                        self.hintContainer?.isHidden = false
                    } else if areaRatio < 0.05 {  // Box too small, too far away
                        let label = String(format: "%@ %.1f%%", prediction.name, prediction.confidence * 100)
                        boundingBoxViews[i].show(frame: rect, label: label, color: UIColor.red)
                        self.hasDetectedDonkey = false
                        self.identifyButton.isHidden = true
                        self.hintContainer?.isHidden = false
                    } else {  // Normal distance
                        let label = String(format: "%@ %.1f%%", prediction.name, prediction.confidence * 100)
                        boundingBoxViews[i].show(frame: rect, label: label, color: UIColor.green)
                        self.hasDetectedDonkey = true
                        self.identifyButton.isHidden = false
                        self.hintContainer?.isHidden = true
                    }
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
                    
                    // Check box size
                    let boxArea = rect.width * rect.height
                    let screenArea = width * height
                    let areaRatio = boxArea / screenArea
                    
                    // Set box color and display logic
                    if areaRatio > 0.8 {  // Box too large, likely false detection
                        boundingBoxViews[i].hide()
                        self.hasDetectedDonkey = false
                        self.identifyButton.isHidden = true
                    } else if areaRatio < 0.05 {  // Box too small, too far away
                        let label = String(format: "%@ %.1f%%", prediction.name, prediction.confidence * 100)
                        boundingBoxViews[i].show(frame: rect, label: label, color: UIColor.red)
                    } else {  // Normal distance
                        let label = String(format: "%@ %.1f%%", prediction.name, prediction.confidence * 100)
                        boundingBoxViews[i].show(frame: rect, label: label, color: UIColor.green)
                    }
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
        if let url = URL(string: "https://www.iowdonkeysanctuary.org") {
            // 使用 SFSafariViewController 開啟網頁
            let safariVC = SFSafariViewController(url: url)
            safariVC.preferredControlTintColor = .black
            safariVC.modalPresentationStyle = .pageSheet
            present(safariVC, animated: true)
        }
    }

    @objc func openDonkeyInfo() {
        selection.selectionChanged()
        guard let donkeyName = currentDonkeyName?.lowercased() else {
            // 如果沒有識別出驢子，顯示提示
            let alert = UIAlertController(title: "Notice", message: "Please identify a donkey first", preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            present(alert, animated: true)
            return
        }
        
        // 根據驢子名稱構建URL
        let baseURL = "https://www.iowdonkeysanctuary.org/product/"
        if let url = URL(string: baseURL + donkeyName) {
            // 使用 SFSafariViewController 開啟網頁
            let safariVC = SFSafariViewController(url: url)
            safariVC.preferredControlTintColor = .black
            safariVC.modalPresentationStyle = .pageSheet
            present(safariVC, animated: true)
        }
    }

    @objc private func saveToPhotoLibrary() {
        guard hasDetectedDonkey else {
            let alert = UIAlertController(title: "Notice", message: "Please identify a donkey first", preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            present(alert, animated: true)
            return
        }
        
        // 拍攝當前畫面
        let settings = AVCapturePhotoSettings()
        self.videoCapture.cameraOutput.capturePhoto(with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
    }
    
    @objc private func image(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        if let error = error {
            let alert = UIAlertController(title: "Save Failed", message: error.localizedDescription, preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            present(alert, animated: true)
        } else {
            let alert = UIAlertController(title: "Save Successful", message: "Recognition results have been saved to photo library", preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            present(alert, animated: true)
        }
    }

    private func setupResultLabels() {
        view.addSubview(resultStackView)
        resultStackView.translatesAutoresizingMaskIntoConstraints = false
        
        // 設置約束
        NSLayoutConstraint.activate([
            resultStackView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            resultStackView.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -view.bounds.height / 3),  // 修改為螢幕下方 1/3
            resultStackView.widthAnchor.constraint(equalToConstant: 200)
        ])
        
        // 添加標籤到堆疊視圖
        resultLabels.forEach { resultStackView.addArrangedSubview($0) }
    }

    @IBAction func identifyButtonTapped(_ sender: Any) {
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
        
        // 保存當前照片
        let ciImage = CIImage(cvPixelBuffer: currentBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            finishProcessing()
            return
        }
        let capturedImage = UIImage(cgImage: cgImage)
        
        // Process current frame
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            let startTime = CACurrentMediaTime()
            
            // Process frame once but perform multiple feature matches
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
            let processingTime = (endTime - startTime) * 1000
            
            // Update status label with processing time
            self.updateStatusLabel(processingTime: processingTime)
            
            // Update UI
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.activityIndicator.stopAnimating()
                
                // Update top 3 results display
                self.resultStackView.isHidden = false
                for (index, result) in topResults.enumerated() {
                    let donkeyName = result.key
                    let maxScore = result.value.maxScore
                    
                    // Calculate normalized confidence score (0-100%)
                    let normalizedScore = min(maxScore * 100, 100)
                    
                    self.resultLabels[index].text = String(format: "%d. %@ (%.1f%%)", 
                        index + 1, 
                        donkeyName, 
                        normalizedScore)
                    
                    // 更新當前識別出的驢子名稱（取最高分的那個）
                    if index == 0 {
                        self.currentDonkeyName = donkeyName
                    }
                }
                
                // Clear remaining labels
                for index in topResults.count..<self.resultLabels.count {
                    self.resultLabels[index].text = ""
                }
                
                // 添加兩次快速震動的觸覺反饋
                let generator = UIImpactFeedbackGenerator(style: .medium)
                generator.prepare()  // 預先準備震動器以減少延遲
                
                // 第一次震動
                generator.impactOccurred()
                
                // 延遲 0.1 秒後進行第二次震動
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    generator.impactOccurred()
                }
                
                // 保存結果和照片
                self.saveResultAndPhoto(capturedImage: capturedImage, results: Array(topResults))
                
                // 10秒後自動清除結果
                DispatchQueue.main.asyncAfter(deadline: .now() + 10.0) { [weak self] in
                    guard let self = self else { return }
                    self.resultStackView.isHidden = true
                    for label in self.resultLabels {
                        label.text = ""
                    }
                    self.currentDonkeyName = nil
                }
                
                self.isProcessing = false
                self.labelName.text = "Ask ELVIS"
            }
        }
    }
    
    private func saveResultAndPhoto(capturedImage: UIImage, results: [(key: String, value: (count: Int, maxScore: Float, totalScore: Float, minScore: Float))]) {
        // 創建一個新的圖像上下文
        let renderer = UIGraphicsImageRenderer(size: capturedImage.size)
        
        let finalImage = renderer.image { context in
            // 繪製原始照片
            capturedImage.draw(in: CGRect(origin: .zero, size: capturedImage.size))
            
            // 設置文字屬性
            let paragraphStyle = NSMutableParagraphStyle()
            paragraphStyle.alignment = .left
            
            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 24, weight: .bold),
                .foregroundColor: UIColor.white,
                .paragraphStyle: paragraphStyle
            ]
            
            // 繪製結果文字
            var yOffset: CGFloat = 50
            for (index, result) in results.enumerated() {
                let donkeyName = result.key
                let maxScore = result.value.maxScore
                let normalizedScore = min(maxScore * 100, 100)
                
                let text = String(format: "%d. %@ (%.1f%%)", index + 1, donkeyName, normalizedScore)
                let textRect = CGRect(x: 20, y: yOffset, width: capturedImage.size.width - 40, height: 30)
                
                // 添加文字背景
                let backgroundRect = CGRect(x: textRect.minX - 10, y: textRect.minY - 5,
                                          width: textRect.width + 20, height: textRect.height + 10)
                context.cgContext.setFillColor(UIColor.black.withAlphaComponent(0.6).cgColor)
                context.cgContext.fill(backgroundRect)
                
                // 繪製文字
                text.draw(in: textRect, withAttributes: attributes)
                yOffset += 40
            }
        }
        
        // 保存到相簿
        UIImageWriteToSavedPhotosAlbum(finalImage, self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)
    }
    
    private func finishProcessing() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.isProcessing = false
            self.activityIndicator.stopAnimating()
            self.labelName.text = "Ask ELVIS"
        }
    }

    private func updateStatusLabel(frameSize: String? = nil, processingTime: Double? = nil) {
        var statusText = ""
        
        // Add frame size if available
        if let size = frameSize {
            statusText += "Frame: \(size)\n"
        }
        
        // Add processing time if available
        if let time = processingTime {
            statusText += "Processing: \(Int(time))ms"
        }
        
        // Update label on main thread
        DispatchQueue.main.async {
            self.statusLabel.text = statusText
        }
    }

    private func setupIdentifyButton() {
        // 設置按鈕的玻璃效果
        let buttonBlurEffect = UIBlurEffect(style: .systemUltraThinMaterial)
        let buttonBlurView = UIVisualEffectView(effect: buttonBlurEffect)
        buttonBlurView.frame = identifyButton.bounds
        buttonBlurView.isUserInteractionEnabled = false
        buttonBlurView.alpha = 0.9  // 按鈕背景更不透明
        buttonBlurView.layer.cornerRadius = 16
        buttonBlurView.clipsToBounds = true
        identifyButton.insertSubview(buttonBlurView, at: 0)
        
        // 設置按鈕樣式
        identifyButton.backgroundColor = .clear
        identifyButton.layer.cornerRadius = 16
        identifyButton.layer.masksToBounds = true
        
        // 設置按鈕文字顏色和字體大小
        identifyButton.setTitleColor(.black, for: .normal)
        identifyButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .semibold)  // 按鈕文字加粗
        identifyButton.titleLabel?.textAlignment = .center
        
        // 添加邊框
        let borderLayer = CALayer()
        borderLayer.frame = identifyButton.bounds
        borderLayer.cornerRadius = 16
        borderLayer.borderWidth = 2.0  // 按鈕邊框更粗
        borderLayer.borderColor = UIColor(white: 1.0, alpha: 0.5).cgColor  // 按鈕邊框更明顯
        identifyButton.layer.addSublayer(borderLayer)
        
        // 添加陰影
        identifyButton.layer.shadowColor = UIColor.black.cgColor
        identifyButton.layer.shadowOffset = CGSize(width: 0, height: 8)  // 按鈕陰影更大
        identifyButton.layer.shadowRadius = 12
        identifyButton.layer.shadowOpacity = 0.2  // 按鈕陰影更明顯
        
        // 設置提示文字的容器視圖
        let hintContainer = UIView()
        hintContainer.translatesAutoresizingMaskIntoConstraints = false
        hintContainer.backgroundColor = UIColor(white: 0.9, alpha: 0.15)  // 提示框背景更透明
        hintContainer.layer.cornerRadius = 16
        hintContainer.layer.masksToBounds = true
        
        // 添加毛玻璃效果
        let hintBlurEffect = UIBlurEffect(style: .systemUltraThinMaterial)
        let hintBlurView = UIVisualEffectView(effect: hintBlurEffect)
        hintBlurView.frame = hintContainer.bounds
        hintBlurView.isUserInteractionEnabled = false
        hintBlurView.alpha = 0.6  // 提示框背景更透明
        hintBlurView.layer.cornerRadius = 16
        hintBlurView.clipsToBounds = true
        hintContainer.insertSubview(hintBlurView, at: 0)
        
        // 添加邊框
        let hintBorderLayer = CALayer()
        hintBorderLayer.frame = hintContainer.bounds
        hintBorderLayer.cornerRadius = 16
        hintBorderLayer.borderWidth = 1.0  // 提示框邊框更細
        hintBorderLayer.borderColor = UIColor(white: 1.0, alpha: 0.2).cgColor  // 提示框邊框更淡
        hintContainer.layer.addSublayer(hintBorderLayer)
        
        // 設置提示文字
        let hintLabel = UILabel()
        hintLabel.text = "Point camera at donkey"
        hintLabel.textColor = .black
        hintLabel.textAlignment = .center
        hintLabel.font = .systemFont(ofSize: 18, weight: .regular)  // 提示文字不加粗
        hintLabel.alpha = 0.8  // 提示文字更透明
        hintLabel.translatesAutoresizingMaskIntoConstraints = false
        hintLabel.numberOfLines = 1
        
        // 將 hintLabel 添加到容器中
        hintContainer.addSubview(hintLabel)
        
        // 將容器添加到主視圖
        view.addSubview(hintContainer)
        
        // 設置約束
        NSLayoutConstraint.activate([
            // identifyButton 約束
            identifyButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            identifyButton.bottomAnchor.constraint(equalTo: labelVersion.topAnchor, constant: -20),
            identifyButton.widthAnchor.constraint(equalToConstant: 300),
            identifyButton.heightAnchor.constraint(equalToConstant: 60),
            
            // 容器約束
            hintContainer.centerXAnchor.constraint(equalTo: identifyButton.centerXAnchor),
            hintContainer.bottomAnchor.constraint(equalTo: identifyButton.bottomAnchor),
            hintContainer.widthAnchor.constraint(equalToConstant: 300),
            hintContainer.heightAnchor.constraint(equalToConstant: 55),
            
            // 提示文字約束
            hintLabel.centerXAnchor.constraint(equalTo: hintContainer.centerXAnchor),
            hintLabel.centerYAnchor.constraint(equalTo: hintContainer.centerYAnchor),
            hintLabel.leadingAnchor.constraint(equalTo: hintContainer.leadingAnchor, constant: 24),
            hintLabel.trailingAnchor.constraint(equalTo: hintContainer.trailingAnchor, constant: -24)
        ])
        
        // 添加容器的陰影
        hintContainer.layer.shadowColor = UIColor.black.cgColor
        hintContainer.layer.shadowOffset = CGSize(width: 0, height: 4)  // 提示框陰影更小
        hintContainer.layer.shadowRadius = 8
        hintContainer.layer.shadowOpacity = 0.1  // 提示框陰影更淡
        
        // 初始狀態：hintLabel 顯示，identifyButton 隱藏
        hintContainer.isHidden = false
        identifyButton.isHidden = true
        
        // 保存 hintContainer 的引用
        self.hintContainer = hintContainer
        
        // 添加點擊動畫
        identifyButton.addTarget(self, action: #selector(buttonTouchDown), for: .touchDown)
        identifyButton.addTarget(self, action: #selector(buttonTouchUp), for: [.touchUpInside, .touchUpOutside, .touchCancel])
        
        // 確保按鈕文字在按鈕內
        identifyButton.contentEdgeInsets = UIEdgeInsets(top: 0, left: 20, bottom: 0, right: 20)
        identifyButton.titleLabel?.adjustsFontSizeToFitWidth = true
        identifyButton.titleLabel?.minimumScaleFactor = 0.5
        
        // 確保按鈕背景層正確更新
        identifyButton.layoutIfNeeded()
        buttonBlurView.frame = identifyButton.bounds
        borderLayer.frame = identifyButton.bounds
        
        // 確保 hintContainer 背景層正確更新
        hintContainer.layoutIfNeeded()
        hintBlurView.frame = hintContainer.bounds
        hintBorderLayer.frame = hintContainer.bounds
    }
    
    @objc private func buttonTouchDown() {
        UIView.animate(withDuration: 0.1) {
            self.identifyButton.transform = CGAffineTransform(scaleX: 0.95, y: 0.95)  // 按鈕按下時縮放更明顯
            self.identifyButton.alpha = 0.8  // 按鈕按下時透明度變化更明顯
        }
        // 添加觸覺反饋
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()
    }
    
    @objc private func buttonTouchUp() {
        UIView.animate(withDuration: 0.1) {
            self.identifyButton.transform = .identity
            self.identifyButton.alpha = 1.0
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
