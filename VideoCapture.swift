import AVFoundation
import CoreVideo
import UIKit

// Protocol for handling video frame capture events
protocol VideoCaptureDelegate: AnyObject {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer)
}

// Select the best back camera device, simplified to only use wide-angle camera
func bestCaptureDevice(for position: AVCaptureDevice.Position) -> AVCaptureDevice {
    if position == .back {
        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) {
            return device
        } else {
            fatalError("後置廣角攝影機不可用")
        }
    } else if position == .front {
        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) {
            return device
        } else {
            fatalError("前置廣角攝影機不可用")
        }
    } else {
        fatalError("不支援的攝影機位置：\(position)")
    }
}

public class VideoCapture: NSObject {
    var previewLayer: AVCaptureVideoPreviewLayer?
    weak var delegate: VideoCaptureDelegate?

    let captureDevice = bestCaptureDevice(for: .back)
    let captureSession = AVCaptureSession()
    let videoOutput = AVCaptureVideoDataOutput()
    var cameraOutput = AVCapturePhotoOutput()
    let queue = DispatchQueue(label: "camera-queue")

    // Configure camera and capture session with specified session preset
    func setUp(
        sessionPreset: AVCaptureSession.Preset = .hd1280x720,  // Lower resolution for better performance
        completion: @escaping (Bool) -> Void
    ) {
        queue.async {
            let success = self.setUpCamera(sessionPreset: sessionPreset)
            DispatchQueue.main.async {
                completion(success)
            }
        }
    }

    // 配置攝影機輸入、輸出和會話屬性
    private func setUpCamera(sessionPreset: AVCaptureSession.Preset) -> Bool {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = sessionPreset

        guard let videoInput = try? AVCaptureDeviceInput(device: captureDevice) else {
            return false
        }

        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        }

        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.connection?.videoOrientation = .portrait
        self.previewLayer = previewLayer

        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
        ]

        videoOutput.videoSettings = settings
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }

        if captureSession.canAddOutput(cameraOutput) {
            captureSession.addOutput(cameraOutput)
        }

        switch UIDevice.current.orientation {
        case .portrait:
            videoOutput.connection(with: .video)?.videoOrientation = .portrait
        case .portraitUpsideDown:
            videoOutput.connection(with: .video)?.videoOrientation = .portraitUpsideDown
        case .landscapeRight:
            videoOutput.connection(with: .video)?.videoOrientation = .landscapeLeft
        case .landscapeLeft:
            videoOutput.connection(with: .video)?.videoOrientation = .landscapeRight
        default:
            videoOutput.connection(with: .video)?.videoOrientation = .portrait
        }

        if let connection = videoOutput.connection(with: .video) {
            self.previewLayer?.connection?.videoOrientation = connection.videoOrientation
        }

        do {
            try captureDevice.lockForConfiguration()
            captureDevice.focusMode = .continuousAutoFocus
            captureDevice.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
            captureDevice.exposureMode = .continuousAutoExposure
            captureDevice.unlockForConfiguration()
        } catch {
            print("無法配置攝影機設備：\(error)")
            return false
        }

        captureSession.commitConfiguration()
        return true
    }

    // 開始視訊捕捉
    public func start() {
        if !captureSession.isRunning {
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.captureSession.startRunning()
            }
        }
    }

    // 停止視訊捕捉
    public func stop() {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }

    // 根據設備方向更新視訊方向
    func updateVideoOrientation() {
        guard let connection = videoOutput.connection(with: .video) else { return }
        switch UIDevice.current.orientation {
        case .portrait:
            connection.videoOrientation = .portrait
        case .portraitUpsideDown:
            connection.videoOrientation = .portraitUpsideDown
        case .landscapeRight:
            connection.videoOrientation = .landscapeLeft
        case .landscapeLeft:
            connection.videoOrientation = .landscapeRight
        default:
            return
        }

        let currentInput = self.captureSession.inputs.first as? AVCaptureDeviceInput
        if currentInput?.device.position == .front {
            connection.isVideoMirrored = true
        } else {
            connection.isVideoMirrored = false
        }

        self.previewLayer?.connection?.videoOrientation = connection.videoOrientation
    }
}

// 處理 AVCaptureVideoDataOutputSampleBufferDelegate 事件
extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(
        _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        delegate?.videoCapture(self, didCaptureVideoFrame: sampleBuffer)
    }

    public func captureOutput(
        _ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // 可選擇處理丟棄的畫面，例如因為緩衝區滿
    }
}
