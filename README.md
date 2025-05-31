Ask Elvis: Equine Long-Range Visual Identification System

Welcome to the Ask Elvis iOS App repository, developed for the Isle of Wight Donkey Sanctuary. This app leverages advanced object detection models under a non-commercial license from Ultralytics to enable long-range visual identification of donkeys by name. Designed for sanctuary staff and visitors, Ask Elvis enhances donkey recognition, supporting welfare and educational activities at the sanctuary.

🛠 Project Overview

Ask Elvis is an iOS application designed to identify donkeys by name at the Isle of Wight Donkey Sanctuary using long-range visual detection. The app supports sanctuary operations by aiding staff in tracking donkey health and welfare and enhances visitor engagement through interactive donkey identification. My contributions include:

ONNX Conversion: Converted trained models to ONNX format for efficient deployment on iOS, ensuring compatibility with CoreML.

App Architecture: Developed a modular Swift-based app with a focus on user-friendly interfaces for sanctuary staff and visitors.

Performance Optimization: Optimized model inference through quantization, reducing latency for real-time identification on iOS devices.

Technical Considerations

Model Conversion: Models are converted to ONNX with preprocessing steps including image normalization and resizing to 640x640. Input data uses RGB images with bounding box annotations in normalized coordinates, aligned with app requirements.

Target Devices: The app targets iOS devices running iOS 14.0 or later (iOS 17.0+ for optimized models). Minimum hardware is iPhone 8 or newer with an A11 Bionic chip.

Experimental Plan: Testing involves evaluating quantized models for accuracy (mAP) and inference time across devices (e.g., iPhone 12 vs. iPhone 15). Sample data from the sanctuary (e.g., donkey images with name labels) would aid deployment testing.

Application Scenarios: Features include offline identification for remote areas of the sanctuary and database synchronization to log donkey sightings, pending discussion with sanctuary staff.

🛠 Quickstart: Setting Up Ask Elvis

Prerequisites

Xcode: Install from the Mac App Store.

iOS Device: Requires iPhone/iPad running iOS 14.0 or later (iOS 17.0+ for optimized models).

Apple Developer Account: Sign up at developer.apple.com.

Installation

Clone the Repository:

git clone https://github.com/ultralytics/yolo-ios-app.git

Open in Xcode:

Open YOLO.xcodeproj in Xcode. Select your Apple Developer account under "Signing & Capabilities."

Add Models:

Convert trained models to ONNX or CoreML format:

from ultralytics import YOLO

model = YOLO("path/to/donkey_model.pt") model.export(format="onnx", int8=True, imgsz=640)

Place models in the YOLO/Models directory.

Run the App:

Connect your iOS device, select it as the run target, and click Run in Xcode.

🚀 Usage

Real-Time Identification: Point the camera at a donkey to display its name and details (e.g., Jimbob, Angel).

Model Selection: Choose from model sizes (nano to x-large) based on device capability and detection range.

📚 Research Plan (ELEC6259 Coursework #3)

Aims: Develop an iOS app for long-range donkey name identification to support Isle of Wight Donkey Sanctuary operations and visitor engagement.

Methodology: Use ONNX for model conversion, Swift for app development, and CoreML for inference. Train models on sanctuary-provided donkey images with name labels. Test on iOS devices for accuracy and speed.

Timeline: Weeks 1–4: Model training and conversion; Weeks 5–8: App development and UI design; Weeks 9–12: Performance testing and refinement (Gantt chart in coursework submission).

Ethical Statement: No human data involved; ethical approval not required. The app respects donkey welfare by minimizing interaction stress.

Health and Safety: No significant risks; app usage adheres to sanctuary safety guidelines (e.g., no smoking/vaping on-site).

Environmental Impact: Minimal, leveraging existing devices for sustainable technology use.

Data Management Plan: Detection logs stored securely on-device with optional cloud sync, encrypted to protect sanctuary data.

Commercial Aspects: Limited to non-commercial use under Ultralytics’ license, with potential for educational tools at the sanctuary.

Legal Aspects: Complies with Ultralytics’ non-commercial license and Apple’s App Store policies. No personal data collected, aligning with GDPR.

💡 Contribute

Contributions are welcome! Review our Contributing Guide and share feedback via our Survey.

📄 License

This project uses a non-commercial license from Ultralytics. See the LICENSE file for details.

🤝 Contact

For issues or questions, use GitHub Issues or join our Discord.
# Ask-Elvis
