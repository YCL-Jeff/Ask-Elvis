import Foundation
import UIKit

/// Manages the visualization of object detection results with bounding boxes and labels, focused on donkey recognition
class BoundingBoxView {
    let shapeLayer: CAShapeLayer
    let textLayer: CATextLayer

    init() {
        shapeLayer = CAShapeLayer()
        shapeLayer.fillColor = UIColor.clear.cgColor
        shapeLayer.lineWidth = 4
        shapeLayer.isHidden = true

        textLayer = CATextLayer()
        textLayer.isHidden = true
        textLayer.contentsScale = UIScreen.main.scale
        textLayer.fontSize = 16
        textLayer.font = UIFont(name: "Avenir-Heavy", size: textLayer.fontSize)
        textLayer.alignmentMode = .center
    }

    func addToLayer(_ parent: CALayer) {
        parent.addSublayer(shapeLayer)
        parent.addSublayer(textLayer)
    }

    func show(frame: CGRect, label: String, color: UIColor = .orange) {
        CATransaction.setDisableActions(true)

        let path = UIBezierPath(roundedRect: frame, cornerRadius: 6.0)
        shapeLayer.path = path.cgPath
        shapeLayer.strokeColor = color.withAlphaComponent(0.8).cgColor
        shapeLayer.isHidden = false

        textLayer.string = label
        textLayer.backgroundColor = color.withAlphaComponent(0.8).cgColor
        textLayer.isHidden = false
        textLayer.foregroundColor = UIColor.white.cgColor

        let attributes = [NSAttributedString.Key.font: textLayer.font as Any]
        let textRect = label.boundingRect(
            with: CGSize(width: 400, height: 100),
            options: .truncatesLastVisibleLine,
            attributes: attributes, context: nil)
        let textSize = CGSize(width: textRect.width + 12, height: textRect.height)
        let textOrigin = CGPoint(x: frame.origin.x - 2, y: frame.origin.y - textSize.height - 2)
        textLayer.frame = CGRect(origin: textOrigin, size: textSize)
    }

    func hide() {
        shapeLayer.isHidden = true
        textLayer.isHidden = true
    }
}
