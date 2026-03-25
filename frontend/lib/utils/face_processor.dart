import 'dart:io';
import 'dart:math';
import 'package:face_detection_tflite/face_detection_tflite.dart';
import 'package:image/image.dart'
    as img; // 🔥 THE FIX: Needed to get image dimensions

class FaceImageProcessor {
  static final FaceDetector _detector = FaceDetector();
  static bool _isInitialized = false;

  static Future<Map<String, dynamic>?> process(File file) async {
    if (!_isInitialized) {
      await _detector.initialize(model: FaceDetectionModel.frontCamera);
      _isInitialized = true;
    }

    try {
      final imageBytes = await file.readAsBytes();

      // 🔥 THE FIX: Get the exact image dimensions so we can normalize the coordinates!
      final decodedImage = img.decodeImage(imageBytes);
      if (decodedImage == null) return null;
      double imgW = decodedImage.width.toDouble();
      double imgH = decodedImage.height.toDouble();

      final faces = await _detector.detectFaces(imageBytes,
          mode: FaceDetectionMode.standard);

      if (faces.isEmpty || faces.first.mesh == null) return null;

      // Keep the raw pixel points for the XAI Heatmap
      final rawPoints = faces.first.mesh!.points;

      // 🔥 THE FIX: Normalize every point to 0.0 - 1.0 space exactly like Python Mediapipe!
      Point nPt(Point p) => Point(p.x / imgW, p.y / imgH);

      // --- Math Helpers ---
      double dist(Point a, Point b) =>
          sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
      double angle(Point a, Point b) => atan2(b.y - a.y, b.x - a.x);
      double safeDiv(double n, double d) => n / (d.abs() > 1e-6 ? d : 1e-6);

      // --- Extract Normalized Landmarks ---
      var lEyeOuter = nPt(rawPoints[33]),
          lEyeInner = nPt(rawPoints[133]),
          lEyeTop = nPt(rawPoints[159]),
          lEyeBot = nPt(rawPoints[145]);
      var rEyeOuter = nPt(rawPoints[263]),
          rEyeInner = nPt(rawPoints[362]),
          rEyeTop = nPt(rawPoints[386]),
          rEyeBot = nPt(rawPoints[374]);
      var mLeft = nPt(rawPoints[61]),
          mRight = nPt(rawPoints[291]),
          mTop = nPt(rawPoints[13]),
          mBot = nPt(rawPoints[14]);
      var lBrow = nPt(rawPoints[70]), rBrow = nPt(rawPoints[300]);

      // --- Feature Calculations (Now safely in 0.0 - 1.0 scale) ---
      double lEyeW = dist(lEyeOuter, lEyeInner);
      double rEyeW = dist(rEyeOuter, rEyeInner);
      double lEyeH = dist(lEyeTop, lEyeBot);
      double rEyeH = dist(rEyeTop, rEyeBot);
      double lEyeOpen = safeDiv(lEyeH, lEyeW);
      double rEyeOpen = safeDiv(rEyeH, rEyeW);
      double eyeOpenAsym = (lEyeOpen - rEyeOpen).abs();
      double lEyeSlant = angle(lEyeOuter, lEyeInner);
      double rEyeSlant = angle(rEyeOuter, rEyeInner);
      double eyeSlantAsym = (lEyeSlant - rEyeSlant).abs();
      double eyeDist = dist(lEyeOuter, rEyeOuter);

      double mouthW = dist(mLeft, mRight);
      double mouthH = dist(mTop, mBot);
      double mouthOpen = safeDiv(mouthH, mouthW);
      double mouthSlant = angle(mLeft, mRight);
      double mouthCornerAsymRaw = (mLeft.y - mRight.y).abs();
      double mouthCornerAsymN = safeDiv(mouthCornerAsymRaw, mouthH);
      double upperLipRaw = (mTop.y - ((mLeft.y + mRight.y) / 2.0)).abs();
      double lowerLipRaw = (mBot.y - ((mLeft.y + mRight.y) / 2.0)).abs();
      double upperLipN = safeDiv(upperLipRaw, mouthH);
      double lowerLipN = safeDiv(lowerLipRaw, mouthH);

      double lBrowEyeRaw = (lBrow.y - lEyeTop.y).abs();
      double rBrowEyeRaw = (rBrow.y - rEyeTop.y).abs();
      double lBrowEyeN = safeDiv(lBrowEyeRaw, lEyeW);
      double rBrowEyeN = safeDiv(rBrowEyeRaw, rEyeW);
      double browAsym = (lBrowEyeN - rBrowEyeN).abs();
      double eyeCornerAsymRaw = (lEyeOuter.y - rEyeOuter.y).abs();
      double eyeCornerAsymN = safeDiv(eyeCornerAsymRaw, eyeDist);

      List<double> rawFeatures = [
        eyeOpenAsym,
        lEyeSlant,
        rEyeSlant,
        eyeSlantAsym,
        mouthOpen,
        mouthSlant,
        mouthCornerAsymRaw,
        upperLipN,
        lowerLipN,
        browAsym,
        eyeCornerAsymRaw,
        mouthCornerAsymN,
        eyeCornerAsymN
      ];
      List<double> features =
          rawFeatures.map((f) => f.isNaN || f.isInfinite ? 0.0 : f).toList();

      // --- Package the exact PIXEL coordinates for the XAI Heatmap ---
      Map<String, List<double>> landmarks = {
        'leftEye': [
          (rawPoints[33].x + rawPoints[133].x) / 2,
          (rawPoints[159].y + rawPoints[145].y) / 2
        ],
        'rightEye': [
          (rawPoints[263].x + rawPoints[362].x) / 2,
          (rawPoints[386].y + rawPoints[374].y) / 2
        ],
        'mouth': [
          (rawPoints[61].x + rawPoints[291].x) / 2,
          (rawPoints[13].y + rawPoints[14].y) / 2
        ],
      };

      return {'features': features, 'landmarks': landmarks};
    } catch (e) {
      print("Face Processing Error: $e");
      return null;
    }
  }
}
