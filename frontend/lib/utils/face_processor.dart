import 'dart:io';
import 'dart:math';
import 'package:face_detection_tflite/face_detection_tflite.dart';

class FaceImageProcessor {
  static final FaceDetector _detector = FaceDetector();
  static bool _isInitialized = false;

  // CHANGED: Now returns a Map containing both the features AND the exact landmarks
  static Future<Map<String, dynamic>?> process(File file) async {
    if (!_isInitialized) {
      await _detector.initialize(model: FaceDetectionModel.frontCamera);
      _isInitialized = true;
    }

    try {
      final imageBytes = await file.readAsBytes();
      final faces = await _detector.detectFaces(imageBytes,
          mode: FaceDetectionMode.standard);

      if (faces.isEmpty || faces.first.mesh == null) return null;
      final points = faces.first.mesh!.points;

      // --- Math Helpers ---
      double dist(Point a, Point b) =>
          sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
      double angle(Point a, Point b) => atan2(b.y - a.y, b.x - a.x);
      double safeDiv(double n, double d) => n / (d.abs() > 1e-6 ? d : 1e-6);

      // --- Extract Specific Landmarks ---
      var lEyeOuter = points[33],
          lEyeInner = points[133],
          lEyeTop = points[159],
          lEyeBot = points[145];
      var rEyeOuter = points[263],
          rEyeInner = points[362],
          rEyeTop = points[386],
          rEyeBot = points[374];
      var mLeft = points[61],
          mRight = points[291],
          mTop = points[13],
          mBot = points[14];
      var lBrow = points[70], rBrow = points[300];

      // --- Feature Calculations ---
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

      // --- THE NEW FIX: Package the exact coordinates! ---
      Map<String, List<double>> landmarks = {
        'leftEye': [
          (lEyeOuter.x + lEyeInner.x) / 2,
          (lEyeTop.y + lEyeBot.y) / 2
        ],
        'rightEye': [
          (rEyeOuter.x + rEyeInner.x) / 2,
          (rEyeTop.y + rEyeBot.y) / 2
        ],
        'mouth': [(mLeft.x + mRight.x) / 2, (mTop.y + mBot.y) / 2],
      };

      // Return both!
      return {'features': features, 'landmarks': landmarks};
    } catch (e) {
      print("Face Processing Error: $e");
      return null;
    }
  }
}
