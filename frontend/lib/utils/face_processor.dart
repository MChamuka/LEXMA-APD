import 'dart:io';
import 'dart:math';
import 'package:face_detection_tflite/face_detection_tflite.dart';

class FaceImageProcessor {
  // Keep an instance of the detector so it doesn't re-load the model every time
  static final FaceDetector _detector = FaceDetector();
  static bool _isInitialized = false;

  static Future<List<double>?> process(File file) async {
    if (!_isInitialized) {
      await _detector.initialize(model: FaceDetectionModel.frontCamera);
      _isInitialized = true;
    }

    try {
      final imageBytes = await file.readAsBytes();

      // Standard mode guarantees we get the 468-point mesh
      final faces = await _detector.detectFaces(imageBytes,
          mode: FaceDetectionMode.standard);

      if (faces.isEmpty || faces.first.mesh == null) return null;

      // Get the exact 468 MediaPipe points
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
      // 1. Eyes
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

      double eyeDist = dist(lEyeOuter, rEyeOuter); // For normalization scaling

      // 2. Mouth
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

      // 3. Brows
      double lBrowEyeRaw = (lBrow.y - lEyeTop.y).abs();
      double rBrowEyeRaw = (rBrow.y - rEyeTop.y).abs();
      double lBrowEyeN = safeDiv(lBrowEyeRaw, lEyeW);
      double rBrowEyeN = safeDiv(rBrowEyeRaw, rEyeW);
      double browAsym = (lBrowEyeN - rBrowEyeN).abs();

      // 4. Eye Corners
      double eyeCornerAsymRaw = (lEyeOuter.y - rEyeOuter.y).abs();
      double eyeCornerAsymN = safeDiv(eyeCornerAsymRaw, eyeDist);

      // --- Final 13-Feature Array ---
      List<double> features = [
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

      // Clean up NaN/Infinity exactly like Python's np.nan_to_num
      return features.map((f) => f.isNaN || f.isInfinite ? 0.0 : f).toList();
    } catch (e) {
      print("Face Processing Error: $e");
      return null;
    }
  }
}
