import 'dart:io';
import 'dart:math' as math;
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import '../utils/pytorch_native.dart';

class FaceXAIService {
  static Future<File> generateHeatmap({
    required File imageFile,
    required Map<String, List<double>> landmarks,
    required List<double> originalFeatures,
    required double baseLogit,
    required int targetClassIndex,
  }) async {
    final bytes = await imageFile.readAsBytes();
    img.Image? original = img.decodeImage(bytes);
    if (original == null) return imageFile;

    img.Image heatmap = img.copyResize(original,
        width: original.width, height: original.height);

    // 1. PERTURB EYES: Zero out all eye-related features
    List<double> eyeTest = List.from(originalFeatures);
    for (int i in [0, 1, 2, 3, 10, 12]) eyeTest[i] = 0.0;
    List<double> eyeProbs = await PyTorchNative.predictFace(eyeTest);
    double eyeDrop = baseLogit - eyeProbs[targetClassIndex];
    eyeDrop = eyeDrop < 0 ? 0 : eyeDrop;

    // 2. PERTURB MOUTH: Zero out mouth features
    List<double> mouthTest = List.from(originalFeatures);
    for (int i in [4, 5, 6, 7, 8, 11]) mouthTest[i] = 0.0;
    List<double> mouthProbs = await PyTorchNative.predictFace(mouthTest);
    double mouthDrop = baseLogit - mouthProbs[targetClassIndex];
    mouthDrop = mouthDrop < 0 ? 0 : mouthDrop;

    double maxDrop = math.max(eyeDrop, mouthDrop);
    maxDrop = maxDrop < 0.001 ? 0.001 : maxDrop;

    int boxW = (heatmap.width * 0.12).toInt();
    int boxH = (heatmap.height * 0.06).toInt();

    void drawThermalBox(List<double> center, double drop) {
      double intensity = drop / maxDrop;
      if (intensity > 0.1) {
        int x1 = center[0].toInt() - boxW, y1 = center[1].toInt() - boxH;
        int x2 = center[0].toInt() + boxW, y2 = center[1].toInt() + boxH;

        for (int yy = y1; yy < y2; yy++) {
          for (int xx = x1; xx < x2; xx++) {
            if (xx >= 0 &&
                xx < heatmap.width &&
                yy >= 0 &&
                yy < heatmap.height) {
              var p = heatmap.getPixel(xx, yy);
              int r = ((intensity * 255) + (1 - intensity) * p.r)
                  .toInt()
                  .clamp(0, 255);
              int g = ((1 - intensity) * p.g).toInt().clamp(0, 255);
              int b = ((1 - intensity) * p.b).toInt().clamp(0, 255);
              heatmap.setPixelRgb(xx, yy, r, g, b);
            }
          }
        }
      }
    }

    // Only draw heatmaps over the features that actually triggered the diagnosis!
    drawThermalBox(landmarks['leftEye']!, eyeDrop);
    drawThermalBox(landmarks['rightEye']!, eyeDrop);
    drawThermalBox(landmarks['mouth']!, mouthDrop);

    final tempDir = await getTemporaryDirectory();
    final xaiFile = File(
        '${tempDir.path}/face_xai_${DateTime.now().millisecondsSinceEpoch}.png');
    await xaiFile.writeAsBytes(img.encodePng(heatmap));
    return xaiFile;
  }

  static List<String> getFaceBiomarkers(String prediction) {
    if (prediction.toLowerCase().contains("parkinson"))
      return ["Hypomimia (Masking)", "Eye Asymmetry", "Mouth Slant"];
    return ["Normal Expression", "Symmetric Muscle Tone"];
  }
}
