import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import '../utils/clock_processor.dart';

class ClockXAIService {
  static Future<File> generateHeatmap({
    required File originalFile,
    required ClassificationModel model,
    required double baseLogit,
    required int targetClassIndex,
  }) async {
    // 1. Get base processed image
    Uint8List baseBytes = await ClockProcessor.process(originalFile);
    img.Image? baseImg = img.decodeImage(baseBytes);
    if (baseImg == null) return originalFile;

    int gridSize = 8; // 8x8 Grid = 64 Inferences
    int boxW = baseImg.width ~/ gridSize;
    int boxH = baseImg.height ~/ gridSize;

    List<List<double>> importanceMap =
        List.generate(gridSize, (_) => List.filled(gridSize, 0.0));
    double maxDrop = 0.0;

    // 2. SLIDING WINDOW OCCLUSION
    for (int y = 0; y < gridSize; y++) {
      for (int x = 0; x < gridSize; x++) {
        img.Image occludedImg = img.copyCrop(baseImg,
            x: 0, y: 0, width: baseImg.width, height: baseImg.height);

        // Block out this grid cell with White (Paper Color)
        img.fillRect(occludedImg,
            x1: x * boxW,
            y1: y * boxH,
            x2: (x + 1) * boxW,
            y2: (y + 1) * boxH,
            color: img.ColorRgb8(255, 255, 255));

        // Test the occluded image
        Uint8List testBytes = img.encodeJpg(occludedImg);
        List<double> probs = await model.getImagePredictionList(testBytes);

        // Calculate how much the confidence DROPPED
        double newLogit = probs[targetClassIndex];
        double drop = baseLogit - newLogit;
        drop = drop < 0 ? 0 : drop;

        importanceMap[y][x] = drop;
        if (drop > maxDrop) maxDrop = drop;
      }
    }

    // 3. DRAW TRUE HEATMAP OVERLAY
    img.Image heatmap = img.copyCrop(baseImg,
        x: 0, y: 0, width: baseImg.width, height: baseImg.height);
    for (int y = 0; y < gridSize; y++) {
      for (int x = 0; x < gridSize; x++) {
        double drop = importanceMap[y][x];
        double intensity = maxDrop > 0 ? drop / maxDrop : 0;

        if (intensity > 0.15) {
          // Only highlight areas that actually mattered
          int x1 = x * boxW, y1 = y * boxH;
          int x2 = (x + 1) * boxW, y2 = (y + 1) * boxH;

          for (int yy = y1; yy < y2; yy++) {
            for (int xx = x1; xx < x2; xx++) {
              var p = heatmap.getPixel(xx, yy);
              // Blend Red Thermal overlay
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

    final tempDir = await getTemporaryDirectory();
    File xaiFile = File(
        '${tempDir.path}/clock_xai_${DateTime.now().millisecondsSinceEpoch}.jpg');
    await xaiFile.writeAsBytes(img.encodeJpg(heatmap));
    return xaiFile;
  }

  static List<String> getClockBiomarkers(String prediction) {
    if (prediction.toLowerCase().contains("alzheimer") ||
        prediction.toLowerCase().contains("positive")) {
      return ["Spatial Neglect", "Irregular Spacing", "Contour Deformity"];
    }
    return ["Intact Visuospatial Function", "Correct Number Placement"];
  }
}
