import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:pytorch_lite/pytorch_lite.dart';

class AudioXAIService {
  static Future<File> generateHeatmap({
    required File specFile,
    required ClassificationModel model,
    required double baseLogit,
    required int targetClassIndex,
  }) async {
    final bytes = await specFile.readAsBytes();
    img.Image? baseImg = img.decodeImage(bytes);
    if (baseImg == null) return specFile;

    int gridSize = 8;
    int boxW = baseImg.width ~/ gridSize;
    int boxH = baseImg.height ~/ gridSize;

    List<List<double>> importanceMap =
        List.generate(gridSize, (_) => List.filled(gridSize, 0.0));
    double maxDrop = 0.0;

    for (int y = 0; y < gridSize; y++) {
      for (int x = 0; x < gridSize; x++) {
        img.Image occludedImg = img.copyCrop(baseImg,
            x: 0, y: 0, width: baseImg.width, height: baseImg.height);

        // Spectrograms have black backgrounds, block with Black
        img.fillRect(occludedImg,
            x1: x * boxW,
            y1: y * boxH,
            x2: (x + 1) * boxW,
            y2: (y + 1) * boxH,
            color: img.ColorRgb8(0, 0, 0));

        Uint8List testBytes = img.encodeJpg(occludedImg);
        List<double> probs = await model.getImagePredictionList(testBytes);

        double drop = baseLogit - probs[targetClassIndex];
        drop = drop < 0 ? 0 : drop;

        importanceMap[y][x] = drop;
        if (drop > maxDrop) maxDrop = drop;
      }
    }

    img.Image heatmap = img.copyCrop(baseImg,
        x: 0, y: 0, width: baseImg.width, height: baseImg.height);
    for (int y = 0; y < gridSize; y++) {
      for (int x = 0; x < gridSize; x++) {
        double intensity = maxDrop > 0 ? importanceMap[y][x] / maxDrop : 0;
        if (intensity > 0.15) {
          for (int yy = y * boxH; yy < (y + 1) * boxH; yy++) {
            for (int xx = x * boxW; xx < (x + 1) * boxW; xx++) {
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

    final tempDir = await getTemporaryDirectory();
    final xaiFile = File(
        '${tempDir.path}/audio_xai_${DateTime.now().millisecondsSinceEpoch}.png');
    await xaiFile.writeAsBytes(img.encodePng(heatmap));
    return xaiFile;
  }

  static List<String> getBiomarkers(String prediction) {
    if (prediction.toLowerCase().contains("parkinson"))
      return ["Vocal Tremor", "Reduced Pitch Range"];
    if (prediction.toLowerCase().contains("alzheimer"))
      return ["Speech Hesitation", "Word Finding Pause"];
    return ["Stable Frequency", "Normal Harmonic Ratio"];
  }
}
