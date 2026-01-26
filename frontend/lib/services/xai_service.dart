import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:pytorch_lite/pytorch_lite.dart';

class XaiService {
  static const int stride = 28;
  static const int maskSize = 28;
  static const int inputSize = 224;

  static Future<Uint8List?> generateHeatmapOverlay(
      File imageFile, ClassificationModel model) async {
    // 1. Prepare Image
    Uint8List bytes = await imageFile.readAsBytes();
    img.Image? original = img.decodeImage(bytes);
    if (original == null) return null;

    img.Image resized =
        img.copyResize(original, width: inputSize, height: inputSize);

    // 2. Get Baseline
    List<double>? probs;
    try {
      probs = await model.getImagePredictionList(
        Uint8List.fromList(img.encodePng(resized)),
      );
    } catch (e) {
      print("⚠️ XAI Error: $e");
      return null;
    }

    if (probs == null || probs.isEmpty) return null;

    // We track Index 1 (The "Sick" class)
    double baselineProb = probs.length > 1 ? probs[1] : probs[0];
    print("XAI BASELINE: $baselineProb");

    // 3. Occlusion Loop
    List<List<double>> heatmap = [];
    double maxDrop = 0.0000001; // Avoid divide by zero

    for (int y = 0; y < inputSize; y += stride) {
      List<double> row = [];
      for (int x = 0; x < inputSize; x += stride) {
        img.Image masked = resized.clone();

        img.fillRect(
          masked,
          x1: x,
          y1: y,
          x2: x + maskSize,
          y2: y + maskSize,
          color: img.ColorRgba8(0, 0, 0, 255),
        );

        List<double>? maskedProbs;
        try {
          maskedProbs = await model.getImagePredictionList(
            Uint8List.fromList(img.encodePng(masked)),
          );
        } catch (e) {/* ignore */}

        double newProb = (maskedProbs != null && maskedProbs.length > 1)
            ? maskedProbs[1]
            : 0.0;

        // Calculate Drop
        double drop = max(0, baselineProb - newProb);
        row.add(drop);

        if (drop > maxDrop) maxDrop = drop;
      }
      heatmap.add(row);
    }

    print("XAI MAX DROP: $maxDrop");

    // 4. Generate Overlay (NO FILTERS - SHOW EVERYTHING)
    img.Image overlay = img.Image(width: inputSize, height: inputSize);
    img.fill(overlay, color: img.ColorRgba8(0, 0, 0, 0));

    for (int i = 0; i < heatmap.length; i++) {
      for (int j = 0; j < heatmap[i].length; j++) {
        double score = heatmap[i][j];

        // Normalize 0.0 - 1.0 (This forces Red Cubes even if drop is tiny)
        double intensity = score / maxDrop;

        // Show everything with even 10% relative importance
        if (intensity > 0.1) {
          int alpha = (intensity * 150).toInt();

          img.fillRect(
            overlay,
            x1: j * stride,
            y1: i * stride,
            x2: (j * stride) + stride,
            y2: (i * stride) + stride,
            color: img.ColorRgba8(255, 0, 0, alpha), // Red
          );
        }
      }
    }

    return Uint8List.fromList(img.encodePng(overlay));
  }
}
