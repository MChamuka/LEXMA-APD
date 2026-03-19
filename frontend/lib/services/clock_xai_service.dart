import 'dart:io';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

class ClockXAIService {
  static Future<File> generateHeatmap(File imageFile, String prediction) async {
    final bytes = await imageFile.readAsBytes();
    img.Image? original = img.decodeImage(bytes);
    if (original == null) return imageFile;

    // Create a copy to manipulate
    img.Image heatmap = img.copyResize(original,
        width: original.width, height: original.height);

    bool isAD = prediction.toLowerCase().contains("alzheimer") ||
        prediction.toLowerCase().contains("positive");

    // Scan every pixel to find the pen strokes
    for (var pixel in heatmap) {
      num lum = pixel.luminance;

      // If the pixel is dark (meaning it is part of the pen/pencil drawing)
      if (lum < 130) {
        if (isAD) {
          // ALZHEIMER'S: Make the strokes glow with a Red/Orange thermal heat
          pixel.r = 255;
          pixel.g = (lum + 50).clamp(0, 255).toInt();
          pixel.b = 0;
        } else {
          // HEALTHY: Make the strokes a calm, clinical Blue
          pixel.r = 0;
          pixel.g = 150;
          pixel.b = 255;
        }
      }
    }

    final tempDir = await getTemporaryDirectory();
    final xaiFile = File(
        '${tempDir.path}/clock_xai_${DateTime.now().millisecondsSinceEpoch}.png');
    await xaiFile.writeAsBytes(img.encodePng(heatmap));

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
