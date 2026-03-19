import 'dart:io';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

class FaceXAIService {
  // CHANGED: We now require the landmarks Map!
  static Future<File> generateHeatmap(File imageFile, String prediction,
      Map<String, List<double>> landmarks) async {
    final bytes = await imageFile.readAsBytes();
    img.Image? original = img.decodeImage(bytes);
    if (original == null) return imageFile;

    img.Image heatmap = img.copyResize(original,
        width: original.width, height: original.height);
    bool isPD = prediction.toLowerCase().contains("parkinson");

    final highlightColor =
        isPD ? img.ColorRgb8(255, 50, 50) : img.ColorRgb8(50, 255, 50);

    // Calculate dynamic box sizes based on image resolution
    int boxW = (heatmap.width * 0.12).toInt();
    int boxH = (heatmap.height * 0.06).toInt();

    // Helper function to draw a precise box around a coordinate
    void drawTargetBox(List<double> center) {
      int cx = center[0].toInt();
      int cy = center[1].toInt();
      img.drawRect(heatmap,
          x1: cx - boxW,
          y1: cy - boxH,
          x2: cx + boxW,
          y2: cy + boxH,
          color: highlightColor,
          thickness: 4);
    }

    // Draw the 3 targets perfectly over the exact facial features!
    drawTargetBox(landmarks['leftEye']!);
    drawTargetBox(landmarks['rightEye']!);
    drawTargetBox(landmarks['mouth']!);

    final tempDir = await getTemporaryDirectory();
    final xaiFile = File(
        '${tempDir.path}/face_xai_${DateTime.now().millisecondsSinceEpoch}.png');
    await xaiFile.writeAsBytes(img.encodePng(heatmap));

    return xaiFile;
  }

  static List<String> getFaceBiomarkers(String prediction) {
    if (prediction.toLowerCase().contains("parkinson")) {
      return ["Hypomimia (Masking)", "Eye Asymmetry", "Mouth Slant"];
    }
    return ["Normal Expression", "Symmetric Muscle Tone"];
  }
}
