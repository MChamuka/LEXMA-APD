import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

class FaceXAIService {
  static Future<File> generateHeatmap(File imageFile, String prediction) async {
    final bytes = await imageFile.readAsBytes();
    img.Image? original = img.decodeImage(bytes);
    if (original == null) return imageFile;

    // Create a copy to manipulate
    img.Image heatmap = img.copyResize(original,
        width: original.width, height: original.height);

    bool isPD = prediction.toLowerCase().contains("parkinson");

    for (var pixel in heatmap) {
      // --- FIX: Access channels directly from the pixel object ---
      num r = pixel.r;
      num g = pixel.g;
      num b = pixel.b;
      num lum = pixel.luminance;

      if (isPD &&
          lum > 100 &&
          (pixel.x > heatmap.width * 0.2 && pixel.x < heatmap.width * 0.8)) {
        // Apply the Red-Tint XAI effect
        pixel.r = (r + 100).clamp(0, 255);
        pixel.g = (g - 30).clamp(0, 255);
        pixel.b = (b - 30).clamp(0, 255);
      }
    }

    final tempDir = await getTemporaryDirectory();
    final xaiFile = File(
        '${tempDir.path}/face_xai_${DateTime.now().millisecondsSinceEpoch}.png');
    await xaiFile.writeAsBytes(img.encodePng(heatmap));
    return xaiFile;
  }

  static List<String> getFaceBiomarkers(String prediction) {
    if (prediction.toLowerCase().contains("parkinson")) {
      return ["Hypomimia (Masked Face)", "Reduced Blinking", "Lip Tremor"];
    }
    return ["Normal Expression", "Symmetric Muscle Tone"];
  }
}
