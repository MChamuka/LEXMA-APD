import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

class AudioXAIService {
  /// Generates a colorized heatmap based on the spectrogram intensity
  static Future<File> generateHeatmap(File specFile, String prediction) async {
    final bytes = await specFile.readAsBytes();
    img.Image? original = img.decodeImage(bytes);
    if (original == null) return specFile;

    // Create a new RGB image for the heatmap
    img.Image heatmap =
        img.Image(width: original.width, height: original.height);

    for (int y = 0; y < original.height; y++) {
      for (int x = 0; x < original.width; x++) {
        // Get grayscale intensity (0-255)
        var pixel = original.getPixel(x, y);
        int luminance = img.getLuminance(pixel).toInt();

        // Apply a "Jet" colormap:
        // High intensity -> Red
        // Medium -> Yellow/Green
        // Low -> Blue
        int r = 0, g = 0, b = 0;

        if (luminance > 128) {
          r = (255 * (luminance - 128) / 127).toInt().clamp(0, 255);
          g = (255 * (255 - luminance) / 127).toInt().clamp(0, 255);
        } else {
          g = (255 * luminance / 128).toInt().clamp(0, 255);
          b = (255 * (128 - luminance) / 128).toInt().clamp(0, 255);
        }

        heatmap.setPixelRgb(x, y, r, g, b);
      }
    }

    final tempDir = await getTemporaryDirectory();
    final xaiFile = File('${tempDir.path}/audio_xai_heatmap.png');
    await xaiFile.writeAsBytes(img.encodePng(heatmap));
    return xaiFile;
  }

  /// Returns medical biomarkers based on the prediction string
  static List<String> getBiomarkers(String prediction) {
    if (prediction.toLowerCase().contains("parkinson")) {
      return ["Vocal Tremor", "Reduced Pitch Range", "Dysphonia"];
    } else if (prediction.toLowerCase().contains("alzheimer")) {
      return ["Speech Hesitation", "Simplified Syntax", "Word Finding Pause"];
    }
    return ["Stable Frequency", "Normal Harmonic Ratio"];
  }
}
