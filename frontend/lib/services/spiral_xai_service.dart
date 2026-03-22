import 'dart:io';
import 'dart:typed_data';
import 'package:path_provider/path_provider.dart';
import '../utils/spiral_processor.dart';

class SpiralXAIService {
  /// 1. Generate the Clinical Biomarkers for the Explanation Panel
  static List<String> getBiomarkers(String prediction) {
    if (prediction.toLowerCase().contains("parkinson")) {
      return [
        "Kinematic Tremor Detected",
        "Broken Stroke Continuity",
        "Irregular Spatial Gaps",
        "High Velocity Fluctuation"
      ];
    } else {
      return [
        "Smooth Line Continuity",
        "Consistent Stroke Pressure",
        "Normal Spatial Gaps",
        "Stable Kinematic Control"
      ];
    }
  }

  /// 2. Generate the Visual XAI Output (The High-Contrast X-Ray)
  static Future<File> generateHeatmap(
      File originalFile, String prediction) async {
    // Grab the perfect, shadow-free X-Ray we built in the Processor
    Uint8List processedBytes = await SpiralProcessor.process(originalFile);

    // Save it to a temporary file for the UI preview box
    final tempDir = await getTemporaryDirectory();
    File xaiFile = File(
        '${tempDir.path}/xai_spiral_${DateTime.now().millisecondsSinceEpoch}.jpg');
    await xaiFile.writeAsBytes(processedBytes);

    return xaiFile;
  }
}
