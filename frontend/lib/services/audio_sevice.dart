import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:fftea/fftea.dart';

class AudioService {
  static Future<File> generateV2Spectrogram(String audioPath) async {
    final bytes = await File(audioPath).readAsBytes();
    final audioData = _wavToFloat(bytes);

    // matches n_fft=2048 from your Python training
    final stft = STFT(2048, Window.hanning(2048));
    List<List<double>> spec = [];

    stft.run(audioData, (Float64x2List freq) {
      final magnitudes = freq.magnitudes();
      // Log scale conversion
      spec.add(magnitudes
          .take(1024)
          .map((m) => 20 * (log(m + 1e-9) / ln10))
          .toList());
    });

    // Normalization logic to match Python standardization
    List<double> flat = spec.expand((e) => e).toList();
    double mean = flat.reduce((a, b) => a + b) / flat.length;
    double variance =
        flat.map((x) => pow(x - mean, 2)).reduce((a, b) => a + b) / flat.length;
    double std = sqrt(variance) + 1e-6;

    img.Image image = img.Image(width: spec.length, height: 1024);

    for (int x = 0; x < spec.length; x++) {
      for (int y = 0; y < 1024; y++) {
        double normalized = (spec[x][y] - mean) / std;
        int pixelVal = (((normalized + 2) / 4) * 255).clamp(0, 255).toInt();
        // Set pixels in the new Image format
        image.setPixelRgb(x, 1023 - y, pixelVal, pixelVal, pixelVal);
      }
    }

    img.Image resized = img.copyResize(image, width: 224, height: 224);
    final tempDir = await getTemporaryDirectory();
    final saveFile = File('${tempDir.path}/input_spec.png');
    await saveFile.writeAsBytes(img.encodePng(resized));
    return saveFile;
  }

  static List<double> _wavToFloat(Uint8List bytes) {
    List<double> floats = [];
    for (int i = 44; i < bytes.length - 1; i += 2) {
      int sample = bytes[i] | (bytes[i + 1] << 8);
      if (sample > 32767) sample -= 65536;
      floats.add(sample / 32768.0);
    }
    return floats;
  }
}
