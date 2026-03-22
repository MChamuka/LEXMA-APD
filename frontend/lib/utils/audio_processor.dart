import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:fftea/fftea.dart';

class AudioProcessor {
  /// Converts a .wav file directly into a 224x224 Grayscale Librosa-matched Spectrogram
  static Future<Uint8List> process(File audioFile) async {
    final bytes = await audioFile.readAsBytes();
    final audioData = _wavToFloat(bytes);

    int nFft = 2048;
    int hopLength = 512; // 🔥 THE FIX: Matches Librosa overlap exactly!
    final window = Window.hanning(nFft);
    final fft = FFT(nFft);

    List<List<double>> spec = [];

    // 1. STFT with Overlapping Frames
    for (int i = 0; i <= audioData.length - nFft; i += hopLength) {
      List<double> frame = audioData.sublist(i, i + nFft).toList();

      // Apply Hanning Window
      for (int j = 0; j < nFft; j++) {
        frame[j] *= window[j];
      }

      final freq = fft.realFft(frame);
      final magnitudes = freq.magnitudes();

      // Convert to Decibels (librosa.amplitude_to_db)
      List<double> dbCol = [];
      for (int j = 0; j <= nFft ~/ 2; j++) {
        double mag = magnitudes[j];
        double db = 20 * (log(mag + 1e-9) / ln10);
        dbCol.add(db);
      }
      spec.add(dbCol);
    }

    // 2. Statistical Z-Score Normalization
    double sum = 0;
    int count = 0;
    for (var col in spec) {
      for (var val in col) {
        sum += val;
        count++;
      }
    }
    double mean = sum / count;

    double sqSum = 0;
    for (var col in spec) {
      for (var val in col) {
        sqSum += pow(val - mean, 2);
      }
    }
    double std = sqrt(sqSum / count) + 1e-6;

    double minVal = double.infinity;
    double maxVal = double.negativeInfinity;

    // Apply Z-Score and find new Min/Max for scaling
    for (int i = 0; i < spec.length; i++) {
      for (int j = 0; j < spec[i].length; j++) {
        spec[i][j] = (spec[i][j] - mean) / std;
        if (spec[i][j] < minVal) minVal = spec[i][j];
        if (spec[i][j] > maxVal) maxVal = spec[i][j];
      }
    }

    // 3. Absolute Min-Max Scaling (Matches cv2.NORM_MINMAX)
    int width = spec.length;
    int height = spec[0].length;
    img.Image image = img.Image(width: width, height: height);

    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        double val = spec[x][y];
        double scaled =
            (val - minVal) / (maxVal - minVal); // Force between 0.0 and 1.0
        int pixelVal = (scaled * 255).clamp(0, 255).toInt();

        // Librosa puts low frequencies at the bottom (y=height-1)
        image.setPixelRgb(x, (height - 1) - y, pixelVal, pixelVal, pixelVal);
      }
    }

    // 4. Resize for PyTorch and output pure bytes
    img.Image resized = img.copyResize(image, width: 224, height: 224);
    return Uint8List.fromList(img.encodeJpg(resized));
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
