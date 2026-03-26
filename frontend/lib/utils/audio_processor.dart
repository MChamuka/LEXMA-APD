import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:fftea/fftea.dart';

class AudioProcessor {
  static Future<Uint8List> process(File audioFile) async {
    final bytes = await audioFile.readAsBytes();
    final audioData = _wavToFloat(bytes);

    int nFft = 2048;
    int hopLength = 512;
    final window = Window.hanning(nFft);
    final fft = FFT(nFft);

    List<List<double>> spec = [];
    List<double> allMags =
        []; // 🔥 NEW: Store all sounds to find the true average

    // 1. STFT - Extract Magnitudes
    for (int i = 0; i <= audioData.length - nFft; i += hopLength) {
      List<double> frame = audioData.sublist(i, i + nFft).toList();
      for (int j = 0; j < nFft; j++) {
        frame[j] *= window[j];
      }

      final freq = fft.realFft(frame);
      final magnitudes = freq.magnitudes();

      List<double> frameMag = [];
      for (int j = 0; j <= nFft ~/ 2; j++) {
        double mag = magnitudes[j];
        frameMag.add(mag);
        allMags.add(mag); // Collect every single pixel's volume
      }
      spec.add(frameMag);
    }

    // 🔥 THE FIX: Sort all sounds from quietest to loudest
    allMags.sort();

    // Pick the 98th percentile as the "Max". This completely ignores loud screen taps!
    int targetIndex =
        (allMags.length * 0.98).toInt().clamp(0, allMags.length - 1);
    double trueMaxMag = allMags[targetIndex];
    if (trueMaxMag < 1e-5)
      trueMaxMag = 1e-5; // Safety fallback for absolute silence

    // 2. Librosa power_to_db (With Dynamic Min/Max Tracking)
    List<List<double>> dbSpec = [];
    double minDb = double.infinity;
    double maxDb = double.negativeInfinity;

    for (int i = 0; i < spec.length; i++) {
      List<double> dbCol = [];
      for (int j = 0; j < spec[i].length; j++) {
        double mag = spec[i][j];

        // Cap the accidental loud clicks
        if (mag > trueMaxMag) mag = trueMaxMag;

        // Convert to decibels
        double db = 20 * (log(max(mag, 1e-9) / trueMaxMag) / ln10);
        dbCol.add(db);

        //Track the absolute limits of this specific recording
        if (db < minDb) minDb = db;
        if (db > maxDb) maxDb = db;
      }
      dbSpec.add(dbCol);
    }

    // Prevent divide-by-zero if the audio is completely flat/silent
    if (maxDb == minDb) maxDb = minDb + 1e-6;

    // 3. Perfect Contrast Stretching (Matches Python's cv2.NORM_MINMAX exactly!)
    int width = dbSpec.length;
    int height = dbSpec[0].length;

    // Safety check for empty recordings
    if (width == 0 || height == 0) {
      img.Image blank = img.Image(width: 224, height: 224);
      return Uint8List.fromList(img.encodeJpg(blank));
    }

    img.Image image = img.Image(width: width, height: height);

    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        double dbVal = dbSpec[x][y];

        // 🔥 THE FIX: Stretch the contrast!
        // Quietest sound becomes pure black (0), loudest becomes pure white (255)
        double scaled = (dbVal - minDb) / (maxDb - minDb);
        int pixelVal = (scaled * 255).clamp(0, 255).toInt();

        image.setPixelRgb(x, (height - 1) - y, pixelVal, pixelVal, pixelVal);
      }
    }

    // 4. Resize and output
    img.Image resized = img.copyResize(image, width: 224, height: 224);
    return Uint8List.fromList(img.encodeJpg(resized));
  }

  // Robust WAV reader that skips Android Metadata
  static List<double> _wavToFloat(Uint8List bytes) {
    List<double> floats = [];
    int dataOffset = 12;

    try {
      while (dataOffset < bytes.length - 8) {
        String chunkId =
            String.fromCharCodes(bytes.sublist(dataOffset, dataOffset + 4));
        int chunkSize = bytes[dataOffset + 4] |
            (bytes[dataOffset + 5] << 8) |
            (bytes[dataOffset + 6] << 16) |
            (bytes[dataOffset + 7] << 24);

        if (chunkId == 'data') {
          dataOffset += 8;
          for (int i = dataOffset;
              i < dataOffset + chunkSize - 1 && i < bytes.length - 1;
              i += 2) {
            int sample = bytes[i] | (bytes[i + 1] << 8);
            if (sample > 32767) sample -= 65536;
            floats.add(sample / 32768.0);
          }
          break;
        } else {
          dataOffset += 8 + chunkSize;
        }
      }
    } catch (e) {
      // Fallback
      for (int i = 44; i < bytes.length - 1; i += 2) {
        int sample = bytes[i] | (bytes[i + 1] << 8);
        if (sample > 32767) sample -= 65536;
        floats.add(sample / 32768.0);
      }
    }
    return floats;
  }
}
