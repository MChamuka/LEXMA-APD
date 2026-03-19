import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

class ClockProcessor {
  static Future<Uint8List> process(File file) async {
    final bytes = await file.readAsBytes();
    img.Image? original = img.decodeImage(bytes);
    if (original == null) return bytes;

    int w = original.width;
    int h = original.height;
    int minX = w, minY = h, maxX = 0, maxY = 0;

    // 1. Scan the ORIGINAL image to find the dark pen ink coordinates
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        var pixel = original.getPixel(x, y);

        // If the pixel is dark enough to be ink
        if (pixel.luminance < 120) {
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
      }
    }

    // If it found no ink (blank paper), just resize the original photo
    if (minX > maxX || minY > maxY) {
      return img.encodeJpg(img.copyResize(original, width: 224, height: 224));
    }

    // 2. Add 5% padding around the ink
    int pad = (w * 0.05).toInt();
    minX = (minX - pad).clamp(0, w - 1);
    minY = (minY - pad).clamp(0, h - 1);
    maxX = (maxX + pad).clamp(0, w - 1);
    maxY = (maxY + pad).clamp(0, h - 1);

    // 3. Crop the ORIGINAL image (Preserving lighting and texture!)
    img.Image cropped = img.copyCrop(original,
        x: minX, y: minY, width: maxX - minX, height: maxY - minY);

    // 4. Resize exactly to 224x224 for PyTorch
    img.Image finalImage = img.copyResize(cropped, width: 224, height: 224);

    return img.encodeJpg(finalImage);
  }
}
