import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

class FaceImageProcessor {
  static Future<Uint8List> process(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    img.Image? original = img.decodeImage(bytes);
    if (original == null) throw Exception("Invalid Image");

    // 1. Resize to 224x224
    img.Image resized = img.copyResize(original, width: 224, height: 224);

    // 2. Grayscale + Blur
    img.Image gray = img.grayscale(resized);
    img.Image blurred = img.gaussianBlur(gray, radius: 1);

    // 3. Encode
    return Uint8List.fromList(img.encodePng(blurred));
  }
}
