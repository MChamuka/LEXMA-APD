import 'dart:io';
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;

class SpiralProcessor {
  static Future<Uint8List> process(File file) async {
    cv.Mat img = cv.imread(file.path);
    if (img.isEmpty) return file.readAsBytesSync();

    // 1. Grayscale & Threshold
    cv.Mat gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);

    // A tiny blur helps Otsu ignore paper texture and camera noise
    cv.Mat blurred = cv.gaussianBlur(gray, (3, 3), 0);

    var threshResult =
        cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    cv.Mat thresh = threshResult.$2;

    // 2. Find Contours
    var contoursResult =
        cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    var contours = contoursResult.$1;

    if (contours.isEmpty) return file.readAsBytesSync();

    // 3. Find the global bounding box
    int minX = thresh.cols, minY = thresh.rows, maxX = 0, maxY = 0;
    for (int i = 0; i < contours.length; i++) {
      var rect = cv.boundingRect(contours[i]);
      if (rect.x < minX) minX = rect.x;
      if (rect.y < minY) minY = rect.y;
      if (rect.x + rect.width > maxX) maxX = rect.x + rect.width;
      if (rect.y + rect.height > maxY) maxY = rect.y + rect.height;
    }

    // 4. Crop with 20px padding
    int pad = 20;
    int x = (minX - pad).clamp(0, thresh.cols);
    int y = (minY - pad).clamp(0, thresh.rows);
    int w = (maxX - minX + 2 * pad).clamp(0, thresh.cols - x);
    int h = (maxY - minY + 2 * pad).clamp(0, thresh.rows - y);

    cv.Mat cropped = thresh.region(cv.Rect(x, y, w, h));

    // 5. Resize to 224x224
    cv.Mat resized =
        cv.resize(cropped, (224, 224), interpolation: cv.INTER_AREA);

    // 6. Back to 3 channels
    cv.Mat finalImg = cv.cvtColor(resized, cv.COLOR_GRAY2RGB);

    // FIX: Encode to JPG instead of PNG to prevent Android transparency bugs in PyTorch
    var encoded = cv.imencode('.jpg', finalImg);
    return encoded.$2;
  }
}
