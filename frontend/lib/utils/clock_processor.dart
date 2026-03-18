import 'dart:io';
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;

class ClockProcessor {
  static Future<Uint8List> process(File file) async {
    cv.Mat img = cv.imread(file.path);
    if (img.isEmpty) return file.readAsBytesSync();

    // 1. Grayscale & Threshold
    cv.Mat gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
    var threshResult = cv.threshold(gray, 80, 255, cv.THRESH_BINARY_INV);
    cv.Mat thresh = threshResult.$2;

    // 2. Morphology (Clean noise)
    cv.Mat kernel = cv.Mat.ones(3, 3, cv.MatType.CV_8UC1);
    cv.Mat morphed = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel);

    // 3. Find Contours
    var contoursResult =
        cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    var contours = contoursResult.$1;

    if (contours.isEmpty) return file.readAsBytesSync();

    // 4. Find the global bounding box
    int minX = img.cols, minY = img.rows, maxX = 0, maxY = 0;
    for (int i = 0; i < contours.length; i++) {
      var rect = cv.boundingRect(contours[i]);
      if (rect.x < minX) minX = rect.x;
      if (rect.y < minY) minY = rect.y;
      if (rect.x + rect.width > maxX) maxX = rect.x + rect.width;
      if (rect.y + rect.height > maxY) maxY = rect.y + rect.height;
    }

    // 5. Crop with 20px padding
    int pad = 20;
    int x = (minX - pad).clamp(0, img.cols);
    int y = (minY - pad).clamp(0, img.rows);
    int w = (maxX - minX + 2 * pad).clamp(0, img.cols - x);
    int h = (maxY - minY + 2 * pad).clamp(0, img.rows - y);

    cv.Mat cropped = img.region(cv.Rect(x, y, w, h));

    // 6. Resize to 224x224 (Fixed the Size tuple issue here!)
    cv.Mat resized =
        cv.resize(cropped, (224, 224), interpolation: cv.INTER_AREA);

    var encoded = cv.imencode('.png', resized);
    return encoded.$2;
  }
}
