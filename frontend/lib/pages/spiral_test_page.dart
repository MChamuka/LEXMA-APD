import 'dart:ui' as ui;
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart' show rootBundle;
import 'dart:math' as math;

class SpiralTestPage extends StatefulWidget {
  final ClassificationModel model;
  const SpiralTestPage({super.key, required this.model});

  @override
  State<SpiralTestPage> createState() => _SpiralTestPageState();
}

class _SpiralTestPageState extends State<SpiralTestPage> {
  List<Offset?> points = [];
  final GlobalKey _drawingKey = GlobalKey();
  String _result = "Draw the spiral...";
  bool _isAnalyzing = false;

  void _clear() => setState(() {
        points.clear();
        _result = "Draw the spiral...";
      });

  Future<void> _analyzeDrawing() async {
    setState(() => _isAnalyzing = true);
    try {
      // 1. Capture the transparent drawing from the UI
      RenderRepaintBoundary boundary = _drawingKey.currentContext!
          .findRenderObject() as RenderRepaintBoundary;
      ui.Image image = await boundary.toImage(pixelRatio: 2.0);
      ByteData? byteData =
          await image.toByteData(format: ui.ImageByteFormat.png);
      Uint8List rawBytes = byteData!.buffer.asUint8List();
      img.Image transparentDrawing = img.decodeImage(rawBytes)!;

      // 2. Load the Template Asset
      ByteData templateByteData =
          await rootBundle.load('assets/spiral_template.png');
      img.Image templateImg =
          img.decodeImage(templateByteData.buffer.asUint8List())!;

      // Resize drawing to match template and paste the ink onto it
      img.Image compositedImg = img.copyResize(transparentDrawing,
          width: templateImg.width, height: templateImg.height);
      img.compositeImage(templateImg, compositedImg);

      // 3. Convert to Grayscale
      img.Image gray = img.grayscale(templateImg);

      // 4. THE MAGIC: Mimic Python's cv2.THRESH_BINARY_INV!
      int w = gray.width;
      int h = gray.height;
      int minX = w, minY = h, maxX = 0, maxY = 0;

      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          var pixel = gray.getPixel(x, y);

          // Invert logic: Dark ink becomes PURE WHITE, Light paper becomes PURE BLACK
          if (pixel.luminance < 150) {
            pixel.r = 255;
            pixel.g = 255;
            pixel.b = 255; // White ink
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
          } else {
            pixel.r = 0;
            pixel.g = 0;
            pixel.b = 0; // Black paper
          }
        }
      }

      // 5. Smart Crop tightly around the white lines
      img.Image finalImage;
      if (minX <= maxX && minY <= maxY) {
        int pad = 20;
        minX = (minX - pad).clamp(0, w - 1);
        minY = (minY - pad).clamp(0, h - 1);
        maxX = (maxX + pad).clamp(0, w - 1);
        maxY = (maxY + pad).clamp(0, h - 1);

        img.Image cropped = img.copyCrop(gray,
            x: minX, y: minY, width: maxX - minX, height: maxY - minY);
        finalImage = img.copyResize(cropped, width: 224, height: 224);
      } else {
        finalImage = img.copyResize(gray, width: 224, height: 224);
      }

      Uint8List inputBytes = Uint8List.fromList(img.encodeJpg(finalImage));

      // 6. Run Prediction
      List<double> probs =
          await widget.model.getImagePredictionList(inputBytes);

      String rawArrayText = probs.toString();
      double pdProb = 0.0;

      // 🔥 THE ULTIMATE MATH FIX 🔥
      if (probs.isNotEmpty) {
        // We explicitly ignore the fake zero PyTorch Lite added.
        // We only grab your true raw value at index 0.
        double rawValue = probs[0];

        // Apply Sigmoid to convert the raw logit (e.g., 5.0764) to a probability (0.99)
        pdProb = 1.0 / (1.0 + math.exp(-rawValue));
      }

      String diagText =
          pdProb > 0.5 ? "Parkinson's Detected" : "Healthy Pattern";
      String prediction =
          "$diagText\nRAW: $rawArrayText\nPROB: ${(pdProb * 100).toStringAsFixed(2)}%";

      // 7. DIAGNOSTIC MODE: Save the AI's "X-Ray" image
      final tempDir = await getTemporaryDirectory();
      File xrayFile = File(
          '${tempDir.path}/xray_spiral_${DateTime.now().millisecondsSinceEpoch}.jpg');
      await xrayFile.writeAsBytes(inputBytes);

      if (mounted) {
        setState(() => _isAnalyzing = false);
        Navigator.pop(context, {
          'image': xrayFile,
          'prediction': prediction,
          'probability': pdProb, // Finally sending the correct probability!
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _result = "Error: $e";
          _isAnalyzing = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Spiral Test")),
      body: Column(children: [
        Expanded(
          child: Center(
            child: Stack(alignment: Alignment.center, children: [
              Opacity(
                  opacity: 1,
                  child: Image.asset("assets/spiral_template.png",
                      width: 300, height: 300)),
              RepaintBoundary(
                key: _drawingKey,
                child: Container(
                  width: 300,
                  height: 300,
                  color: Colors.transparent,
                  child: GestureDetector(
                    onPanUpdate: (d) {
                      setState(() {
                        RenderBox box = _drawingKey.currentContext!
                            .findRenderObject() as RenderBox;
                        Offset pos = box.globalToLocal(d.globalPosition);
                        if (pos.dx >= 0 &&
                            pos.dx <= 300 &&
                            pos.dy >= 0 &&
                            pos.dy <= 300) {
                          points.add(pos);
                        }
                      });
                    },
                    onPanEnd: (_) => points.add(null),
                    child: CustomPaint(painter: SpiralPainter(points)),
                  ),
                ),
              ),
            ]),
          ),
        ),
        Text(_result,
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
        Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
          ElevatedButton(onPressed: _clear, child: const Text("Clear")),
          ElevatedButton(
              onPressed: _isAnalyzing ? null : _analyzeDrawing,
              child: const Text("Analyze & Submit")),
        ]),
        const SizedBox(height: 30),
      ]),
    );
  }
}

class SpiralPainter extends CustomPainter {
  final List<Offset?> points;
  SpiralPainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    Paint p = Paint()
      ..color = Colors.blue.shade900
      ..strokeWidth = 4.0
      ..strokeCap = StrokeCap.round;

    for (int i = 0; i < points.length - 1; i++) {
      if (points[i] != null && points[i + 1] != null) {
        canvas.drawLine(points[i]!, points[i + 1]!, p);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter old) => true;
}
