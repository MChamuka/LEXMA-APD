import 'dart:ui' as ui;
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:image/image.dart' as img;

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
      RenderRepaintBoundary boundary = _drawingKey.currentContext!
          .findRenderObject() as RenderRepaintBoundary;
      ui.Image image = await boundary.toImage(pixelRatio: 2.0);
      ByteData? byteData =
          await image.toByteData(format: ui.ImageByteFormat.png);

      img.Image processed = img.decodeImage(byteData!.buffer.asUint8List())!;
      img.Image gray = img.grayscale(processed);
      for (var p in gray) {
        p.r = p.g = p.b = (p.luminance < 200) ? 255 : 0;
      }

      img.Image resized = img.copyResize(gray, width: 224, height: 224);
      Uint8List input = Uint8List.fromList(img.encodePng(resized));

      // FIXED: Binary Model Logic
      List<double> probs = await widget.model.getImagePredictionList(input);
      double score = probs[0];

      setState(() {
        _result = (score > 0.5)
            ? "Parkinson's Detected (${(score * 100).toStringAsFixed(1)}%)"
            : "Healthy (${((1 - score) * 100).toStringAsFixed(1)}%)";
      });
    } catch (e) {
      setState(() => _result = "Error: $e");
    } finally {
      setState(() => _isAnalyzing = false);
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
              // Template stays back, model doesn't see it
              Opacity(
                  opacity: 0.1,
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
                            pos.dy <= 300) points.add(pos);
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
              child: const Text("Analyze")),
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
      ..color = Colors.blue
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;
    for (int i = 0; i < points.length - 1; i++) {
      if (points[i] != null && points[i + 1] != null)
        canvas.drawLine(points[i]!, points[i + 1]!, p);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter old) => true;
}
