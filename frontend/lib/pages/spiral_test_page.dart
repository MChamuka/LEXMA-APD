import 'dart:ui' as ui;
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:path_provider/path_provider.dart';
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
      // 1. Capture the drawing
      RenderRepaintBoundary boundary = _drawingKey.currentContext!
          .findRenderObject() as RenderRepaintBoundary;
      ui.Image image = await boundary.toImage(pixelRatio: 2.0);
      ByteData? byteData =
          await image.toByteData(format: ui.ImageByteFormat.png);
      Uint8List rawBytes = byteData!.buffer.asUint8List();

      // 2. Save the raw drawing to a File so HomePage can display it
      final tempDir = await getTemporaryDirectory();
      File originalFile = File(
          '${tempDir.path}/spiral_${DateTime.now().millisecondsSinceEpoch}.png');
      await originalFile.writeAsBytes(rawBytes);

      // 3. Process for PyTorch (Resize to 224x224)
      img.Image processed = img.decodeImage(rawBytes)!;
      img.Image resized = img.copyResize(processed, width: 224, height: 224);
      Uint8List input = Uint8List.fromList(img.encodeJpg(resized));

      // 4. Run the Binary Model Prediction
      List<double> probs = await widget.model.getImagePredictionList(input);
      double score = probs[0];

      String prediction = (score > 0.5) ? "Parkinson's" : "Healthy";

      // 5. Instantly close the page and send the Care Package back to Home!
      if (mounted) {
        Navigator.pop(context, {
          'image': originalFile,
          'prediction': prediction,
        });
      }
    } catch (e) {
      setState(() {
        _result = "Error: $e";
        _isAnalyzing = false;
      });
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
                  opacity: 0.1,
                  child: Image.asset("assets/spiral_template.png",
                      width: 300, height: 300)),
              RepaintBoundary(
                key: _drawingKey,
                child: Container(
                  width: 300,
                  height: 300,
                  // FIX 1: Pure white background instead of transparent!
                  color: Colors.white,
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
      // FIX 2: Black ink creates better ML contrast than blue
      ..color = Colors.black
      ..strokeWidth = 4
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
