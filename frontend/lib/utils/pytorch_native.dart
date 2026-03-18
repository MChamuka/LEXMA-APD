import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

class PyTorchNative {
  static const platform = MethodChannel('com.lexma.apd/pytorch');

  // Android's C++ library needs an absolute file path, so we copy the asset to a temp folder first.
  static Future<String> _getAbsolutePath() async {
    final byteData = await rootBundle.load('assets/models/PDFace.ptl');
    final file = File('${(await getTemporaryDirectory()).path}/PDFace.ptl');
    await file.writeAsBytes(byteData.buffer
        .asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));
    return file.path;
  }

  static Future<List<double>?> predictFace(List<double> features) async {
    try {
      final String modelPath = await _getAbsolutePath();

      // Send the 13 features to our Kotlin code!
      final List<dynamic> result = await platform.invokeMethod('predictFace', {
        'features': features,
        'modelPath': modelPath,
      });

      return result.cast<double>();
    } catch (e) {
      print("❌ Native PyTorch Error: $e");
      return null;
    }
  }
}
