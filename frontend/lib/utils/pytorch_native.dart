import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

class PyTorchNative {
  static const platform = MethodChannel('com.lexma.apd/pytorch');
  static Future<String> _getAbsolutePath() async {
    final byteData =
        await rootBundle.load('assets/models/face_pure_mobile.ptl');
    final file =
        File('${(await getTemporaryDirectory()).path}/face_pure_mobile.ptl');
    await file.writeAsBytes(byteData.buffer
        .asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));
    return file.path;
  }

  static Future<List<double>> predictFace(List<double> features) async {
    final String modelPath = await _getAbsolutePath();
    final List<dynamic> result = await platform.invokeMethod('predictFace', {
      'features': features,
      'modelPath': modelPath,
    });
    return result.cast<double>();
  }
}
