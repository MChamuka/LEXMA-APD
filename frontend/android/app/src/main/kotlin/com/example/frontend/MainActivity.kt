package com.example.frontend

import androidx.annotation.NonNull
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.lexma.apd/pytorch"

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            if (call.method == "predictFace") {
                val features = call.argument<List<Double>>("features")
                val modelPath = call.argument<String>("modelPath")

                if (features != null && modelPath != null) {
                    try {
                        // 1. Convert Dart's double array to a Kotlin FloatArray
                        val floatArray = features.map { it.toFloat() }.toFloatArray()
                        
                        // 2. Create a 1x13 PyTorch Tensor
                        val shape = longArrayOf(1, 13)
                        val tensor = Tensor.fromBlob(floatArray, shape)

                        // 3. Load the model and run the prediction (Updated for PyTorch 2.1.0)
                        val module = Module.load(modelPath)
                        val outputTensor = module.forward(IValue.from(tensor)).toTensor()
                        val scores = outputTensor.dataAsFloatArray

                        // 4. Send the probabilities back to Flutter!
                        result.success(scores.toList())
                    } catch (e: Exception) {
                        result.error("PYTORCH_ERROR", e.message, null)
                    }
                } else {
                    result.error("INVALID_ARGS", "Missing features or path", null)
                }
            } else {
                result.notImplemented()
            }
        }
    }
}