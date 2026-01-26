import 'dart:typed_data';
import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

// --- PACKAGES ---
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image/image.dart' as img;
import 'package:fftea/fftea.dart';

// --- MODEL ENGINES ---
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

// --- PAGES ---
import 'spiral_test_page.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  // --- MODELS ---
  ClassificationModel? _faceModel;
  ClassificationModel? _spiralModel;
  ClassificationModel? _audioModel;
  ClassificationModel? _dementiaModel;

  tfl.Interpreter? _lifestyleInterpreter;

  bool _areModelsLoaded = false;

  // Audio Recorder
  final AudioRecorder _audioRecorder = AudioRecorder();
  bool _isRecording = false;

  // --- DATA ---
  final List<File?> _images = [null, null, null, null, null];
  final List<String> _results = [
    "Not Analyzed",
    "Not Analyzed",
    "Not Analyzed",
    "Not Analyzed",
    "Not Analyzed"
  ];

  // --- LIFESTYLE VARIABLES ---
  int gender = 0;
  int ethnicity = 0;
  int education = 0;
  int smoking = 0;
  int hypertension = 0;
  int diabetes = 0;
  int depression = 0;

  final ageController = TextEditingController();
  final bmiController = TextEditingController();
  final alcoholController = TextEditingController();
  final activityController = TextEditingController();
  final dietController = TextEditingController();
  final sleepController = TextEditingController();
  final tasksController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _loadAllModels();
  }

  @override
  void dispose() {
    _audioRecorder.dispose();
    ageController.dispose();
    bmiController.dispose();
    alcoholController.dispose();
    activityController.dispose();
    dietController.dispose();
    sleepController.dispose();
    tasksController.dispose();
    super.dispose();
  }

  Future<ClassificationModel?> loadSafePytorch(
      String path, String labelPath) async {
    try {
      stderr.writeln("🚀 ATTEMPTING LOAD: $path");
      final model = await PytorchLite.loadClassificationModel(
        path,
        224,
        224,
        3,
        labelPath: labelPath,
      );
      stderr.writeln("✅ LOAD SUCCESS: $path");
      return model;
    } catch (e) {
      stderr.writeln("❌ LOAD FAILED: $path - $e");
      return null;
    }
  }

  Future<void> _loadAllModels() async {
    try {
      stderr.writeln("--- STARTING MODEL BOOT SEQUENCE ---");

      _faceModel = await loadSafePytorch(
          "assets/models/PDFace.ptl", "assets/labels.txt");
      _spiralModel = await loadSafePytorch(
          "assets/models/spiralHandPD.ptl", "assets/spiral_labels.txt");
      _audioModel = await loadSafePytorch(
          "assets/models/ewadbVoice.ptl", "assets/labels.txt");
      _dementiaModel = await loadSafePytorch(
          "assets/models/simplecnn_dementia_mobile.ptl", "assets/labels.txt");

      try {
        _lifestyleInterpreter =
            await tfl.Interpreter.fromAsset('assets/models/lifestyle.tflite');
        stderr.writeln("✅ TFLite Model Loaded");
      } catch (e) {
        stderr.writeln("❌ Failed to load Lifestyle TFLite: $e");
      }
    } catch (globalError) {
      stderr.writeln("💣 GLOBAL CRASH IN _loadAllModels: $globalError");
    } finally {
      if (mounted) {
        setState(() => _areModelsLoaded = true);
      }
    }
  }

  Future<void> _pickImage(int index, ClassificationModel? model) async {
    if (model == null) return;
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);

    if (image != null) {
      setState(() {
        _images[index] = File(image.path);
        _results[index] = "Analyzing...";
      });
      try {
        Uint8List imageBytes;
        if (index == 0) {
          imageBytes = await FaceImageProcessor.process(File(image.path));
        } else {
          imageBytes = await File(image.path).readAsBytes();
        }

        String prediction = await model.getImagePrediction(imageBytes);
        setState(() => _results[index] = prediction);
      } catch (e) {
        setState(() => _results[index] = "Error: $e");
      }
    }
  }

  Future<void> _handleAudioRecording() async {
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) return;

    if (!_isRecording) {
      try {
        final directory = await getTemporaryDirectory();
        String path = '${directory.path}/temp_audio.wav';
        await _audioRecorder
            .start(const RecordConfig(encoder: AudioEncoder.wav), path: path);
        setState(() {
          _isRecording = true;
          _results[2] = "Recording...";
        });
      } catch (e) {
        stderr.writeln("Recording Error: $e");
      }
    } else {
      final path = await _audioRecorder.stop();
      setState(() => _isRecording = false);
      if (path != null && _audioModel != null) {
        setState(() => _results[2] = "Analyzing...");
        try {
          File spectrogramFile =
              await AudioSpectrogramGenerator.generateSpectrogram(path);
          setState(() => _images[2] = spectrogramFile);
          Uint8List bytes = await spectrogramFile.readAsBytes();
          String prediction = await _audioModel!.getImagePrediction(bytes);
          setState(() => _results[2] = prediction);
        } catch (e) {
          setState(() => _results[2] = "Failed: $e");
        }
      }
    }
  }

  List<double> _oneHot(int selectedIndex, int totalCategories) {
    List<double> output = List.filled(totalCategories, 0.0);
    if (selectedIndex >= 0 && selectedIndex < totalCategories) {
      output[selectedIndex] = 1.0;
    }
    return output;
  }

  void _runLifestyleAnalysis() {
    if (_lifestyleInterpreter == null) return;
    try {
      List<double> inputRow = [];
      inputRow.addAll(_oneHot(gender, 2));
      inputRow.addAll(_oneHot(ethnicity, 4));
      inputRow.addAll(_oneHot(education, 4));
      inputRow.addAll(_oneHot(smoking, 2));
      inputRow.addAll(_oneHot(hypertension, 2));
      inputRow.addAll(_oneHot(diabetes, 2));
      inputRow.addAll(_oneHot(depression, 2));

      inputRow.add(double.tryParse(ageController.text) ?? 0.0);
      inputRow.add(double.tryParse(bmiController.text) ?? 0.0);
      inputRow.add(double.tryParse(alcoholController.text) ?? 0.0);
      inputRow.add(double.tryParse(activityController.text) ?? 0.0);
      inputRow.add(double.tryParse(dietController.text) ?? 0.0);
      inputRow.add(double.tryParse(sleepController.text) ?? 0.0);
      inputRow.add(double.tryParse(tasksController.text) ?? 0.0);

      var output = List.filled(1 * 3, 0.0).reshape([1, 3]);
      _lifestyleInterpreter!.run([inputRow], output);
      List<double> probs = output[0];

      int maxIndex = 0;
      double maxVal = probs[0];
      for (int i = 1; i < probs.length; i++) {
        if (probs[i] > maxVal) {
          maxVal = probs[i];
          maxIndex = i;
        }
      }

      String diag = (maxIndex == 0)
          ? "Healthy"
          : (maxIndex == 1 ? "Alzheimer's" : "Parkinson's");
      setState(
          () => _results[4] = "$diag (${(maxVal * 100).toStringAsFixed(1)}%)");
    } catch (e) {
      setState(() => _results[4] = "Error");
    }
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Multimodal Diagnosis")),
      body: !_areModelsLoaded
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              padding: const EdgeInsets.all(16),
              children: [
                _buildModelCard(0, "Face Analysis", "Upload Face", _faceModel,
                    type: 'image'),
                _buildModelCard(1, "Spiral Test", "Start Test", _spiralModel,
                    type: 'spiral'),
                _buildModelCard(2, "Voice Analysis",
                    _isRecording ? "Stop" : "Record Voice", _audioModel,
                    type: 'audio'),
                _buildModelCard(
                    3, "MRI Scan", "Upload Brain Scan", _dementiaModel,
                    type: 'image'),
                _buildLifestyleCard(),
              ],
            ),
    );
  }

  Widget _buildLifestyleCard() {
    return Card(
      margin: const EdgeInsets.only(bottom: 20),
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            const Text("Lifestyle Analysis",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            const Icon(Icons.medical_information,
                size: 50, color: Colors.blueGrey),
            Text("Result: ${_results[4]}",
                style: const TextStyle(fontWeight: FontWeight.bold)),
            ElevatedButton(
                onPressed: () => _openLifestyleForm(),
                child: const Text("Enter Patient Data")),
          ],
        ),
      ),
    );
  }

  Widget _buildModelCard(
      int index, String title, String btnText, ClassificationModel? model,
      {required String type}) {
    bool isMissing = model == null;
    IconData icon = (type == 'audio')
        ? Icons.mic
        : (type == 'spiral' ? Icons.gesture : Icons.image);

    return Card(
      margin: const EdgeInsets.only(bottom: 20),
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Text(title,
                style:
                    const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            if (type != 'spiral')
              Container(
                height: 100,
                width: double.infinity,
                color: Colors.grey[100],
                child: _images[index] == null
                    ? Icon(icon, size: 40, color: Colors.grey)
                    : Image.file(_images[index]!, fit: BoxFit.cover),
              )
            else
              const Icon(Icons.gesture, size: 80, color: Colors.blueAccent),
            const SizedBox(height: 10),
            Text("Result: ${_results[index]}",
                style: const TextStyle(fontWeight: FontWeight.bold)),
            ElevatedButton.icon(
              onPressed: isMissing
                  ? null
                  : () {
                      if (type == 'spiral')
                        Navigator.push(
                            context,
                            MaterialPageRoute(
                                builder: (c) =>
                                    SpiralTestPage(model: _spiralModel!)));
                      else if (type == 'audio')
                        _handleAudioRecording();
                      else
                        _pickImage(index, model);
                    },
              icon: Icon(icon),
              label: Text(isMissing ? "Load Failed" : btnText),
            ),
          ],
        ),
      ),
    );
  }

  void _openLifestyleForm() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text("Lifestyle Assessment"),
        content: SingleChildScrollView(
          child: Column(
            children: [
              _buildDropdown("Gender", ["Female", "Male"], (v) => gender = v),
              _buildDropdown(
                  "Ethnicity",
                  ["Caucasian", "African", "Asian", "Other"],
                  (v) => ethnicity = v),
              _buildDropdown(
                  "Education",
                  ["None", "High School", "Bachelor", "Higher"],
                  (v) => education = v),
              _buildDropdown("Smoking", ["No", "Yes"], (v) => smoking = v),
              _buildDropdown("High BP", ["No", "Yes"], (v) => hypertension = v),
              _buildDropdown("Diabetes", ["No", "Yes"], (v) => diabetes = v),
              _buildDropdown(
                  "Depression", ["No", "Yes"], (v) => depression = v),
              _buildNumField("Age", ageController),
              _buildNumField("BMI", bmiController),
              _buildNumField("Alcohol", alcoholController),
              _buildNumField("Exercise", activityController),
              _buildNumField("Diet", dietController),
              _buildNumField("Sleep", sleepController),
              _buildNumField("Tasks", tasksController),
            ],
          ),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text("Cancel")),
          ElevatedButton(
              onPressed: _runLifestyleAnalysis, child: const Text("Analyze"))
        ],
      ),
    );
  }

  Widget _buildDropdown(
      String label, List<String> items, Function(int) onChanged) {
    return DropdownButtonFormField<String>(
      decoration: InputDecoration(labelText: label),
      items: items
          .asMap()
          .entries
          .map((e) =>
              DropdownMenuItem(value: e.key.toString(), child: Text(e.value)))
          .toList(),
      onChanged: (val) => onChanged(int.parse(val!)),
    );
  }

  Widget _buildNumField(String label, TextEditingController controller) {
    return TextField(
        controller: controller,
        keyboardType: TextInputType.number,
        decoration: InputDecoration(labelText: label));
  }
}

class FaceImageProcessor {
  static Future<Uint8List> process(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    img.Image? original = img.decodeImage(bytes);
    if (original == null) throw Exception("Invalid Image");
    img.Image resized = img.copyResize(original, width: 224, height: 224);
    img.Image gray = img.grayscale(resized);
    return Uint8List.fromList(img.encodePng(gray));
  }
}

class AudioSpectrogramGenerator {
  static Future<File> generateSpectrogram(String audioPath) async {
    final bytes = await File(audioPath).readAsBytes();
    final audioData = _wavToFloat(bytes);
    final stft = STFT(2048, Window.hanning(2048));
    List<List<double>> spec = [];
    stft.run(audioData, (Float64x2List freq) {
      final magnitudes = freq.magnitudes();
      spec.add(magnitudes
          .take(1024)
          .map((m) => 10 * (log(m + 1e-9) / ln10))
          .toList());
    });
    img.Image image = img.Image(width: spec.length, height: spec[0].length);
    // Note: Standard grayscale mapping for spectrogram
    for (int x = 0; x < spec.length; x++) {
      for (int y = 0; y < spec[0].length; y++) {
        int color = (spec[x][y].abs() * 2).toInt().clamp(0, 255);
        image.setPixelRgb(x, spec[0].length - 1 - y, color, color, color);
      }
    }
    img.Image resized = img.copyResize(image, width: 224, height: 224);
    final tempDir = await getTemporaryDirectory();
    final saveFile = File('${tempDir.path}/spectrogram.png');
    await saveFile.writeAsBytes(img.encodePng(resized));
    return saveFile;
  }

  static List<double> _wavToFloat(Uint8List bytes) {
    List<double> floats = [];
    for (int i = 44; i < bytes.length - 1; i += 2) {
      int sample = bytes[i] | (bytes[i + 1] << 8);
      if (sample > 32767) sample -= 65536;
      floats.add(sample / 32768.0);
    }
    return floats;
  }
}
