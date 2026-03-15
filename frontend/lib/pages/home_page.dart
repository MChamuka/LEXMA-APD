import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

// --- PAGES ---
import 'spiral_test_page.dart';

// --- MODELS & WIDGETS ---
import '../models/modality_model.dart';
import '../widgets/ui_components.dart';
import '../widgets/lifestyle_form_sheet.dart';
import '../utils/face_processor.dart';

// --- SERVICES ---
import '../services/lifestyle_service.dart';
import '../services/audio_sevice.dart';
import '../services/audio_xai_service.dart';
import '../services/face_xai_service.dart';

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

  final AudioRecorder _audioRecorder = AudioRecorder();
  bool _isRecording = false;

  final List<File?> _images = [null, null, null, null, null];
  final List<String> _results = [
    "Not Analyzed",
    "Not Analyzed",
    "Not Analyzed",
    "Not Analyzed",
    "Not Analyzed"
  ];

  List<MapEntry<String, double>> _topFactors = [];
  List<String> _audioBiomarkers = [];
  List<String> _faceBiomarkers = [];

  int _currentStep = 0;
  bool _showHelp = false;

  void _setStep(int i) {
    setState(() => _currentStep = i.clamp(0, 4));
    HapticFeedback.selectionClick();
  }

  String _shortResult(String s) =>
      (s.length > 28) ? "${s.substring(0, 28)}..." : s;

  Color _resultColor(String r) {
    final t = r.toLowerCase();
    if (t.contains("healthy")) return Colors.green;
    if (t.contains("alzheimer")) return Colors.redAccent;
    if (t.contains("parkinson")) return Colors.orangeAccent;
    if (t.contains("error")) return Colors.red;
    if (t.contains("analyzing") || t.contains("recording")) {
      return Colors.blueGrey;
    }
    return Colors.grey;
  }

  String _resultBadgeText(String r) {
    final t = r.toLowerCase();
    if (t.contains("not analyzed")) return "Not done";
    if (t.contains("analyzing")) return "Working";
    if (t.contains("recording")) return "Recording";
    if (t.contains("error")) return "Error";
    return "Done";
  }

  IconData _badgeIcon(String r) {
    final t = r.toLowerCase();
    if (t.contains("not analyzed")) return Icons.radio_button_unchecked;
    if (t.contains("analyzing")) return Icons.hourglass_bottom;
    if (t.contains("recording")) return Icons.mic;
    if (t.contains("error")) return Icons.error_outline;
    return Icons.check_circle_outline;
  }

  bool _isStepDone(int i) =>
      !_results[i].toLowerCase().contains("not analyzed");

  List<ModalityUI> _mods() => [
        ModalityUI(
          index: 0,
          stepName: "Face",
          title: "Face Check",
          subtitle: "Upload a face photo for analysis",
          icon: Icons.face_retouching_natural,
          primaryButton: "Upload Photo",
          type: 'image',
          model: _faceModel,
          explainTitle: "Face Explanation",
          explainBody: _faceBiomarkers.isEmpty
              ? "After analysis, key facial biomarkers will appear here."
              : null,
        ),
        ModalityUI(
          index: 1,
          stepName: "Spiral",
          title: "Spiral Test",
          subtitle: "Draw the spiral with guided steps",
          icon: Icons.gesture,
          primaryButton: "Start Spiral Test",
          type: 'spiral',
          model: _spiralModel,
          explainTitle: "Spiral Explanation",
          explainBody: "This test helps detect motor-control patterns.",
        ),
        ModalityUI(
          index: 2,
          stepName: "Voice",
          title: "Voice Check",
          subtitle: "Record a short voice sample",
          icon: Icons.mic,
          primaryButton: _isRecording ? "Stop Recording" : "Record Voice",
          type: 'audio',
          model: _audioModel,
          explainTitle: "Voice Explanation",
          explainBody: _audioBiomarkers.isEmpty
              ? "After analysis, voice biomarkers will appear here."
              : null,
        ),
        ModalityUI(
          index: 3,
          stepName: "MRI",
          title: "Brain Scan",
          subtitle: "Upload MRI scan image",
          icon: Icons.image_search,
          primaryButton: "Upload Scan",
          type: 'image',
          model: _dementiaModel,
          explainTitle: "MRI Explanation",
          explainBody: "After analysis, the diagnosis result will be shown.",
        ),
        ModalityUI(
          index: 4,
          stepName: "Lifestyle",
          title: "Lifestyle Check",
          subtitle: "Answer a few questions",
          icon: Icons.monitor_heart,
          primaryButton: "Enter Patient Data",
          type: 'lifestyle',
          model: null,
          explainTitle: "Lifestyle Explanation",
          explainBody: _topFactors.isEmpty
              ? "After analysis, top contributing factors will show here."
              : null,
        ),
      ];

  @override
  void initState() {
    super.initState();
    _loadAllModels();
  }

  @override
  void dispose() {
    _audioRecorder.dispose();
    super.dispose();
  }

  Future<ClassificationModel?> loadSafePytorch(
      String path, String labelPath, int numberOfClasses) async {
    try {
      return await PytorchLite.loadClassificationModel(path, 224, 224, 3,
          labelPath: labelPath);
    } catch (e) {
      stderr.writeln("❌ LOAD FAILED: $path - $e");
      return null;
    }
  }

  Future<void> _loadAllModels() async {
    try {
      _faceModel = await loadSafePytorch(
          "assets/models/PDFace.ptl", "assets/PDFace_labels.txt", 2);
      _spiralModel = await loadSafePytorch(
          "assets/models/spiralHandPD.ptl", "assets/spiral_labels.txt", 2);
      _audioModel = await loadSafePytorch(
          "assets/models/ewadbVoice.ptl", "assets/audio_labels.txt", 3);
      _dementiaModel = await loadSafePytorch(
          "assets/models/ADClock.ptl", "assets/labels.txt", 3);
      try {
        _lifestyleInterpreter =
            await tfl.Interpreter.fromAsset('assets/models/lifestyle.tflite');
      } catch (e) {
        stderr.writeln("❌ Failed to load Lifestyle TFLite: $e");
      }
    } finally {
      if (mounted) setState(() => _areModelsLoaded = true);
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
        File originalFile = File(image.path);

        if (index == 0) {
          imageBytes = await FaceImageProcessor.process(originalFile);
          String prediction = await model.getImagePrediction(imageBytes);
          File xaiImage =
              await FaceXAIService.generateHeatmap(originalFile, prediction);
          setState(() {
            _images[0] = xaiImage;
            _results[0] = prediction;
            _faceBiomarkers = FaceXAIService.getFaceBiomarkers(prediction);
          });
        } else {
          imageBytes = await originalFile.readAsBytes();
          String prediction = await model.getImagePrediction(imageBytes);
          setState(() => _results[index] = prediction);
        }
      } catch (e) {
        setState(() => _results[index] = "Error: $e");
      }
    }
  }

  Future<void> _handleAudioRecording() async {
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) return;

    if (!_isRecording) {
      final directory = await getTemporaryDirectory();
      String path = '${directory.path}/temp_audio.wav';
      await _audioRecorder.start(const RecordConfig(encoder: AudioEncoder.wav),
          path: path);
      setState(() {
        _isRecording = true;
        _results[2] = "Recording...";
      });
    } else {
      final path = await _audioRecorder.stop();
      setState(() => _isRecording = false);

      if (path != null && _audioModel != null) {
        setState(() => _results[2] = "Analyzing...");
        try {
          File rawSpec = await AudioService.generateV2Spectrogram(path);
          Uint8List bytes = await rawSpec.readAsBytes();
          String prediction = await _audioModel!.getImagePrediction(bytes);
          File heatmap =
              await AudioXAIService.generateHeatmap(rawSpec, prediction);
          setState(() {
            _images[2] = heatmap;
            _results[2] = prediction;
            _audioBiomarkers = AudioXAIService.getBiomarkers(prediction);
          });
        } catch (e) {
          setState(() => _results[2] = "Error: $e");
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

  List<double> _buildInputRow(
      Map<String, double> numericals, Map<String, int> cats) {
    List<double> row = [];
    row.addAll(_oneHot(cats['gender']!, 2));
    row.addAll(_oneHot(cats['ethnicity']!, 4));
    row.addAll(_oneHot(cats['education']!, 4));
    row.addAll(_oneHot(cats['smoking']!, 2));
    row.addAll(_oneHot(cats['hypertension']!, 2));
    row.addAll(_oneHot(cats['diabetes']!, 2));
    row.addAll(_oneHot(cats['depression']!, 2));

    row.add(DataConfig.normalize('Age', numericals['Age']!));
    row.add(DataConfig.normalize('BMI', numericals['BMI']!));
    row.add(DataConfig.normalize('Alcohol', numericals['Alcohol']!));
    row.add(DataConfig.normalize('Activity', numericals['Activity']!));
    row.add(DataConfig.normalize('Diet', numericals['Diet']!));
    row.add(DataConfig.normalize('Sleep', numericals['Sleep']!));
    row.add(DataConfig.normalize('Tasks', numericals['Tasks']!));
    return row;
  }

  void _runLifestyleAnalysis(
      Map<String, double> rawNumerical, Map<String, int> cats) {
    if (_lifestyleInterpreter == null) return;
    try {
      List<double> inputRow = _buildInputRow(rawNumerical, cats);
      var output = List.filled(1 * 3, 0.0).reshape([1, 3]);
      _lifestyleInterpreter!.run([inputRow], output);
      List<double> probs = List<double>.from(output[0]);

      int maxIndex = 0;
      double maxVal = probs[0];
      for (int i = 1; i < probs.length; i++) {
        if (probs[i] > maxVal) {
          maxVal = probs[i];
          maxIndex = i;
        }
      }

      String diag = (maxIndex == 0)
          ? "Parkinson's"
          : (maxIndex == 1)
              ? "Alzheimer's"
              : "Healthy";

      _topFactors.clear();
      if (maxIndex != 2) _calculateXAI(rawNumerical, cats, probs, maxIndex);

      setState(
          () => _results[4] = "$diag (${(maxVal * 100).toStringAsFixed(1)}%)");
    } catch (e) {
      setState(() => _results[4] = "Error");
    }
    Navigator.pop(context);
  }

  void _calculateXAI(Map<String, double> rawNumerical, Map<String, int> cats,
      List<double> baseProbs, int targetClass) {
    Map<String, double> impactScores = {};

    rawNumerical.forEach((key, val) {
      Map<String, double> perturbed = Map.from(rawNumerical);
      perturbed[key] = val + (DataConfig.stds[key] ?? 1.0);
      List<double> perturbedInput = _buildInputRow(perturbed, cats);
      var output = List.filled(1 * 3, 0.0).reshape([1, 3]);
      _lifestyleInterpreter!.run([perturbedInput], output);
      impactScores[key] =
          (baseProbs[targetClass] - output[0][targetClass]).abs();
    });

    double baseHypScore = _testCategoricalFlip(rawNumerical, cats, targetClass,
        flipHypertension: true);
    impactScores['High BP'] = (baseProbs[targetClass] - baseHypScore).abs();

    double baseSmokeScore = _testCategoricalFlip(
        rawNumerical, cats, targetClass,
        flipSmoking: true);
    impactScores['Smoking'] = (baseProbs[targetClass] - baseSmokeScore).abs();

    var sortedEntries = impactScores.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    _topFactors = sortedEntries.take(3).toList();
  }

  double _testCategoricalFlip(
      Map<String, double> nums, Map<String, int> cats, int targetClass,
      {bool flipHypertension = false, bool flipSmoking = false}) {
    Map<String, int> testCats = Map.from(cats);
    if (flipHypertension) {
      testCats['hypertension'] = (testCats['hypertension'] == 0) ? 1 : 0;
    }
    if (flipSmoking) testCats['smoking'] = (testCats['smoking'] == 0) ? 1 : 0;

    List<double> input = _buildInputRow(nums, testCats);
    var output = List.filled(1 * 3, 0.0).reshape([1, 3]);
    _lifestyleInterpreter!.run([input], output);
    return output[0][targetClass];
  }

  void _openLifestyleForm() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => LifestyleFormSheet(
        onAnalyze: _runLifestyleAnalysis,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Theme(
      data: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.blue,
        scaffoldBackgroundColor: const Color(0xFFF7F8FA),
      ),
      child: Scaffold(
        appBar: AppBar(
          title: const Text("LEXMA-APD"),
          centerTitle: true,
          actions: [
            IconButton(
                icon: const Icon(Icons.help_outline),
                onPressed: () => setState(() => _showHelp = !_showHelp)),
          ],
        ),
        body: !_areModelsLoaded
            ? const Center(child: CircularProgressIndicator())
            : SafeArea(
                child: Column(
                  children: [
                    if (_showHelp) _buildHelpBanner(),
                    _buildStepStrip(),
                    Expanded(
                      child: ListView(
                        padding: const EdgeInsets.all(16),
                        children: [
                          _buildOverviewCard(),
                          const SizedBox(height: 12),
                          ..._mods().map((m) => _buildStepCard(m)).toList(),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
      ),
    );
  }

  Widget _buildHelpBanner() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      padding: const EdgeInsets.all(12),
      color: Colors.blue.withOpacity(0.08),
      child: const Text("Follow the steps. Each step shows a clear result."),
    );
  }

  Widget _buildStepStrip() {
    final mods = _mods();
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      padding: const EdgeInsets.all(10),
      child: Row(
        children: List.generate(mods.length, (i) {
          final done = _isStepDone(i);
          final active = _currentStep == i;
          return Expanded(
            child: InkWell(
              onTap: () => _setStep(i),
              child: Column(
                children: [
                  Icon(done ? Icons.check_circle : Icons.circle_outlined,
                      color: active ? Colors.blue : Colors.grey),
                  Text(mods[i].stepName,
                      style: TextStyle(
                          fontSize: 11,
                          color: active ? Colors.blue : Colors.grey)),
                ],
              ),
            ),
          );
        }),
      ),
    );
  }

  Widget _buildOverviewCard() {
    return Container(
      padding: const EdgeInsets.all(14),
      color: Colors.white,
      child: Row(
        children: [
          const Icon(Icons.medical_information, color: Colors.blue),
          const SizedBox(width: 12),
          const Expanded(
              child: Text("Guided Screening",
                  style: TextStyle(fontWeight: FontWeight.bold))),
          TextButton(
              onPressed: () => _setStep((_currentStep + 1).clamp(0, 4)),
              child: const Text("Next"))
        ],
      ),
    );
  }

  Widget _buildStepCard(ModalityUI m) {
    final idx = m.index;
    final result = _results[idx];
    final color = _resultColor(result);
    final isMissing = (m.type != 'lifestyle') && (m.model == null);

    return Card(
      margin: const EdgeInsets.only(bottom: 14),
      color: Colors.white,
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(m.icon, color: Colors.blue),
                const SizedBox(width: 12),
                Expanded(
                    child: Text(m.title,
                        style: const TextStyle(fontWeight: FontWeight.bold))),
                StatusPill(
                    icon: _badgeIcon(result),
                    text: _resultBadgeText(result),
                    color: color),
              ],
            ),
            const SizedBox(height: 12),
            if (m.type == 'image' || m.type == 'audio')
              PreviewBox(
                  child: _images[idx] != null
                      ? Image.file(_images[idx]!, fit: BoxFit.cover)
                      : Center(
                          child: Icon(m.icon, size: 42, color: Colors.grey)))
            else if (m.type == 'spiral')
              const PreviewBox(
                  child: Center(
                      child:
                          Icon(Icons.gesture, size: 50, color: Colors.grey))),
            const SizedBox(height: 10),
            Text("Result: $result",
                style: TextStyle(color: color, fontWeight: FontWeight.bold)),
            _buildExplanationPanel(m),
            const SizedBox(height: 12),
            ElevatedButton.icon(
              onPressed: isMissing
                  ? null
                  : () {
                      _setStep(idx);
                      if (m.type == 'spiral') {
                        Navigator.push(
                            context,
                            MaterialPageRoute(
                                builder: (c) =>
                                    SpiralTestPage(model: _spiralModel!)));
                      } else if (m.type == 'audio') {
                        _handleAudioRecording();
                      } else if (m.type == 'lifestyle') {
                        _openLifestyleForm();
                      } else {
                        _pickImage(idx, m.model);
                      }
                    },
              icon: Icon(isMissing ? Icons.warning_amber : m.icon),
              label: Text(m.primaryButton),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildExplanationPanel(ModalityUI m) {
    List<Widget> chips = [];
    if (m.index == 0) {
      chips = _faceBiomarkers
          .map((b) => SmallChip(text: b, icon: Icons.visibility))
          .toList();
    } else if (m.index == 2)
      chips = _audioBiomarkers
          .map((b) => SmallChip(text: b, icon: Icons.multitrack_audio))
          .toList();
    else if (m.index == 4)
      chips = _topFactors
          .map((e) => SmallChip(text: e.key, icon: Icons.analytics))
          .toList();

    return Container(
      padding: const EdgeInsets.all(8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(m.explainTitle,
              style: const TextStyle(fontWeight: FontWeight.bold)),
          if (chips.isEmpty)
            Text(m.explainBody ?? "Run step to see explanations.")
          else
            Wrap(spacing: 8, children: chips),
        ],
      ),
    );
  }
}
