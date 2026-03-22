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
import '../utils/pytorch_native.dart';

// --- MODELS & WIDGETS ---
import '../models/modality_model.dart';
import '../widgets/ui_components.dart';
import '../widgets/lifestyle_form_sheet.dart';

// --- UTILS (ADDED PREPROCESSORS HERE) ---
import '../utils/face_processor.dart';
import '../utils/clock_processor.dart';
import '../utils/spiral_processor.dart'; // 🔥 ADDED YOUR NEW SPIRAL PROCESSOR

// --- SERVICES ---
import '../services/lifestyle_service.dart';
import '../services/audio_sevice.dart';
import '../services/audio_xai_service.dart';
import '../services/face_xai_service.dart';
import '../services/clock_xai_service.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  // --- MODELS ---
  ClassificationModel? _spiralModel;
  ClassificationModel? _audioModel;
  ClassificationModel? _dementiaModel;
  tfl.Interpreter? _lifestyleInterpreter;
  bool _areModelsLoaded = false;

  final AudioRecorder _audioRecorder = AudioRecorder();
  bool _isRecording = false;

  final List<File?> _images = List.filled(5, null);
  final List<String> _results = List.filled(5, "Not Analyzed");
  final List<double?> _probs = List.filled(5, null);

  List<MapEntry<String, double>> _topFactors = [];
  List<String> _audioBiomarkers = [];
  List<String> _faceBiomarkers = [];

  int _currentStep = 0;
  bool _showHelp = false;

  void _setStep(int i) {
    setState(() => _currentStep = i.clamp(0, 4));
    HapticFeedback.selectionClick();
  }

  // --- UI HELPERS ---
  Color _resultColor(String r) {
    final t = r.toLowerCase();
    if (t.contains("healthy")) return Colors.green;
    if (t.contains("alzheimer")) return Colors.redAccent;
    if (t.contains("parkinson")) return Colors.orangeAccent;
    if (t.contains("error")) return Colors.red;
    if (t.contains("analyzing") || t.contains("recording"))
      return Colors.blueGrey;
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

  // --- MODALITIES CONFIG ---
  List<ModalityUI> _mods() => [
        ModalityUI(
          index: 0,
          stepName: "Face",
          title: "Face Check",
          subtitle: "Upload a face photo for analysis",
          icon: Icons.face_retouching_natural,
          primaryButton: "Upload Photo",
          type: 'image',
          model: "native_bridge",
          explainTitle: "Face Explanation",
          explainBody: _faceBiomarkers.isEmpty
              ? "After analysis, key facial biomarkers will appear here."
              : null,
        ),
        // 🔥 CHANGED SPIRAL MODALITY TO BE AN IMAGE UPLOAD
        ModalityUI(
          index: 1,
          stepName: "Spiral",
          title: "Spiral Test",
          subtitle: "Upload a photo of your drawn spiral",
          icon: Icons.draw,
          primaryButton: "Upload Spiral Photo",
          type: 'image', // Now treated as standard image upload
          model: _spiralModel,
          explainTitle: "Spiral Explanation",
          explainBody:
              "This test analyzes motor-control patterns from your pen strokes.",
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
          stepName: "CDT",
          title: "Clock Drawing",
          subtitle: "Upload ClockDrawing image",
          icon: Icons.image_search,
          primaryButton: "Upload CDT",
          type: 'image',
          model: _dementiaModel,
          explainTitle: "ClockDrawing Explanation",
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
      stderr.writeln("MODEL LOAD FAILED! Path: $path | Error: $e");
      return null;
    }
  }

  Future<void> _loadAllModels() async {
    try {
      _spiralModel = await loadSafePytorch(
          "assets/models/handpd_spiral_model_mobile_fixed.ptl",
          "assets/spiral_labels.txt",
          2);
      _audioModel = await loadSafePytorch(
          "assets/models/finalADPDVoice.ptl", "assets/audio_labels.txt", 3);
      _dementiaModel = await loadSafePytorch(
          "assets/models/finalNhats_fixed.ptl", "assets/labels.txt", 2);
      _lifestyleInterpreter =
          await tfl.Interpreter.fromAsset('assets/models/lifestyle.tflite');
    } finally {
      if (mounted) setState(() => _areModelsLoaded = true);
    }
  }

  Future<void> _pickImage(int index, dynamic model) async {
    if (model == null) return;
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);

    if (image != null) {
      setState(() {
        _images[index] = File(image.path);
        _results[index] = "Analyzing...";
      });

      try {
        File originalFile = File(image.path);
        String prediction = "Error";

        if (index == 0) {
          // --- FACE MODEL ---
          final faceData = await FaceImageProcessor.process(originalFile);
          if (faceData == null) {
            setState(() => _results[0] = "Error: No face detected");
            return;
          }

          List<double> features = faceData['features'];
          Map<String, List<double>> landmarks = faceData['landmarks'];

          List<double> probs = await PyTorchNative.predictFace(features);
          double score = probs[0];
          prediction = score > 0.5 ? "Parkinson's" : "Healthy";

          File xaiImage = await FaceXAIService.generateHeatmap(
              originalFile, prediction, landmarks);

          setState(() {
            _images[0] = xaiImage;
            _results[0] = prediction;
            _probs[0] = score;
            _faceBiomarkers = FaceXAIService.getFaceBiomarkers(prediction);
          });
        } else if (index == 1) {
          // 🔥 --- THE NEW SPIRAL IMAGE UPLOAD PIPELINE ---
          Uint8List processedSpiralBytes =
              await SpiralProcessor.process(originalFile);

          List<double> probs = await (model as ClassificationModel)
              .getImagePredictionList(processedSpiralBytes);

          // CAPTURE RAW OUTPUT FOR UI
          String rawArrayText = probs.toString();
          double pdProb = 0.0;

          if (probs.isNotEmpty) {
            if (probs.length >= 2) {
              pdProb = probs[1];
            } else {
              pdProb = probs[0];
            }
          }

          String diagText =
              pdProb > 0.5 ? "Parkinson's Detected" : "Healthy Pattern";

          // 🔥 ADDED RAW AND PROBABILITY TO THE UI BADGE
          prediction =
              "$diagText\nRAW: $rawArrayText\nPROB: ${(pdProb * 100).toStringAsFixed(2)}%";

          // Save the processed X-Ray so you can see OpenCV's handiwork
          final tempDir = await getTemporaryDirectory();
          File xrayFile = File(
              '${tempDir.path}/xray_spiral_${DateTime.now().millisecondsSinceEpoch}.jpg');
          await xrayFile.writeAsBytes(processedSpiralBytes);

          setState(() {
            _images[1] = xrayFile;
            _results[1] = prediction;
            _probs[1] = pdProb;
          });
        } else if (index == 3) {
          // --- CDT MODEL ---
          Uint8List processedClockBytes =
              await ClockProcessor.process(originalFile);
          List<double> probs = await (model as ClassificationModel)
              .getImagePredictionList(processedClockBytes);

          double adProb = probs[1];
          prediction = adProb > 0.5 ? "Alzheimer's" : "Healthy";
          File xaiImage =
              await ClockXAIService.generateHeatmap(originalFile, prediction);

          setState(() {
            _images[3] = xaiImage;
            _results[3] = prediction;
            _probs[3] = adProb;
          });
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

          List<double> vProbs =
              await _audioModel!.getImagePredictionList(bytes);
          double vScore = vProbs[1];
          String prediction = vScore > 0.5 ? "Parkinson's Detected" : "Healthy";

          File heatmap =
              await AudioXAIService.generateHeatmap(rawSpec, prediction);
          setState(() {
            _images[2] = heatmap;
            _results[2] = prediction;
            _probs[2] = vScore;
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
    if (selectedIndex >= 0 && selectedIndex < totalCategories)
      output[selectedIndex] = 1.0;
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
    row.addAll([
      DataConfig.normalize('Age', numericals['Age']!),
      DataConfig.normalize('BMI', numericals['BMI']!),
      DataConfig.normalize('Alcohol', numericals['Alcohol']!),
      DataConfig.normalize('Activity', numericals['Activity']!),
      DataConfig.normalize('Diet', numericals['Diet']!),
      DataConfig.normalize('Sleep', numericals['Sleep']!),
      DataConfig.normalize('Tasks', numericals['Tasks']!)
    ]);
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

      setState(() {
        _results[4] = "$diag (${(maxVal * 100).toStringAsFixed(1)}%)";
        _probs[4] = maxVal;
      });
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
    if (flipHypertension)
      testCats['hypertension'] = (testCats['hypertension'] == 0) ? 1 : 0;
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
      builder: (context) =>
          LifestyleFormSheet(onAnalyze: _runLifestyleAnalysis),
    );
  }

  Widget _buildFusionDashboard() {
    double totalWeight = 0;
    double fusedProb = 0;

    for (int i = 0; i < 5; i++) {
      if (_probs[i] != null) {
        fusedProb += _probs[i]!;
        totalWeight++;
      }
    }

    if (totalWeight == 0) {
      return Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(
          children: [
            const Icon(Icons.hub, color: Colors.blue),
            const SizedBox(width: 12),
            const Expanded(
                child: Text("Decision-Level Fusion",
                    style: TextStyle(fontWeight: FontWeight.bold))),
            TextButton(
                onPressed: () => _setStep((_currentStep + 1).clamp(0, 4)),
                child: const Text("Start"))
          ],
        ),
      );
    }

    fusedProb = fusedProb / totalWeight;
    bool isDetected = fusedProb > 0.5;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: isDetected
              ? [Colors.red.shade50, Colors.orange.shade50]
              : [Colors.green.shade50, Colors.teal.shade50],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
            color: isDetected
                ? Colors.red.withOpacity(0.3)
                : Colors.green.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.hub, color: isDetected ? Colors.red : Colors.green),
              const SizedBox(width: 8),
              const Text("MULTIMODAL FUSION ENGINE",
                  style: TextStyle(
                      fontWeight: FontWeight.w900,
                      fontSize: 13,
                      letterSpacing: 1.2)),
            ],
          ),
          const SizedBox(height: 12),
          if (_probs[0] != null) _buildFusionRow("Face Model", _probs[0]!),
          if (_probs[1] != null) _buildFusionRow("Spiral Model", _probs[1]!),
          if (_probs[2] != null) _buildFusionRow("Voice Model", _probs[2]!),
          if (_probs[3] != null) _buildFusionRow("CDT Model", _probs[3]!),
          if (_probs[4] != null) _buildFusionRow("Lifestyle Data", _probs[4]!),
          const Divider(height: 24),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text("Final Multimodal Score",
                      style: TextStyle(fontSize: 12, color: Colors.black54)),
                  Text("${(fusedProb * 100).toStringAsFixed(1)}%",
                      style: TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.w900,
                          color: isDetected ? Colors.red : Colors.green)),
                ],
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: isDetected ? Colors.red : Colors.green,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  isDetected ? "Anomaly Detected" : "Healthy Pattern",
                  style: const TextStyle(
                      color: Colors.white, fontWeight: FontWeight.bold),
                ),
              )
            ],
          )
        ],
      ),
    );
  }

  Widget _buildFusionRow(String label, double prob) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text("↳ $label",
              style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: Colors.black87)),
          Text("${(prob * 100).toStringAsFixed(1)}%",
              style:
                  const TextStyle(fontSize: 13, fontWeight: FontWeight.w800)),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Theme(
      data: ThemeData(
          useMaterial3: true,
          colorSchemeSeed: Colors.blue,
          scaffoldBackgroundColor: const Color(0xFFF7F8FA)),
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
                          _buildFusionDashboard(),
                          const SizedBox(height: 12),
                          ..._mods().map((m) => _buildStepCard(m)),
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

  Widget _buildStepCard(ModalityUI m) {
    final idx = m.index;
    final result = _results[idx];
    final color = _resultColor(result);
    final isMissing = (m.type != 'lifestyle') &&
        (m.model == null) &&
        (m.model != "native_bridge");

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
            // 🔥 REMOVED THE COMPLEX 'SPIRAL' PREVIEW BOX ROUTING
            if (m.type == 'image' || m.type == 'audio')
              PreviewBox(
                  child: _images[idx] != null
                      ? Image.file(_images[idx]!, fit: BoxFit.contain)
                      : Center(
                          child: Icon(m.icon, size: 42, color: Colors.grey))),
            const SizedBox(height: 10),
            Text("Result: $result",
                style: TextStyle(color: color, fontWeight: FontWeight.bold)),
            _buildExplanationPanel(m),
            const SizedBox(height: 12),
            ElevatedButton.icon(
              onPressed: isMissing
                  ? null
                  : () async {
                      _setStep(idx);
                      // 🔥 REMOVED THE OLD NAVIGATOR.PUSH CODE
                      if (m.type == 'audio') {
                        _handleAudioRecording();
                      } else if (m.type == 'lifestyle') {
                        _openLifestyleForm();
                      } else {
                        // THIS NOW HANDLES FACE, SPIRAL, AND CDT!
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
    if (m.index == 0)
      chips = _faceBiomarkers
          .map((b) => SmallChip(text: b, icon: Icons.visibility))
          .toList();
    else if (m.index == 2)
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
