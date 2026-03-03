import 'dart:typed_data';
import 'dart:io';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart'; // NEW (haptics)
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

// --- SERVICES ---
import '../services/lifestyle_service.dart';
import '../services/audio_sevice.dart';
import '../services/audio_Xai_service.dart';
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

  // --- XAI RESULTS ---
  List<MapEntry<String, double>> _topFactors = [];
  List<String> _audioBiomarkers = [];
  List<String> _faceBiomarkers = [];

  // ===== UI (NEW) =====
  int _currentStep = 0; // 0..4
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

  List<_ModalityUI> _mods() => [
        _ModalityUI(
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
        _ModalityUI(
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
        _ModalityUI(
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
        _ModalityUI(
          index: 3,
          stepName: "MRI",
          title: "Brain Scan",
          subtitle: "Upload MRI scan image",
          icon: Icons.image_search,
          primaryButton: "Upload Scan",
          type: 'image',
          model: _dementiaModel,
          explainTitle: "MRI Explanation",
          explainBody:
              "After analysis, the diagnosis result will be shown. (You can add MRI biomarkers later.)",
        ),
        _ModalityUI(
          index: 4,
          stepName: "Lifestyle",
          title: "Lifestyle Check",
          subtitle: "Answer a few questions (guided form)",
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
      String path, String labelPath, int numberOfClasses) async {
    try {
      stderr.writeln("🚀 ATTEMPTING LOAD: $path with $numberOfClasses classes");
      final model = await PytorchLite.loadClassificationModel(
        path,
        224,
        224,
        3,
        labelPath: labelPath,
      );
      return model;
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
    } catch (globalError) {
      stderr.writeln("💣 GLOBAL CRASH: $globalError");
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
          // FACE ANALYSIS
          imageBytes = await FaceImageProcessor.process(originalFile);
          String prediction = await model.getImagePrediction(imageBytes);

          // XAI heatmap
          File xaiImage =
              await FaceXAIService.generateHeatmap(originalFile, prediction);

          setState(() {
            _images[0] = xaiImage;
            _results[0] = prediction;
            _faceBiomarkers = FaceXAIService.getFaceBiomarkers(prediction);
          });
        } else {
          // MRI / Other images
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
          // 1. Generate spectrogram for model
          File rawSpec = await AudioService.generateV2Spectrogram(path);

          // 2. Predict
          Uint8List bytes = await rawSpec.readAsBytes();
          String prediction = await _audioModel!.getImagePrediction(bytes);

          // 3. XAI heatmap
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

  List<double> _buildInputRow(Map<String, double> numericals) {
    List<double> row = [];
    row.addAll(_oneHot(gender, 2));
    row.addAll(_oneHot(ethnicity, 4));
    row.addAll(_oneHot(education, 4));
    row.addAll(_oneHot(smoking, 2));
    row.addAll(_oneHot(hypertension, 2));
    row.addAll(_oneHot(diabetes, 2));
    row.addAll(_oneHot(depression, 2));

    row.add(DataConfig.normalize('Age', numericals['Age']!));
    row.add(DataConfig.normalize('BMI', numericals['BMI']!));
    row.add(DataConfig.normalize('Alcohol', numericals['Alcohol']!));
    row.add(DataConfig.normalize('Activity', numericals['Activity']!));
    row.add(DataConfig.normalize('Diet', numericals['Diet']!));
    row.add(DataConfig.normalize('Sleep', numericals['Sleep']!));
    row.add(DataConfig.normalize('Tasks', numericals['Tasks']!));
    return row;
  }

  void _runLifestyleAnalysis() {
    if (_lifestyleInterpreter == null) return;
    try {
      Map<String, double> rawNumerical = {
        'Age': double.tryParse(ageController.text) ?? 0.0,
        'BMI': double.tryParse(bmiController.text) ?? 0.0,
        'Alcohol': double.tryParse(alcoholController.text) ?? 0.0,
        'Activity': double.tryParse(activityController.text) ?? 0.0,
        'Diet': double.tryParse(dietController.text) ?? 0.0,
        'Sleep': double.tryParse(sleepController.text) ?? 0.0,
        'Tasks': double.tryParse(tasksController.text) ?? 0.0,
      };

      List<double> inputRow = _buildInputRow(rawNumerical);
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

      // --- MAPPING ---
      String diag;
      if (maxIndex == 0) {
        diag = "Parkinson's";
      } else if (maxIndex == 1) {
        diag = "Alzheimer's";
      } else {
        diag = "Healthy";
      }

      _topFactors.clear();
      if (maxIndex != 2) {
        _calculateXAI(rawNumerical, probs, maxIndex);
      }

      setState(
          () => _results[4] = "$diag (${(maxVal * 100).toStringAsFixed(1)}%)");
    } catch (e) {
      setState(() => _results[4] = "Error");
    }
    Navigator.pop(context);
  }

  void _calculateXAI(Map<String, double> rawNumerical, List<double> baseProbs,
      int targetClass) {
    Map<String, double> impactScores = {};

    rawNumerical.forEach((key, val) {
      Map<String, double> perturbed = Map.from(rawNumerical);
      perturbed[key] = val + (DataConfig.stds[key] ?? 1.0);
      List<double> perturbedInput = _buildInputRow(perturbed);
      var output = List.filled(1 * 3, 0.0).reshape([1, 3]);
      _lifestyleInterpreter!.run([perturbedInput], output);
      impactScores[key] =
          (baseProbs[targetClass] - output[0][targetClass]).abs();
    });

    double baseHypScore =
        _testCategoricalFlip(rawNumerical, targetClass, flipHypertension: true);
    impactScores['High BP'] = (baseProbs[targetClass] - baseHypScore).abs();

    double baseSmokeScore =
        _testCategoricalFlip(rawNumerical, targetClass, flipSmoking: true);
    impactScores['Smoking'] = (baseProbs[targetClass] - baseSmokeScore).abs();

    var sortedEntries = impactScores.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    _topFactors = sortedEntries.take(3).toList();
  }

  double _testCategoricalFlip(Map<String, double> nums, int targetClass,
      {bool flipHypertension = false, bool flipSmoking = false}) {
    int oldHyp = hypertension;
    int oldSmoke = smoking;
    if (flipHypertension) hypertension = (hypertension == 0) ? 1 : 0;
    if (flipSmoking) smoking = (smoking == 0) ? 1 : 0;
    List<double> input = _buildInputRow(nums);
    var output = List.filled(1 * 3, 0.0).reshape([1, 3]);
    _lifestyleInterpreter!.run([input], output);
    hypertension = oldHyp;
    smoking = oldSmoke;
    return output[0][targetClass];
  }

  // ===========================
  //          UI BUILD
  // ===========================
  @override
  Widget build(BuildContext context) {
    return Theme(
      data: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.blue,
        scaffoldBackgroundColor: const Color(0xFFF7F8FA),
        textTheme: Theme.of(context).textTheme.apply(
              bodyColor: const Color(0xFF1F2937),
              displayColor: const Color(0xFF1F2937),
            ),
      ),
      child: Scaffold(
        appBar: AppBar(
          title: const Text("LEXMA-APD"),
          centerTitle: true,
          actions: [
            IconButton(
              tooltip: "Help",
              icon: const Icon(Icons.help_outline),
              onPressed: () => setState(() => _showHelp = !_showHelp),
            ),
          ],
        ),
        body: !_areModelsLoaded
            ? _buildLoadingScreen()
            : SafeArea(
                child: Column(
                  children: [
                    if (_showHelp) _buildHelpBanner(),
                    _buildStepStrip(),
                    Expanded(
                      child: ListView(
                        padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
                        children: [
                          _buildOverviewCard(),
                          const SizedBox(height: 12),
                          ..._mods().map((m) => _buildStepCard(m)).toList(),
                          const SizedBox(height: 10),
                          _buildFooterNote(),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
      ),
    );
  }

  Widget _buildLoadingScreen() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: const [
            SizedBox(
              width: 46,
              height: 46,
              child: CircularProgressIndicator(strokeWidth: 4),
            ),
            SizedBox(height: 16),
            Text(
              "Loading AI models…",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
            ),
            SizedBox(height: 6),
            Text(
              "Please wait a moment",
              style: TextStyle(color: Colors.grey),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHelpBanner() {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.blue.withOpacity(0.08),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.blue.withOpacity(0.25)),
      ),
      child: const Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline, size: 18),
          SizedBox(width: 10),
          Expanded(
            child: Text(
              "Follow the steps from left to right. Each step shows a clear result and an explanation panel to help understand why the AI decided that.",
              style: TextStyle(fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStepStrip() {
    final mods = _mods();
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 10),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            blurRadius: 10,
            color: Colors.black.withOpacity(0.06),
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: List.generate(mods.length, (i) {
          final done = _isStepDone(i);
          final active = _currentStep == i;
          return Expanded(
            child: InkWell(
              borderRadius: BorderRadius.circular(14),
              onTap: () => _setStep(i),
              child: Container(
                padding: const EdgeInsets.symmetric(vertical: 10),
                decoration: BoxDecoration(
                  color: active
                      ? Colors.blue.withOpacity(0.08)
                      : Colors.transparent,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      done ? Icons.check_circle : Icons.circle_outlined,
                      size: 18,
                      color: done
                          ? Colors.green
                          : (active ? Colors.blue : Colors.grey),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      mods[i].stepName,
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: active ? FontWeight.w700 : FontWeight.w500,
                        color: active ? Colors.blue : Colors.grey[700],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        }),
      ),
    );
  }

  Widget _buildOverviewCard() {
    final doneCount =
        List.generate(5, (i) => _isStepDone(i)).where((x) => x).length;
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            blurRadius: 12,
            color: Colors.black.withOpacity(0.06),
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 46,
            height: 46,
            decoration: BoxDecoration(
              color: Colors.blue.withOpacity(0.10),
              borderRadius: BorderRadius.circular(14),
            ),
            child: const Icon(Icons.medical_information, color: Colors.blue),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  "Guided Screening",
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800),
                ),
                const SizedBox(height: 2),
                Text(
                  "Completed $doneCount / 5 steps",
                  style: TextStyle(color: Colors.grey[700], fontSize: 13),
                ),
              ],
            ),
          ),
          TextButton.icon(
            onPressed: () => _setStep((_currentStep + 1).clamp(0, 4)),
            icon: const Icon(Icons.arrow_forward),
            label: const Text("Next"),
          )
        ],
      ),
    );
  }

  Widget _buildStepCard(_ModalityUI m) {
    final idx = m.index;
    final result = _results[idx];
    final color = _resultColor(result);

    final isMissing = (m.type != 'lifestyle') && (m.model == null);

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            blurRadius: 12,
            color: Colors.black.withOpacity(0.06),
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: Colors.blue.withOpacity(0.10),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(m.icon, color: Colors.blue),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      m.title,
                      style: const TextStyle(
                          fontSize: 16, fontWeight: FontWeight.w800),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      m.subtitle,
                      style: TextStyle(fontSize: 12, color: Colors.grey[700]),
                    ),
                  ],
                ),
              ),
              _StatusPill(
                icon: _badgeIcon(result),
                text: _resultBadgeText(result),
                color: color,
              ),
            ],
          ),

          const SizedBox(height: 12),

          // Preview area
          if (m.type == 'image' || m.type == 'audio')
            _PreviewBox(
              child: (_images[idx] != null)
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(14),
                      child: Image.file(_images[idx]!, fit: BoxFit.cover),
                    )
                  : Center(
                      child: Icon(
                        m.type == 'audio' ? Icons.graphic_eq : m.icon,
                        size: 42,
                        color: Colors.grey[500],
                      ),
                    ),
            )
          else if (m.type == 'spiral')
            _PreviewBox(
              child: Center(
                child: Icon(Icons.gesture, size: 50, color: Colors.grey[500]),
              ),
            ),

          const SizedBox(height: 10),

          // Result panel
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: color.withOpacity(0.08),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: color.withOpacity(0.35)),
            ),
            child: Row(
              children: [
                Icon(Icons.assessment_outlined, color: color),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    _shortResult(result),
                    style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w800,
                        color: color),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 10),

          // Explanation panel
          _buildExplanationPanel(m),

          const SizedBox(height: 12),

          // Big primary button
          SizedBox(
            width: double.infinity,
            height: 48,
            child: ElevatedButton.icon(
              onPressed: isMissing
                  ? null
                  : () {
                      _setStep(idx);

                      if (m.type == 'spiral') {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (c) =>
                                SpiralTestPage(model: _spiralModel!),
                          ),
                        );
                      } else if (m.type == 'audio') {
                        _handleAudioRecording();
                      } else if (m.type == 'lifestyle') {
                        _openLifestyleForm();
                      } else {
                        _pickImage(idx, m.model);
                      }
                    },
              icon: Icon(isMissing ? Icons.warning_amber : m.icon),
              label: Text(isMissing ? "Model load failed" : m.primaryButton),
              style: ElevatedButton.styleFrom(
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14)),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildExplanationPanel(_ModalityUI m) {
    final idx = m.index;

    List<Widget> chips = [];

    if (idx == 0 && _faceBiomarkers.isNotEmpty) {
      chips = _faceBiomarkers
          .map((b) => _SmallChip(text: b, icon: Icons.visibility))
          .toList();
    } else if (idx == 2 && _audioBiomarkers.isNotEmpty) {
      chips = _audioBiomarkers
          .map((b) => _SmallChip(text: b, icon: Icons.multitrack_audio))
          .toList();
    } else if (idx == 4 && _topFactors.isNotEmpty) {
      chips = _topFactors
          .map((e) => _SmallChip(text: e.key, icon: Icons.analytics))
          .toList();
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFF7F8FA),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.black.withOpacity(0.06)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            m.explainTitle,
            style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w800),
          ),
          const SizedBox(height: 6),
          if (chips.isEmpty)
            Text(
              m.explainBody ?? "Run this step to see explanations.",
              style: TextStyle(fontSize: 12, color: Colors.grey[700]),
            )
          else
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: chips,
            ),
        ],
      ),
    );
  }

  Widget _buildFooterNote() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 8),
      child: Text(
        "Note: LEXMA-APD provides screening support with explainable insights. It does not replace a clinical diagnosis.",
        style: TextStyle(fontSize: 12, color: Colors.grey[700]),
        textAlign: TextAlign.center,
      ),
    );
  }

  // ===========================
  //   LIFESTYLE FORM (NEW UI)
  // ===========================
  void _openLifestyleForm() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) {
        return Container(
          padding: EdgeInsets.only(
            left: 16,
            right: 16,
            top: 14,
            bottom: MediaQuery.of(context).viewInsets.bottom + 16,
          ),
          decoration: const BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.vertical(top: Radius.circular(22)),
          ),
          child: SafeArea(
            top: false,
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 42,
                        height: 42,
                        decoration: BoxDecoration(
                          color: Colors.blue.withOpacity(0.10),
                          borderRadius: BorderRadius.circular(14),
                        ),
                        child:
                            const Icon(Icons.monitor_heart, color: Colors.blue),
                      ),
                      const SizedBox(width: 12),
                      const Expanded(
                        child: Text(
                          "Lifestyle Assessment",
                          style: TextStyle(
                              fontSize: 16, fontWeight: FontWeight.w900),
                        ),
                      ),
                      IconButton(
                        onPressed: () => Navigator.pop(context),
                        icon: const Icon(Icons.close),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    "Fill the fields below, then tap Analyze.",
                    style: TextStyle(color: Colors.grey[700], fontSize: 12),
                  ),
                  const SizedBox(height: 14),
                  _buildDropdownLarge(
                      "Gender", ["Female", "Male"], (v) => gender = v),
                  _buildDropdownLarge(
                      "Ethnicity",
                      ["Caucasian", "African", "Asian", "Other"],
                      (v) => ethnicity = v),
                  _buildDropdownLarge(
                      "Education",
                      ["None", "High School", "Bachelor", "Higher"],
                      (v) => education = v),
                  _buildDropdownLarge(
                      "Smoking", ["No", "Yes"], (v) => smoking = v),
                  _buildDropdownLarge(
                      "High BP", ["No", "Yes"], (v) => hypertension = v),
                  _buildDropdownLarge(
                      "Diabetes", ["No", "Yes"], (v) => diabetes = v),
                  _buildDropdownLarge(
                      "Depression", ["No", "Yes"], (v) => depression = v),
                  const SizedBox(height: 10),
                  _buildNumFieldLarge("Age", ageController),
                  _buildNumFieldLarge("BMI", bmiController),
                  _buildNumFieldLarge("Alcohol", alcoholController),
                  _buildNumFieldLarge("Exercise", activityController),
                  _buildNumFieldLarge("Diet", dietController),
                  _buildNumFieldLarge("Sleep", sleepController),
                  _buildNumFieldLarge("Tasks", tasksController),
                  const SizedBox(height: 14),
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton(
                          onPressed: () => Navigator.pop(context),
                          style: OutlinedButton.styleFrom(
                            minimumSize: const Size.fromHeight(48),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(14),
                            ),
                          ),
                          child: const Text("Cancel"),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton(
                          onPressed: _runLifestyleAnalysis,
                          style: ElevatedButton.styleFrom(
                            minimumSize: const Size.fromHeight(48),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(14),
                            ),
                          ),
                          child: const Text("Analyze"),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildDropdownLarge(
      String label, List<String> items, Function(int) onChanged) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: DropdownButtonFormField<String>(
        decoration: InputDecoration(
          labelText: label,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(14)),
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
        ),
        items: items
            .asMap()
            .entries
            .map((e) =>
                DropdownMenuItem(value: e.key.toString(), child: Text(e.value)))
            .toList(),
        onChanged: (val) => onChanged(int.parse(val!)),
      ),
    );
  }

  Widget _buildNumFieldLarge(String label, TextEditingController controller) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: TextField(
        controller: controller,
        keyboardType: TextInputType.number,
        decoration: InputDecoration(
          labelText: label,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(14)),
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
        ),
      ),
    );
  }
}

// ===========================
//     SMALL UI CLASSES
// ===========================
class _ModalityUI {
  final int index;
  final String stepName;
  final String title;
  final String subtitle;
  final IconData icon;
  final String primaryButton;
  final String type; // 'image' | 'audio' | 'spiral' | 'lifestyle'
  final ClassificationModel? model;

  final String explainTitle;
  final String? explainBody;

  _ModalityUI({
    required this.index,
    required this.stepName,
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.primaryButton,
    required this.type,
    required this.model,
    required this.explainTitle,
    required this.explainBody,
  });
}

class _PreviewBox extends StatelessWidget {
  final Widget child;
  const _PreviewBox({required this.child});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 140,
      width: double.infinity,
      decoration: BoxDecoration(
        color: const Color(0xFFF1F3F6),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.black.withOpacity(0.06)),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: child,
      ),
    );
  }
}

class _StatusPill extends StatelessWidget {
  final IconData icon;
  final String text;
  final Color color;

  const _StatusPill(
      {required this.icon, required this.text, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: color.withOpacity(0.10),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withOpacity(0.35)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: color),
          const SizedBox(width: 6),
          Text(
            text,
            style: TextStyle(
                fontSize: 12, fontWeight: FontWeight.w800, color: color),
          ),
        ],
      ),
    );
  }
}

class _SmallChip extends StatelessWidget {
  final String text;
  final IconData icon;
  const _SmallChip({required this.text, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.black.withOpacity(0.08)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: Colors.grey[700]),
          const SizedBox(width: 6),
          Text(
            text,
            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700),
          ),
        ],
      ),
    );
  }
}

// ===========================
//  YOUR EXISTING UTIL CLASSES
// ===========================
class FaceImageProcessor {
  static Future<Uint8List> process(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    img.Image? original = img.decodeImage(bytes);
    if (original == null) throw Exception("Invalid Image");

    // 1. Resize to 224x224
    img.Image resized = img.copyResize(original, width: 224, height: 224);

    // 2. Grayscale + Blur
    img.Image gray = img.grayscale(resized);
    img.Image blurred = img.gaussianBlur(gray, radius: 1);

    // 3. Encode
    return Uint8List.fromList(img.encodePng(blurred));
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
