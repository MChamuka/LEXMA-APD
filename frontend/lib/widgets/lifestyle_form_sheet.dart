import 'package:flutter/material.dart';

class LifestyleFormSheet extends StatefulWidget {
  final void Function(Map<String, double> numericals, Map<String, int> cats)
      onAnalyze;

  const LifestyleFormSheet({super.key, required this.onAnalyze});

  @override
  State<LifestyleFormSheet> createState() => _LifestyleFormSheetState();
}

class _LifestyleFormSheetState extends State<LifestyleFormSheet> {
  final _formKey = GlobalKey<FormState>();

  // Default Categorical Values (Initialized to 0 / No / Male / Caucasian)
  final Map<String, int> _cats = {
    'gender': 0,
    'ethnicity': 0,
    'education': 0,
    'smoking': 0,
    'hypertension': 0,
    'diabetes': 0,
    'depression': 0,
  };

  // Controllers for Numerical Data
  final TextEditingController _ageCtrl = TextEditingController();
  final TextEditingController _bmiCtrl = TextEditingController();
  final TextEditingController _alcoholCtrl = TextEditingController();
  final TextEditingController _activityCtrl = TextEditingController();
  final TextEditingController _dietCtrl = TextEditingController();
  final TextEditingController _sleepCtrl = TextEditingController();
  final TextEditingController _tasksCtrl = TextEditingController();

  @override
  void dispose() {
    _ageCtrl.dispose();
    _bmiCtrl.dispose();
    _alcoholCtrl.dispose();
    _activityCtrl.dispose();
    _dietCtrl.dispose();
    _sleepCtrl.dispose();
    _tasksCtrl.dispose();
    super.dispose();
  }

  void _submit() {
    if (_formKey.currentState!.validate()) {
      // Gather all validated numericals
      Map<String, double> numericals = {
        'Age': double.parse(_ageCtrl.text),
        'BMI': double.parse(_bmiCtrl.text),
        'Alcohol': double.parse(_alcoholCtrl.text),
        'Activity': double.parse(_activityCtrl.text),
        'Diet': double.parse(_dietCtrl.text),
        'Sleep': double.parse(_sleepCtrl.text),
        'Tasks': double.parse(_tasksCtrl.text),
      };

      // Send the clean data back to the HomePage
      widget.onAnalyze(numericals, _cats);
    }
  }

  // --- UI Helpers ---
  Widget _buildSectionHeader(String title, IconData icon) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12.0),
      child: Row(
        children: [
          Icon(icon, color: Colors.blue, size: 20),
          const SizedBox(width: 8),
          Text(title,
              style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: Colors.black87)),
        ],
      ),
    );
  }

  Widget _buildNumberField(
      String label, TextEditingController controller, double min, double max) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: TextFormField(
        controller: controller,
        keyboardType: TextInputType.number,
        decoration: InputDecoration(
          labelText: label,
          hintText: "Range: $min - $max",
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        ),
        validator: (value) {
          if (value == null || value.isEmpty) return 'Required';
          double? val = double.tryParse(value);
          if (val == null) return 'Must be a number';
          if (val < min || val > max)
            return 'Enter a value between $min and $max';
          return null;
        },
      ),
    );
  }

  Widget _buildDropdown(String label, String key, Map<int, String> options) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: DropdownButtonFormField<int>(
        value: _cats[key],
        decoration: InputDecoration(
          labelText: label,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        ),
        items: options.entries
            .map((e) => DropdownMenuItem(value: e.key, child: Text(e.value)))
            .toList(),
        onChanged: (val) {
          if (val != null) setState(() => _cats[key] = val);
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final yesNoOptions = {0: "No", 1: "Yes"};

    return Container(
      height: MediaQuery.of(context).size.height * 0.85,
      padding: const EdgeInsets.all(20),
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.only(
            topLeft: Radius.circular(24), topRight: Radius.circular(24)),
      ),
      child: Form(
        key: _formKey,
        child: Column(
          children: [
            // Drag Handle
            Container(
              width: 40,
              height: 5,
              margin: const EdgeInsets.only(bottom: 20),
              decoration: BoxDecoration(
                  color: Colors.grey.shade300,
                  borderRadius: BorderRadius.circular(10)),
            ),
            const Text("Patient Lifestyle Data",
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            const SizedBox(height: 16),
            Expanded(
              child: ListView(
                children: [
                  // --- DEMOGRAPHICS ---
                  _buildSectionHeader("Demographics", Icons.person),
                  _buildNumberField("Age", _ageCtrl, 60, 90),
                  _buildDropdown("Gender", "gender", {0: "Male", 1: "Female"}),
                  _buildDropdown("Ethnicity", "ethnicity", {
                    0: "Caucasian",
                    1: "African American",
                    2: "Asian",
                    3: "Other"
                  }),
                  _buildDropdown("Education", "education", {
                    0: "None",
                    1: "High School",
                    2: "Bachelor's",
                    3: "Higher"
                  }),

                  // --- LIFESTYLE FACTORS ---
                  _buildSectionHeader(
                      "Lifestyle Factors", Icons.directions_run),
                  _buildNumberField("BMI", _bmiCtrl, 15, 40),
                  _buildDropdown("Smoking History", "smoking", yesNoOptions),
                  _buildNumberField(
                      "Alcohol (Units/Week)", _alcoholCtrl, 0, 20),
                  _buildNumberField(
                      "Physical Activity (Hours/Week)", _activityCtrl, 0, 10),
                  _buildNumberField("Diet Quality Score", _dietCtrl, 0, 10),
                  _buildNumberField("Sleep Quality Score", _sleepCtrl, 4, 10),

                  // --- MEDICAL HISTORY ---
                  _buildSectionHeader(
                      "Medical & Cognitive", Icons.medical_services),
                  _buildDropdown("Hypertension", "hypertension", yesNoOptions),
                  _buildDropdown("Diabetes", "diabetes", yesNoOptions),
                  _buildDropdown("Depression", "depression", yesNoOptions),
                  // NOTE: Based on your previous Dart code, 'Tasks' was treated as a
                  // numerical input for normalization. Set here as a 0-10 score.
                  _buildNumberField(
                      "Difficulty Completing Tasks Score", _tasksCtrl, 0, 10),
                ],
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              height: 50,
              child: ElevatedButton(
                onPressed: _submit,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                ),
                child: const Text("Analyze Patient Data",
                    style:
                        TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              ),
            )
          ],
        ),
      ),
    );
  }
}
