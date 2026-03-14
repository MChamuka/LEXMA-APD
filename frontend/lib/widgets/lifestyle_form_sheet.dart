import 'package:flutter/material.dart';

class LifestyleFormSheet extends StatefulWidget {
  final Function(Map<String, double> numericals, Map<String, int> categoricals)
      onAnalyze;

  const LifestyleFormSheet({super.key, required this.onAnalyze});

  @override
  State<LifestyleFormSheet> createState() => _LifestyleFormSheetState();
}

class _LifestyleFormSheetState extends State<LifestyleFormSheet> {
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
  void dispose() {
    ageController.dispose();
    bmiController.dispose();
    alcoholController.dispose();
    activityController.dispose();
    dietController.dispose();
    sleepController.dispose();
    tasksController.dispose();
    super.dispose();
  }

  void _submitData() {
    Map<String, double> rawNumerical = {
      'Age': double.tryParse(ageController.text) ?? 0.0,
      'BMI': double.tryParse(bmiController.text) ?? 0.0,
      'Alcohol': double.tryParse(alcoholController.text) ?? 0.0,
      'Activity': double.tryParse(activityController.text) ?? 0.0,
      'Diet': double.tryParse(dietController.text) ?? 0.0,
      'Sleep': double.tryParse(sleepController.text) ?? 0.0,
      'Tasks': double.tryParse(tasksController.text) ?? 0.0,
    };

    Map<String, int> categoricals = {
      'gender': gender,
      'ethnicity': ethnicity,
      'education': education,
      'smoking': smoking,
      'hypertension': hypertension,
      'diabetes': diabetes,
      'depression': depression,
    };

    widget.onAnalyze(rawNumerical, categoricals);
  }

  @override
  Widget build(BuildContext context) {
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
                    child: const Icon(Icons.monitor_heart, color: Colors.blue),
                  ),
                  const SizedBox(width: 12),
                  const Expanded(
                    child: Text(
                      "Lifestyle Assessment",
                      style:
                          TextStyle(fontSize: 16, fontWeight: FontWeight.w900),
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
              _buildDropdownLarge("Smoking", ["No", "Yes"], (v) => smoking = v),
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
                            borderRadius: BorderRadius.circular(14)),
                      ),
                      child: const Text("Cancel"),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: _submitData,
                      style: ElevatedButton.styleFrom(
                        minimumSize: const Size.fromHeight(48),
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14)),
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
