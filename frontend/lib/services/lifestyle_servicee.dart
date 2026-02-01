class DataConfig {
  static const Map<String, double> means = {
    'Age': 72.2828,
    'BMI': 27.4349,
    'Alcohol': 10.0399,
    'Activity': 4.9679,
    'Diet': 4.9534,
    'Sleep': 7.0241,
    'Tasks': 5.0353,
  };

  static const Map<String, double> stds = {
    'Age': 10.6943,
    'BMI': 7.2154,
    'Alcohol': 5.7223,
    'Activity': 2.8740,
    'Diet': 2.8908,
    'Sleep': 1.7584,
    'Tasks': 2.9132,
  };
  static double normalize(String key, double rawValue) {
    double mean = means[key] ?? 0.0;
    double std = stds[key] ?? 1.0;
    return (rawValue - mean) / std;
  }
}
