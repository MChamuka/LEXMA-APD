import 'package:flutter/material.dart';
import 'package:pytorch_lite/pytorch_lite.dart';

class ModalityUI {
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

  ModalityUI({
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
