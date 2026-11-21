# EcoVision — project notes

## Problem statement
Waste sorting is easier when common waste items can be identified consistently. EcoVision explores whether transfer learning can classify images into six broad waste categories and provide a simple educational disposal prompt.

## Aim
Build and evaluate an image-classification prototype for cardboard, glass, metal, paper, plastic and trash.

## Proposed objectives
1. Prepare and inspect a labelled image dataset.
2. Train a MobileNetV2 transfer-learning classifier.
3. Evaluate performance on held-out data.
4. Build a simple image-upload interface.
5. Document limitations and possible improvements.

## Suggested evaluation
Report accuracy, macro precision, macro recall, macro F1-score and a confusion matrix. Include per-class metrics because class imbalance can hide poor performance on minority classes.

## Suggested future work
Grad-CAM explainability, object detection for multiple items, unknown-class rejection, and city-specific recycling guidance.
