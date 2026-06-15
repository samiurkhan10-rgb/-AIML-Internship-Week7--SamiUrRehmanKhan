# -AIML-Internship-Week7--SamiUrRehmanKhan
uilt and compared 3 deep learning models on CIFAR-10 (60K images, 10 classes) using TensorFlow/Keras: a Dense baseline (49.68%), CNN with BatchNorm, Dropout &amp; augmentation (83.53%), and MobileNetV2 transfer learning (89.24%). Applied regularisation ablation, data augmentation, and two-phase fine-tuning.

Week 7 — Deep Learning with TensorFlow & Keras

Overview

This project focuses on image classification using deep learning techniques on the CIFAR-10 dataset. Three distinct neural network architectures were built, trained, evaluated, and compared: a Dense (Fully Connected) baseline, a Convolutional Neural Network (CNN) trained from scratch, and MobileNetV2 via transfer learning.

The project covers the complete deep learning workflow, including exploratory data analysis (EDA), data preprocessing, model architecture design, regularisation techniques, data augmentation, transfer learning with two-phase fine-tuning, model evaluation, and a final three-model benchmark comparison.


Dataset

Dataset: CIFAR-10 (built into TensorFlow/Keras)

AttributeValueTotal Images60,000Training Set50,000 imagesTest Set10,000 imagesImage Dimensions32 × 32 × 3 (RGB)Number of Classes10Images per Class5,000 (train) / 1,000 (test)Missing ValuesNone

Classes

airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck


Objectives

The main objectives of this project were:


Understand forward propagation and backpropagation mathematically
Build and train a Dense network as a performance baseline
Implement a CNN from scratch with progressive regularisation
Apply data augmentation to reduce overfitting
Leverage pre-trained ImageNet weights via MobileNetV2 transfer learning
Compare all three models on accuracy, F1, ROC-AUC, parameters, and training time
Evaluate models using confusion matrices, classification reports, and ROC curves



Technologies Used

LibraryPurposePythonCore programming languageTensorFlow 2.19Deep learning frameworkKerasHigh-level neural network APINumPyArray operations and pixel normalisationPandasResults comparison and DataFramesMatplotlibTraining history plots and visualisationsSeabornConfusion matrix heatmapsScikit-LearnClassification report, ROC-AUC, metrics


Models Implemented

1. Dense Network (Baseline)


Input: Flattened 32×32×3 image → 3,072-element vector
Architecture: Dense(512) → Dropout(0.3) → Dense(256) → Dropout(0.3) → Dense(128) → Dense(10, softmax)
Callbacks: EarlyStopping (patience=10) + ModelCheckpoint
Parameters: 1,738,890


2. CNN from Scratch

Three variants built progressively:

CNN Baseline (No Regularisation)


3 convolutional blocks: Conv2D(32) → Conv2D(64) → Conv2D(128) with MaxPooling2D


CNN + Batch Normalisation


BatchNormalization added after each Conv2D layer


CNN + BatchNorm + Dropout (Full)


Dropout(0.25) after each MaxPool + Dropout(0.5) before final Dense
Callbacks: EarlyStopping (patience=15) + ModelCheckpoint + ReduceLROnPlateau(factor=0.5, patience=7)
Parameters: 667,434


CNN + Augmentation


ImageDataGenerator: rotation ±15°, width/height shift ±10%, horizontal flip, zoom ±10%


3. MobileNetV2 Transfer Learning

Phase 1 — Feature Extraction


MobileNetV2 base fully frozen (weights="imagenet")
Custom head: GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.3) → Dense(10, softmax)
Optimiser: Adam(lr=1e-3)


Phase 2 — Fine-Tuning


Last 30 layers of base unfrozen
Optimiser: Adam(lr=1e-5) — 100× lower to prevent catastrophic forgetting
Parameters: 2,588,490



Regularisation Techniques

TechniquePurposeDropoutRandomly zeroes neurons to prevent co-adaptationBatch NormalisationNormalises activations per mini-batch; stabilises trainingEarlyStoppingHalts training when val_loss stops improvingReduceLROnPlateauHalves learning rate when training plateausData AugmentationArtificially expands training set via random transforms


Evaluation Metrics

Test Accuracy — Fraction of correctly classified test images

Macro F1-Score — Harmonic mean of precision and recall, averaged across all 10 classes equally

ROC-AUC (OvR) — One-vs-Rest multiclass AUC; measures ranking quality across all classes

Confusion Matrix — 10×10 matrix showing per-class prediction errors

Per-Class Accuracy — Individual accuracy for each of the 10 CIFAR-10 classes


Key Results

Three-Model Final Benchmark

ModelTest AccuracyMacro F1ROC-AUCParametersTrain TimeDense Network49.68%0.49090.88011,738,8901.3 minCNN + Augmentation83.53%0.83310.9859667,43418.0 minMobileNetV2 (TL)89.24%0.89220.99282,588,4906.4 min

Regularisation Ablation Study

CNN VariantTest AccuracyOverfitting GapVal LossNo Regularisation74.10%7.93%0.7281BatchNorm Only78.27%10.45%0.6796BN + Dropout85.14%6.23%0.5213

Transfer Learning Results

PhaseVal AccuracyTrainable ParamsLRPhase 1 (Feature Extraction)87.47%330,5061e-3Phase 2 (Fine-Tuning)89.55%1,856,9061e-5


CNN Per-Class Performance (Best Model: MobileNetV2)

ClassPrecisionRecallF1-Scoreairplane0.860.830.85automobile0.920.940.93bird0.820.730.77cat0.730.680.71deer0.800.830.82dog0.870.640.74frog0.710.950.81horse0.890.890.89ship0.910.930.92truck0.880.930.90


Dashboard

A complete 6-chart dashboard (week7_dashboard.png) was generated covering:


MobileNetV2 training history (Phase 1 + Phase 2 with transition marker)
10×10 Confusion matrix heatmap
Per-class accuracy comparison — all three models
Three-model accuracy and F1 bar chart
10 misclassified test images with true and predicted labels
ROC curves for all three models


Model Persistence

The best-performing model was saved and verified with a 5-image inference demo.

Generated files:


week7_best_model.keras — MobileNetV2 fine-tuned model (full architecture + weights)
week7_dashboard.png — 6-chart summary dashboard
