# 🍇 Grape Segmentation using YOLO Models

## 📌 Project Overview

This research project focuses on **grape segmentation using deep learning techniques** in real-world vineyard conditions.
The aim is to improve agricultural productivity by enabling accurate detection and segmentation of grape clusters.

The study presents a **comparative analysis of YOLOv11, YOLOv12, and YOLOv26** to evaluate their performance in terms of accuracy and speed.

---

## 👨‍🎓 Author Details

* Laxmi Narayan | UniversityID 2211981212
* Kumud Sharma | UniversityID 2211981207

* **University:** Chitkara University

---

## 🏷️ Project Type

**Research Project**

---

## 🎯 Objectives

* Perform accurate grape segmentation
* Compare YOLOv11, YOLOv12, and YOLOv26 models
* Evaluate performance using standard metrics
* Identify the best model for real-world agricultural use

---

## ⚙️ Technologies Used

* Python
* YOLO (Ultralytics)
* Deep Learning
* Computer Vision
* GPU Training

---

## 🔬 Methodology

* Used a labeled dataset of vineyard images
* Applied instance segmentation techniques
* Trained all models using the same configuration for fair comparison
* Evaluated models based on accuracy and inference speed

---

## 📊 Evaluation Metrics

* Precision
* Recall
* F1-Score
* mAP@0.5
* mAP@0.5:0.95
* Inference Time

---

## 📈 Results Summary

* **YOLOv11** → Highest accuracy
* **YOLOv26** → Best balance between speed and performance
* **YOLOv12** → Lower performance due to lack of pretrained weights

### 📊 Metrics Table

| Model | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | Inference (ms/img) | Epochs |
|-------|-----------|--------|----|---------|--------------|-------------------|--------|
| YOLOv11 | 0.599 | 0.481 | 0.534 | 0.525 | 0.284 | 21.05 | 500 |
| YOLOv12 | 0.230 | 0.399 | 0.292 | 0.303 | 0.137 | 20.62 | 500 |
| YOLOv26 | 0.478 | 0.513 | 0.495 | 0.411 | 0.227 | 16.81 | 500 |

### 📉 Training Curves

![Training Curves (All Metrics)](paper_results/figures/training_curves_all_metrics.png)

![Training Curves Panel](paper_results/figures/training_curves_panel_2x2.png)

![Training Loss (Box & Seg)](paper_results/figures/training_loss_box_seg.png)

### 📊 Performance Charts

![Final Metrics Grouped Bar](paper_results/figures/final_metrics_grouped_bar.png)

![Inference Time Comparison](paper_results/figures/inference_time_bar.png)

### 🖼️ Qualitative Results

![Qualitative Comparison Grid](paper_results/figures/qualitative_comparison_grid.png)

### 🔍 Sample Predictions

**YOLOv11**

![YOLOv11 Prediction 1](paper_results/predictions/yolov11_test_predictions/CDY_2051_png_jpg.rf.73671822e12124d814968c8d3644386c.jpg)
![YOLOv11 Prediction 2](paper_results/predictions/yolov11_test_predictions/CDY_2052_png_jpg.rf.50f54ec9aec74a6497050ef0714041a1.jpg)

**YOLOv12**

![YOLOv12 Prediction 1](paper_results/predictions/yolov12_test_predictions/CDY_2051_png_jpg.rf.73671822e12124d814968c8d3644386c.jpg)
![YOLOv12 Prediction 2](paper_results/predictions/yolov12_test_predictions/CDY_2052_png_jpg.rf.50f54ec9aec74a6497050ef0714041a1.jpg)

**YOLOv26**

![YOLOv26 Prediction 1](paper_results/predictions/yolov26_test_predictions/CDY_2051_png_jpg.rf.73671822e12124d814968c8d3644386c.jpg)
![YOLOv26 Prediction 2](paper_results/predictions/yolov26_test_predictions/CDY_2052_png_jpg.rf.50f54ec9aec74a6497050ef0714041a1.jpg)

---

## 🚀 Current Status

* Model training completed
* Evaluation completed
* Research paper finalized

---

## 🔮 Future Scope

* Use larger and more diverse datasets
* Optimize models for real-time deployment
* Integrate with smart farming systems (drones/robots)

---

## 📄 Paper & Presentation

* [Research Paper (PDF)](2211981212_LaxmiNarayan.pdf)
* [Presentation Slides (PPTX)](IOHE_External_PPT.pptx)

---

## 📚 References

* YOLO Research Papers
* Deep Learning in Agriculture
* Computer Vision for Fruit Detection

---

## 💡 Note

This project is developed for **academic and research purposes** in the domain of precision agriculture.
