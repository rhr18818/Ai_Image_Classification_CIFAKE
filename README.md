# AI Image Detection Using Multi-Stream MobileNetV3

A lightweight deep learning model for detecting AI-generated images using multi-stream feature fusion with MobileNetV3-Small architecture.

## 🎯 Overview

This project implements a robust AI image detection system that combines multiple feature extraction techniques to differentiate between real and AI-generated images. The model achieves high accuracy while maintaining computational efficiency through the use of MobileNetV3-Small architecture.

## 🔑 Key Features

### Multi-Stream Feature Extraction
The model processes images through **four distinct feature streams** (9 channels total):

1. **RGB Channels (3 channels)** - Standard color information
2. **Color Distribution Difference (3 channels)** - Detects "too perfect" color transitions in AI images through 4-bit quantization
3. **FFT Spectrum (1 channel)** - Captures frequency domain artifacts (checkerboard patterns) common in GAN/Diffusion models
4. **Luminance Gradients (2 channels)** - Sobel edge detection for horizontal (Gx) and vertical (Gy) texture analysis

### Lightweight Architecture
- **Model**: MobileNetV3-Small (~2.5M parameters)
- **Transfer Learning**: Pretrained on ImageNet with modified input/output layers
- **Input**: 9-channel tensor (224×224) instead of standard 3-channel RGB
- **Output**: Binary classification (Real vs Fake)

## 📊 Dataset

The model is trained on the **CIFAKE** dataset from Kaggle:
- Training and test splits with balanced REAL/FAKE classes
- Images resized to 224×224 for MobileNet compatibility
- Dataset structure:
  ```
  /train/REAL/
  /train/FAKE/
  /test/REAL/
  /test/FAKE/
  ```

## 🛠️ Technical Implementation

### Feature Extraction Logic

**Color Distribution Difference (Cyan Box)**
```python
# Simulates 4-bit quantization to detect unnatural color smoothness
quantized = (img_np // 16) * 16
diff = np.abs(img_np - quantized) / 255.0
```

**FFT Spectrum Analysis (Purple Box)**
```python
# Reveals frequency-domain artifacts in AI-generated images
magnitude_spectrum = 20 * log(|FFT(grayscale)|)
```

**Luminance Gradients (Green Box)**
```python
# Sobel operators for texture and edge analysis
gx = Sobel(gray, 1, 0)  # Horizontal gradients
gy = Sobel(gray, 0, 1)  # Vertical gradients
```

### Model Architecture Modifications

1. **Input Layer**: Modified from 3 to 9 channels
   - First 3 channels initialized with pretrained RGB weights
   - Remaining 6 channels initialized with Kaiming normal initialization

2. **Output Layer**: Changed from 1000 classes (ImageNet) to 2 classes (Real/Fake)

## 🚀 Usage

### Installation
```python
pip install torchinfo torch torchvision numpy opencv-python pillow scikit-learn matplotlib psutil
```

### Training Configuration
- **Batch Size**: 32
- **Learning Rate**: 0.0005
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 5 (configurable)

### Running the Model
The notebook includes:
- Automated data loading with progress tracking
- Training loop with batch-level progress indicators
- Comprehensive evaluation metrics
- Model checkpoint saving

## 📈 Evaluation Metrics

The model reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate for fake detection
- **Recall**: Sensitivity in detecting fake images
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **Latency**: Average inference time per image (ms)
- **Memory Usage**: RAM consumption during evaluation

## 📁 Project Structure

```
idea-1_model/
├── README.md
├── idea1_mobilenet_fusion.pth          # Trained model weights
└── ipynb files/
    └── mobilenet-fft-color-stats-robust-ai-detection.ipynb
```

## 🔬 Research Basis

This implementation is based on the concept of multi-modal feature fusion for AI image detection, combining:
- **Computer Vision**: Traditional image processing techniques
- **Deep Learning**: Transfer learning with efficient architectures
- **Signal Processing**: Frequency domain analysis via FFT
- **Physics-Based Features**: Gradient analysis for luminance patterns

## 💡 Why This Approach Works

1. **Frequency Artifacts**: GANs and diffusion models leave characteristic patterns in the frequency domain
2. **Color Quantization**: AI models often produce unnaturally smooth color transitions
3. **Gradient Anomalies**: Physics-based features reveal inconsistencies in lighting and texture
4. **Efficiency**: MobileNetV3 provides excellent accuracy-to-parameter ratio

## 🎓 Citation

If you use this work, please cite the CIFAKE dataset:
```
CIFAKE: Real and AI-Generated Synthetic Images
Available at: Kaggle Datasets
```

## 📝 License

This project is intended for research and educational purposes.
