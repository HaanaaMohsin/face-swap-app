
# Face Swapping App by Adil Khan

Welcome to the **Face Swapping App** repository! This application leverages **Streamlit** and **OpenCV**, powered by **InsightFace**, to perform seamless face swapping on images and videos. Whether you want to swap faces in pictures or videos, this app provides a simple and efficient solution.

---

## Features

- **Image Face Swap**: Effortlessly swap faces between a source and target image.
- **Video Face Swap**: Apply face swapping across all frames of a target video using a source face image.
- **High Precision**: Built on **InsightFace** deep learning models for accurate face detection and swapping.
- **User-Friendly Interface**: Interact with the app through a sleek and intuitive **Streamlit** interface.
 - **Eye Disease Detection (Swin Transformer)**: Classify retinal fundus images using a Swin Transformer classifier.

---

## How It Works

1. **Image Face Swapping**:
   - Detects faces in both the source and target images.
   - Replaces the target face with the source face using the `inswapper_128.onnx` model.
   
2. **Video Face Swapping**:
   - Processes each frame of the video to detect and swap faces to match the source face.

---

## Installation

Follow these steps to get started:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AdiKhanOfficial/face-swap-app.git
   cd face-swap-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   - For the Streamlit interface:
     ```bash
     streamlit run app.py
     ```
   - Or, simply run:
     ```bash
     python run.py
     ```

---

## Requirements

- **Python 3.9**  
- **Streamlit**  
- **OpenCV**  
- **InsightFace**
- **PyTorch**, **torchvision**, **timm**, **Pillow** (for Eye Disease Detection)

**Download Links**:
- [Python 3.9](https://www.python.org/downloads/release/python-390/)
- [InSwapper Model (`inswapper_128.onnx`)](https://cdn.adikhanofficial.com/python/insightface/models/inswapper_128.onnx)

---

## Usage

1. Open the application using Streamlit or `run.py`.
2. Upload your **source face image** and the **target image/video**.
3. Click the process button and let the magic happen!
4. Once processing is complete, download the swapped images or videos directly from the app.

---

## Eye Disease Detection (Swin Transformer)

This repository also includes a Swin Transformer-based classifier for retinal fundus images.

### Dataset Format

Use an ImageFolder-style dataset:

```text
data/
  train/
    DiseaseA/
      img1.jpg
      ...
    DiseaseB/
      ...
  val/  # optional (if missing, the trainer will split automatically)
    DiseaseA/
    DiseaseB/
```

If `val/` is not present, a validation split (default 20%) is created from `data/`.

### Training

Run the trainer as a module (recommended to ensure imports work):

```bash
python -m eye_disease.train /path/to/data \
  --output-dir ./checkpoints \
  --model-name swin_tiny_patch4_window7_224 \
  --image-size 224 \
  --epochs 20 \
  --batch-size 32 \
  --lr 3e-4
```

Useful flags:

- `--balance`: class-balanced sampling for imbalanced datasets
- `--freeze-backbone`: fine-tune only the classifier head
- `--no-amp`: disable mixed precision

Checkpoints are saved to `--output-dir` as `model_best.pth` and `model_last.pth` with metadata
(classes, image size, model name) embedded.

### Inference (CLI)

```bash
# Single image
python -m eye_disease.infer ./checkpoints/model_best.pth --image /path/to/fundus.jpg

# Folder of images
python -m eye_disease.infer ./checkpoints/model_best.pth --folder /path/to/folder
```

### Inference (Streamlit)

Launch the app and choose "Eye Disease Detection" in the sidebar:

```bash
streamlit run app.py
# or
python run.py
```

Upload your trained checkpoint (`.pth`) and fundus images to see predictions and class probabilities.

> Tip: GPU acceleration is recommended. The Streamlit page lets you choose device; "auto" will prefer CUDA if available.

---

## Results
<img src='https://raw.githubusercontent.com/AdiKhanOfficial/face_swapping/refs/heads/main/Results/Result.jpg' style='width:100%'/>

---

## Contributing

Contributions are welcome! If you’d like to improve the app or add new features, feel free to fork the repository, make your changes, and submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## About

Developed by **Adil Khan**.  
For more projects and updates, visit my [GitHub Profile](https://github.com/AdiKhanOfficial) or follow me on social media.

---

Feel free to reach out with any questions or feedback! 😊

