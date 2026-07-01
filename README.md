# Image Classification with PyTorch & FastAPI

A production-style Image Classification project built with **PyTorch** and **FastAPI** using the **CIFAR-10** dataset.

The project demonstrates the complete deep learning workflow:

* Dataset preparation
* Data augmentation
* CNN model training
* Model evaluation
* Inference on new images
* REST API deployment with FastAPI

---

# Features

* Clean project architecture
* Modular PyTorch codebase
* CIFAR-10 dataset support
* Data augmentation
* Training & validation pipeline
* Accuracy, Precision, Recall, and F1-score
* Checkpoint saving/loading
* Loss & accuracy visualization
* Image inference
* FastAPI REST API
* Easy to extend with pretrained models

---

# Project Structure

```text
Image-Classification/
│
├── app/
│   ├── main.py
│   └── schemas.py
│
├── checkpoints/
├── data/
├── logs/
├── outputs/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── engine.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── metrics.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

---

# Dataset

This project uses the **CIFAR-10** dataset.

Classes:

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

Dataset Statistics

* Training Images: 50,000
* Test Images: 10,000
* Image Size: 32 × 32
* Number of Classes: 10

---

# Model Architecture

```
        Input Image
            ↓
        Conv Block
            ↓
        Conv Block
            ↓
         Max Pool
            ↓
        Conv Block
            ↓
        Conv Block
            ↓
         Max Pool
            ↓
        Conv Block
            ↓
        Conv Block
            ↓
         Max Pool
            ↓
         Flatten
            ↓
      Fully Connected
            ↓
         Dropout
            ↓
      Fully Connected
            ↓
    Output (10 Classes)
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/maroofiums/Images-Classification.git

cd Image-Classification
```

Create a virtual environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Train the Model

```bash
python -m src.train
```

During training the project will:

* Train the CNN
* Validate after every epoch
* Save the best checkpoint
* Generate training plots

Saved checkpoint:

```
checkpoints/
    best_model.pth
```

---

# Evaluate the Model

```bash
python -m src.evaluate
```

Evaluation includes:

* Test Loss
* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

---

# Run Inference

```bash
python -m src.inference
```

Example Output

```python
{
    "class": "dog",
    "confidence": 0.9873
}
```

---

# Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

Open your browser:

```
http://127.0.0.1:8000/docs
```

Swagger UI allows you to upload an image and receive predictions.

Example Response

```json
{
    "predicted_class": "cat",
    "confidence": 0.9968
}
```

---

# Project Workflow

```
         Dataset
            ↓
       Preprocessing
            ↓
         Training
            ↓
        Validation
            ↓
        Checkpoint
            ↓
        Evaluation
            ↓
        Inference
            ↓
     FastAPI Deployment
```

---

# Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Scikit-learn
* Matplotlib
* Pillow
* FastAPI
* Uvicorn
* Pydantic
* tqdm

---

# Future Improvements

* Transfer Learning (ResNet, EfficientNet, ConvNeXt)
* TensorBoard Integration
* Mixed Precision Training
* Early Stopping
* Learning Rate Warmup
* Docker Support
* GitHub Actions CI/CD
* MLflow Experiment Tracking
* Batch Inference API
* ONNX Export
* TorchScript Deployment

---

# License

This project is released under the MIT License.

---

# Acknowledgements

* PyTorch
* Torchvision
* FastAPI
* CIFAR-10 Dataset
