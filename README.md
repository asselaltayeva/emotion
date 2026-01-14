# Emotion Detection

A facial emotion recognition system that uses machine learning to classify emotions from facial images. The system extracts facial landmarks using MediaPipe and trains a Random Forest Classifier to predict emotions.

## Features

- **Face Landmark Extraction**: Utilizes MediaPipe to extract 468 facial landmarks (1404 features with x, y, z coordinates)
- **Machine Learning Classification**: Trains a Random Forest Classifier on the extracted features
- **Automated Data Processing**: Processes images organized by emotion categories
- **Model Persistence**: Saves trained models for reuse
- **Performance Metrics**: Provides accuracy scores and confusion matrix for model evaluation

## Project Structure

```
emotion/
├── prepare_data.py      # Extract face landmarks from images and prepare training data
├── train_model.py       # Train Random Forest Classifier on prepared data
├── utils.py            # Utility functions for face landmark detection
├── requirements.txt    # Project dependencies
├── .gitignore         # Git ignore file
└── README.md          # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/asselaltayeva/emotion.git
cd emotion
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- `opencv-python` (4.12.0.88) - Image processing
- `mediapipe` (0.10.31) - Face landmark detection
- `numpy` (2.2.6) - Numerical computations
- `scikit-learn` (1.8.0) - Machine learning algorithms
- `scipy` (1.17.0) - Scientific computing
- `sounddevice` (0.5.3) - Audio processing
- Other supporting libraries (see `requirements.txt`)

## Usage

### 1. Prepare Your Data

Organize your training images in a directory structure where each subdirectory represents an emotion category:

```
data/
├── happy/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── sad/
│   ├── image1.jpg
│   └── ...
├── angry/
└── ...
```

### 2. Extract Face Landmarks

Run the data preparation script to extract facial landmarks from your images:

```bash
python prepare_data.py
```

This will:
- Process all images in the `./data` directory
- Extract 468 facial landmarks (x, y, z coordinates) per face
- Save the features and labels to `data.txt`

### 3. Train the Model

Train the Random Forest Classifier on the prepared data:

```bash
python train_model.py
```

This will:
- Load data from `data.txt`
- Split data into training and testing sets (80/20 split)
- Train a Random Forest Classifier
- Evaluate the model and print accuracy metrics
- Save the trained model to `./model`

### 4. Using the Trained Model

You can load and use the trained model in your own scripts:

```python
import pickle
import numpy as np
from utils import get_face_landmarks

# Load the trained model
with open('./model', 'rb') as f:
    model = pickle.load(f)

# Extract landmarks from a new image
image_path = 'path/to/your/image.jpg'
landmarks = get_face_landmarks(image_path)

if len(landmarks) == 1404:
    # Make prediction
    prediction = model.predict([landmarks])
    print(f"Predicted emotion: {prediction[0]}")
```

## How It Works

1. **Face Detection**: MediaPipe's FaceMesh detects faces in images and identifies 468 landmarks
2. **Feature Extraction**: Each landmark's x, y, z coordinates are normalized and flattened into a 1404-dimensional feature vector
3. **Model Training**: Random Forest Classifier learns patterns in facial landmarks associated with different emotions
4. **Prediction**: New images are processed through the same pipeline to predict emotions

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 1404 (468 landmarks × 3 coordinates)
- **Train/Test Split**: 80/20
- **Random State**: 42 (for reproducibility)

## Technical Notes

- The system requires faces to be visible and properly lit for accurate landmark detection
- Only processes images with exactly 1404 detected landmarks (468 × 3)
- MediaPipe face mesh uses a confidence threshold of 0.5 for face detection
- Landmarks are normalized by subtracting the minimum values for each coordinate

## Future Improvements

- [ ] Add real-time emotion detection from webcam
- [ ] Support for video file processing
- [ ] Implement deep learning models (CNN, ResNet)
- [ ] Add data augmentation techniques
- [ ] Create a web interface for easy interaction
- [ ] Expand emotion categories
- [ ] Add confidence scores for predictions

## License

This project is open source and available for educational and research purposes.

## Author

Assel Altayeva - [@asselaltayeva](https://github.com/asselaltayeva)
