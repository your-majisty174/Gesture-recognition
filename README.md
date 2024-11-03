# Hand Gesture Recognition with MediaPipe Using Hidden Markov Models

This project estimates hand poses using MediaPipe in Python. It recognizes hand signs and finger gestures using a simple Machine Learning Model with detected key points, enhanced by Hidden Markov Models (HMM) to analyze gesture sequences.

![Hand Gesture Recognition Demo](https://user-images.githubusercontent.com/37477845/102222442-c452cd00-3f26-11eb-93ec-c387c98231be.gif)

## Project Contents
- Sample program for hand gesture recognition
- Hand sign recognition model (TFLite)
- Finger gesture recognition model (TFLite)
- Learning data and notebooks for hand sign recognition
- Learning data and notebooks for finger gesture recognition

## Requirements
To run this project, you need the following Python packages:
- **mediapipe**: Version 0.8.1
- **OpenCV**: Version 3.4.2 or later
- **TensorFlow**: Version 2.3.0 or later (or TensorFlow nightly 2.5.0.dev for LSTM model)
- **scikit-learn**: Version 0.23.2 or later (optional for displaying confusion matrix)
- **matplotlib**: Version 3.3.2 or later (optional for displaying confusion matrix)

## Demo
To run the demo using your webcam, use the following command:
```bash
python app.py
```

### Command-Line Options
You can specify options when running the demo:
- `--device`: Camera device number (default: 0)
- `--width`: Width of the camera capture (default: 960)
- `--height`: Height of the camera capture (default: 540)
- `--use_static_image_mode`: Use static image mode for MediaPipe (default: unspecified)
- `--min_detection_confidence`: Detection confidence threshold (default: 0.5)
- `--min_tracking_confidence`: Tracking confidence threshold (default: 0.5)

## Directory Structure
```
│  app.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │          
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│          
└─utils
    └─cvfpscalc.py
```

### File Descriptions
- **app.py**: Main program for inference and data collection.
- **keypoint_classification.ipynb**: Notebook for training the hand sign recognition model.
- **point_history_classification.ipynb**: Notebook for training the finger gesture recognition model.
- **model/**: Contains files related to model training and inference for both hand signs and finger gestures.
- **utils/cvfpscalc.py**: A utility for measuring frames per second (FPS).

## Training Process
You can collect and modify training data to retrain the models for hand sign and finger gesture recognition.

### Hand Sign Recognition Training
1. **Collect Learning Data**: Press "k" to start saving key points. When you press "0" to "9", the key points are saved to `model/keypoint_classifier/keypoint.csv`. 
   - Each entry includes a class ID (the number pressed) and key point coordinates.
  
2. **Model Training**: Open `keypoint_classification.ipynb` in Jupyter Notebook and run it from top to bottom. Change `NUM_CLASSES` if you add more classes and update `model/keypoint_classifier/keypoint_classifier_label.csv` accordingly.

### Finger Gesture Recognition Training
1. **Collect Learning Data**: Press "h" to save fingertip coordinate history. Press "0" to "9" to log the coordinates in `model/point_history_classifier/point_history.csv`. 
   - Each entry includes a class ID and coordinate history.

2. **Model Training**: Open `point_history_classification.ipynb` in Jupyter Notebook and run it from top to bottom. Change `NUM_CLASSES` if you add more classes and update `model/point_history_classifier/point_history_classifier_label.csv` as needed.

## Hidden Markov Models (HMM) for Gesture Recognition
In this project, we use Hidden Markov Models (HMM) to enhance the recognition of gestures over time. HMMs are particularly effective for modeling sequences of data where the states (in this case, hand gestures) are not directly observable. 

### How HMM Works in This Context:
1. **State Representation**: Each gesture is represented as a sequence of states corresponding to different key points detected by MediaPipe.
2. **Observations**: The HMM receives observations from the captured key points in real-time, creating a dynamic model of the user's hand movements.
3. **Training**: The model is trained using sequences of key point data collected during the gesture execution, allowing it to learn the typical transition probabilities between gestures.
4. **Decoding**: During recognition, the model uses the Viterbi algorithm to find the most likely sequence of gestures based on the observed key points, making it robust against variations in speed and style of gesture execution.

### Advantages of Using HMM
- **Temporal Dynamics**: HMM captures the temporal dynamics of gestures, allowing it to distinguish between gestures that might have similar static positions.
- **Robustness**: The model is less sensitive to noise in the data, which is common in real-time applications.
- **Flexibility**: HMM can adapt to different gesture patterns over time, improving recognition accuracy as more data is collected.

## Reference
- [MediaPipe Documentation](https://mediapipe.dev/) 
 
# License 
hand-gesture-recognition-using-mediapipe is under [Apache v2 license](LICENSE).
