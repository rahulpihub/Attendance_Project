# 📸 Face Attendance Detection

This project leverages **YOLO** for face detection and **Streamlit** for the user interface, allowing real-time detection and logging of employee attendance. It uses a webcam to capture faces and logs attendance to a CSV file.

## 🛠️ Requirements

Before running the project, ensure you have the following installed:

- Python 3.x
- `ultralytics` (YOLO model)
- `roboflow` (for dataset management)
- `opencv-python` (for real-time video capture)
- `streamlit` (for the user interface)

Install required dependencies with:

```bash
pip install ultralytics supervision roboflow opencv-python streamlit
```

## 🚀 Project Structure

1. **Yolo_Code Training.ipynb**:
   - Loads a pre-trained YOLO model (`yolo11n.pt`).
   - Downloads the dataset from Roboflow and trains the model for face detection.

2. **img_capture.py**:
   - Captures images using the webcam for face data collection.
   - Saves the captured images in the `captured_images` directory.

3. **yolo.py**:
   - Loads the trained YOLO model (`best.pt`).
   - Runs real-time face detection via webcam.
   - Logs attendance (logged in/out) to a CSV file.
   - Displays live video feed with annotations on detected faces.

## 📸 How to Capture Images

Use the `img_capture.py` script to capture 200 images for training the model. Run the script in a **Streamlit** environment:

```bash
streamlit run img_capture.py
```

## 🎯 How to Train the Model

Use the Jupyter notebook `Yolo_Code Training.ipynb` to train the YOLO model:

```bash
# Run the notebook or this command in terminal
!yolo task=detect mode=train model=yolo11n.pt data={dataset.location}/data.yaml epochs=50 imgsz=640 plots=True
```

## 📹 Real-Time Detection

Run `yolo.py` in Streamlit to start the real-time face detection:

```bash
streamlit run yolo.py
```

- **Start Button**: Begins face detection.
- **Stop Button**: Stops the detection process.
- **New Employee Detection**: Resets the system for new employee detection.

### Features:
- 🚶‍♂️ **Real-Time Detection**: Tracks employee attendance in real-time.
- 🗂️ **Attendance Logging**: Logs detected employees' names with timestamps to a CSV file.
- 🎥 **Live Video Feed**: Displays webcam feed with annotations for each detected face.

## ⚙️ How to Run

1. **Capture Images**: Run the `img_capture.py` script to capture images for training.
2. **Train the Model**: Use the notebook `Yolo_Code Training.ipynb` to train the YOLO model.
3. **Run Detection**: Start the detection by running the `yolo.py` Streamlit app.

## 📝 Data Storage

- **Captured Images**: Stored in the `captured_images/` directory.
- **Detection Logs**: Stored in `detections.csv`, with columns: 
  - Employee Name
  - Timestamp
  - Status (Logged-In/Logged-Out)

## 🔧 Troubleshooting

- **No Face Detected**: Ensure good lighting and a clear view of faces.
- **Webcam Not Working**: Check your camera permissions and drivers.
