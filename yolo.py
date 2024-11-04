import streamlit as st
import os
import cv2
import pandas as pd
from ultralytics import YOLO
import time

# Load YOLO model
def load_model(model_path='R:\\ihub\\Work\\Oct\\29-10-2024\\best.pt'):
    model = YOLO(model_path)  # Load with ultralytics
    return model

# Append detected names to CSV
def append_to_csv(name, status, csv_file='R:\\ihub\\Work\\Oct\\29-10-2024\\detections.csv'):
    df = pd.DataFrame({'Employee Name': [name], 
                       'Timestamp': [time.strftime("%Y-%m-%d %H:%M:%S")], 
                       'Status': [status]})
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

# Run inference on video with real-time display
def detect_video(model, stframe, output_dir='R:\\ihub\\Work\\Oct\\29-10-2024\\output'):
    # Initialize video capture from the webcam
    cap = cv2.VideoCapture(0)
    os.makedirs(output_dir, exist_ok=True)

    # Create a VideoWriter to save the annotated video
    output_video_path = os.path.join(output_dir, 'annotated_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Default to 30 FPS if unavailable
    out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    detected_names = set()  # Use a set to track detected names
    detection_occurred = False
    logged_in_names = set()  # Track logged-in employees

    while cap.isOpened() and not st.session_state.stop_detection:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame_class_names = []

        if hasattr(results[0], 'boxes'):
            for box in results[0].boxes:
                class_name = model.names[int(box.cls[0].item())]
                frame_class_names.append(class_name)

                # If the detected name is new and not logged in, log them in
                if class_name not in logged_in_names:
                    logged_in_names.add(class_name)  # Mark as logged in
                    append_to_csv(class_name, 'Logged-In')  # Append the detected name to CSV with Logged-In status
                    detection_occurred = True  # Indicate that a detection has occurred
                    st.write(f"{class_name} has logged in.")
                    time.sleep(5)  # Wait for 5 seconds after logging in

                else:
                    # If the detected name was previously detected, consider it as logged out
                    append_to_csv(class_name, 'Logged-Out')  # Append the detected name to CSV with Logged-Out status
                    st.write(f"{class_name} has logged out.")
                    logged_in_names.remove(class_name)  # Mark as logged out
                    time.sleep(5)  # Wait for 5 seconds after logging out

        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # Send annotated frame to Streamlit for real-time display
        stframe.image(annotated_frame, channels="BGR")

    cap.release()
    out.release()

    return output_video_path, detection_occurred

# Initialize the YOLO model
model = load_model()

# Initialize Streamlit session state for managing detection state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "stop_detection" not in st.session_state:
    st.session_state.stop_detection = False
if "detection_in_progress" not in st.session_state:
    st.session_state.detection_in_progress = False  # Flag to track detection progress

# Streamlit interface
st.title("Ihub - Attendance")
st.write("Click the button to start")

# Create a placeholder for video display
stframe = st.empty()

# Start detection if not logged in
if not st.session_state.logged_in:
    if st.button("Start", key="start_detection"):
        # Reset stop detection flag
        st.session_state.stop_detection = False
        st.session_state.detection_in_progress = True  # Set detection in progress flag
        # Run detection on the webcam feed
        output_video_path, detection_occurred = detect_video(model, stframe)
        
        # Display final results and login message if a known face is detected
        if detection_occurred:
            st.write("Detection occurred and data recorded.")

# Show the stop button only if detection is in progress
if st.session_state.detection_in_progress and st.button("Stop", key="stop_button"):
    st.session_state.stop_detection = True  # Set the flag to stop detection
    st.session_state.detection_in_progress = False  # Reset detection in progress flag
    st.write("Detection has been stopped.")

# Automatic new employee detection if already logged in
if st.session_state.logged_in:
    # Run detection again immediately after logging in
    output_video_path, detection_occurred = detect_video(model, stframe)
    
    # Check if a new name was found
    if detection_occurred:
        st.write("Detection occurred and data recorded.")
    else:
        st.write("No new detections. Ready for further detection.")
        
    # Reset the login state for a new detection if desired
    if st.button("Detect New Employee", key="new_employee_detection"):
        st.session_state.logged_in = False  # Reset login state for new detection
        st.write("Ready for new employee detection.")
