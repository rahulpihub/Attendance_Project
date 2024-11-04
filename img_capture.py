import cv2
import time
import os
import streamlit as st

# Create a directory to save images
save_dir = 'captured_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def capture_images(num_images):
    cap = cv2.VideoCapture(0)  # Use the default camera

    # Allow some time for the camera to warm up
    time.sleep(2)

    for i in range(num_images):
        ret, frame = cap.read()  # Capture a frame from the camera
        if ret:
            # Save the captured image
            img_name = os.path.join(save_dir, f'image_{i+1}.jpg')
            cv2.imwrite(img_name, frame)
            st.image(frame, caption=f'Captured {img_name}', channels="BGR")  # Display the captured image
            time.sleep(0.1)  # Wait for 0.1 seconds before capturing the next image
        else:
            st.error("Failed to capture image.")

    cap.release()
    st.success("Image capturing complete.")
    cv2.destroyAllWindows()

# Streamlit app
st.title("Image Capture App")
st.write("Click the button to start capturing images.")

if st.button("Start Capturing"):
    st.write("You have 10 seconds to get ready...")
    time.sleep(10)  # Wait for 10 seconds before starting to capture images
    capture_images(200)  # Capture 300 images
