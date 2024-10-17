import cv2
import numpy as np
import tensorflow as tf
import time

# Load your pre-trained model
model = tf.keras.models.load_model('/Users/dechathon_niamsa-ard/Documents/Dechathon_N/MitrCharcheep/Dataset_Creator/cnn_model_xtra.h5')

# Define class names based on your model's output
class_names = ['Class A', 'Class B', 'Class C']  # Adjust these to your actual class names

# Set the video capture device (0 is typically the default camera)
cap = cv2.VideoCapture('0')

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Function to preprocess the frame for the model
def preprocess_frame(frame, target_size=(224, 224)):  # Adjust size according to your model
    frame = cv2.resize(frame, target_size)
    frame = frame.astype('float32') / 255.0  # Normalize if your model expects normalized input
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Set the time interval for prediction (10 seconds)
interval = 10  # seconds
last_prediction_time = time.time()

# Variables to store the last predicted class and probability
last_class_name = ''
last_probability = 0.0

# Start the video stream and process frames
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Every 10 seconds, make a prediction
        current_time = time.time()
        if current_time - last_prediction_time >= interval:
            processed_frame = preprocess_frame(frame)
            predictions = model.predict(processed_frame)
            predicted_class = np.argmax(predictions)
            last_probability = predictions[0][predicted_class]
            last_class_name = class_names[predicted_class]
            last_prediction_time = current_time

        # Get the frame dimensions (height and width)
        height, width, _ = frame.shape

        # Display the last predicted class and probability at the bottom of the frame
        text = f"Class: {last_class_name}, Probability: {last_probability:.2f}"
        
        # Set the position for the text (10 pixels from the bottom and centered horizontally)
        text_position = (10, height - 10)  # 10 pixels from the bottom
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Show the frame with prediction
        cv2.imshow('Live Stream', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()