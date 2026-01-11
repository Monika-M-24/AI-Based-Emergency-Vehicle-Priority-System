import cv2
import numpy as np
from flask import Flask, render_template, Response, request
import time  # Import time for managing the green light duration

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained Haar Cascade Classifier for vehicle detection
vehicle_cascade = cv2.CascadeClassifier('haarcascade_car.xml')  # Ensure the path is correct

# Check if the Haar Cascade file is loaded correctly
if vehicle_cascade.empty():
    raise IOError("Haar Cascade file not found or failed to load. Check the path.")

# Video Capture object
cap = None  # This will be set when we select the video source

# Variable to track the green light state
green_light_on = False
last_emergency_detected_time = None  # Track when the last emergency vehicle was detected
green_light_duration = 10  # Seconds the green light will stay on after detection

# Function to detect and classify emergency vehicles (ambulance/fire truck)
def detect_emergency_vehicle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect vehicles using the Haar Cascade Classifier
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    detected_emergency_vehicle = False
    detected_non_emergency_vehicle = False

    for (x, y, w, h) in vehicles:
        # Crop the detected vehicle area to analyze its potential type
        vehicle_roi = frame[y:y + h, x:x + w]

        # Simple example check: Detect red color (common for fire trucks/ambulances)
        lower_red = (0, 0, 100)
        upper_red = (100, 100, 255)
        mask = cv2.inRange(vehicle_roi, lower_red, upper_red)
        red_pixels = cv2.countNonZero(mask)

        # If a significant portion of the vehicle is red, we assume it's an emergency vehicle.
        if red_pixels > 200:  # Threshold for red pixels (adjust as needed)
            detected_emergency_vehicle = True
            label = "Emergency Vehicle (Ambulance)"
        else:
            detected_non_emergency_vehicle = True
            label = "Vehicle Detected"

        # Draw a rectangle around the detected vehicle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return detected_emergency_vehicle, detected_non_emergency_vehicle, frame


# Function to simulate controlling the traffic light
def control_traffic_light(detected_emergency_vehicle, detected_non_emergency_vehicle, frame):
    global green_light_on, last_emergency_detected_time

    current_time = time.time()

    if detected_emergency_vehicle:
        # Emergency vehicle detected, make the green light stay ON
        if not green_light_on:
            green_light_on = True
            last_emergency_detected_time = current_time  # Reset the green light duration timer
        cv2.putText(frame, "Green Light!", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (300, 400), (500, 600), (0, 255, 0), -1)  # Simulating green light box

    elif green_light_on:
        # If green light is already on and an emergency vehicle is not detected anymore, we check the duration
        if current_time - last_emergency_detected_time > green_light_duration:
            # If enough time has passed since the last emergency vehicle detection, turn off the green light
            green_light_on = False
            cv2.putText(frame, "Red Light", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (300, 400), (500, 600), (0, 0, 255), -1)  # Simulating red light box
        else:
            # Keep the green light on
            cv2.putText(frame, "Green Light ( Emergency)", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (300, 400), (500, 600), (0, 255, 0), -1)  # Simulating green light box

    elif detected_non_emergency_vehicle:
        # Non-emergency vehicle detected, red light is on
        green_light_on = False
        cv2.putText(frame, "Red Light", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.rectangle(frame, (300, 400), (500, 600), (0, 0, 255), -1)  # Simulating red light box

    else:
        # No vehicle detected, default to red light
        green_light_on = False
        cv2.putText(frame, "Red Light", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.rectangle(frame, (300, 400), (500, 600), (0, 0, 255), -1)  # Simulating red light box

    return frame


# Function to get frames from the webcam or video file and stream them to the browser
def generate_frames():
    global cap
    if cap is None:
        return  # Early exit if `cap` is not initialized yet.

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emergency vehicle and other vehicle
        detected_emergency_vehicle, detected_non_emergency_vehicle, frame = detect_emergency_vehicle(frame)

        # Simulate traffic light control based on vehicle detection
        frame = control_traffic_light(detected_emergency_vehicle, detected_non_emergency_vehicle, frame)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# Route to set up video source input
@app.route('/', methods=['GET', 'POST'])
def index():
    global cap
    if request.method == 'POST':
        video_source = request.form.get('video_source')

        if video_source == 'webcam':
            cap = cv2.VideoCapture(0)  # Use webcam
        elif video_source == 'video_file':
            video_file = request.files.get('video_file')  # Retrieve the uploaded video file
            if video_file:
                # Save the uploaded video file with a unique name
                video_path = 'uploaded_video.mp4'
                video_file.save(video_path)  # Save the uploaded file
                cap = cv2.VideoCapture(video_path)  # Open the uploaded video file
            else:
                return "No video file uploaded", 400  # Return error if no file is uploaded

    return render_template('index.html')  # Render the main page


# Route to return the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
