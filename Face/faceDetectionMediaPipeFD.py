import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection and Drawing utilities
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for webcam, or provide a video file path

# Configure Face Detection
with mp_face_detection.FaceDetection(
    model_selection=0,  # 0 for close-range (webcam), 1 for long-range detection
    min_detection_confidence=0.5  # Minimum confidence for detection
) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            break

        # Convert frame to RGB (MediaPipe uses RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for face detection
        results = face_detection.process(rgb_frame)

        # Draw the face detection results
        if results.detections:
            for detection in results.detections:
                # Draw bounding box and landmarks
                mp_drawing.draw_detection(frame, detection)

        # Display the output
        cv2.imshow('Face Detection', frame)

        # Exit with 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
