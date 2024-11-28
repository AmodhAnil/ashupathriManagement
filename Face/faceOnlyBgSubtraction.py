import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for webcam, or use a file path

# Face Mesh parameters
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # Includes detailed facial landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            break

        # Convert frame to RGB (MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect face mesh
        results = face_mesh.process(rgb_frame)

        # Create a black mask of the same size as the frame
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Convert landmarks to a NumPy array of (x, y) coordinates
                h, w, _ = frame.shape
                points = np.array(
                    [(int(landmark.x * w), int(landmark.y * h)) 
                     for landmark in face_landmarks.landmark], 
                    dtype=np.int32
                )

                # Create a convex hull around the points
                hull = cv2.convexHull(points)

                # Fill the convex hull on the mask
                cv2.fillPoly(mask, [hull], 255)

        # Smooth the mask using Gaussian blur
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        # Create a result image with the face isolated
        face_only = cv2.bitwise_and(frame, frame, mask=mask)

        # Display the resulting frame
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Face Only', face_only)

        # Exit with 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
