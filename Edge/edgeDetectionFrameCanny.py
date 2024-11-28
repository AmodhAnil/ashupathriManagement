import cv2
import numpy as np

def capture_on_key_press_canny(source=0, low_threshold=50, high_threshold=150):
    """
    Open a video source and allow capturing frames on key press for edge detection using Canny.
    
    Parameters:
    source (int or str): 0 for webcam, or path to video file
    low_threshold (int): Lower threshold for Canny edge detection
    high_threshold (int): Upper threshold for Canny edge detection
    """
    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError("Could not open video source")
    
    try:
        print("Press 'C' to capture a frame for edge detection using Canny. Press 'Q' to quit.")
        
        while True:
            # Read frame from the video source
            ret, frame = cap.read()
            if not ret:
                print("No frame captured. Exiting...")
                break
            
            # Display the live feed
            cv2.imshow('Live Feed (Press C to capture, Q to quit)', frame)
            cv2.setWindowTitle('Live Feed (Press C to capture, Q to quit)', 
                               'Live Feed')

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):  # Capture frame on 'C'
                # Convert the captured frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply Canny edge detection
                edges = cv2.Canny(gray, low_threshold, high_threshold)
                
                # Create overlay by combining original and edges
                overlay = frame.copy()
                overlay[edges > 0] = [0, 255, 0]  # Green edges
                
                # Stack frames for display
                edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                combined_frame = np.hstack((frame, edges_3channel, overlay))
                
                # Display the captured and processed frame
                cv2.imshow('Captured Frame and Edge Detection', combined_frame)
                cv2.setWindowTitle('Captured Frame and Edge Detection', 
                                   'Original | Edges | Overlay')
                
                # Save the processed images
                cv2.imwrite('canny_frame_original.jpg', frame)
                cv2.imwrite('canny_frame_edges.jpg', edges)
                cv2.imwrite('canny_frame_overlay.jpg', overlay)
                print("Frame captured and saved! Press any key to return to live feed.")
                
                # Wait for a key press to return to live feed
                cv2.waitKey(0)
                cv2.destroyWindow('Captured Frame and Edge Detection')
                
            elif key == ord('q'):  # Quit on 'Q'
                print("Exiting...")
                break

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

def main():
    """
    Main function to run the capture on key press application with Canny edge detection.
    """
    try:
        # Parameters
        VIDEO_SOURCE = 0  # Use 0 for webcam or provide path to video file
        LOW_THRESHOLD = 50   # Lower threshold for Canny edge detection
        HIGH_THRESHOLD = 150  # Upper threshold for Canny edge detection

        print("Starting live feed. Press 'C' to capture or 'Q' to quit.")
        
        capture_on_key_press_canny(
            source=VIDEO_SOURCE,
            low_threshold=LOW_THRESHOLD,
            high_threshold=HIGH_THRESHOLD
        )
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
