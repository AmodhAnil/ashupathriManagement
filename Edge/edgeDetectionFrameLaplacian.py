import cv2
import numpy as np

def capture_on_key_press(source=0, kernel_size=3, scale=1, delta=0):
    """
    Open a video source and allow capturing frames on key press for edge detection.
    
    Parameters:
    source (int or str): 0 for webcam, or path to video file
    kernel_size (int): Size of the Laplacian kernel (must be odd)
    scale (int): Optional scaling factor for edge detection
    delta (int): Optional delta value added to results
    """
    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError("Could not open video source")
    
    try:
        print("Press 'C' to capture a frame for edge detection. Press 'Q' to quit.")
        
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
                # Process the captured frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size, scale=scale, delta=delta)
                laplacian = np.uint8(np.absolute(laplacian))
                _, laplacian_binary = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
                
                # Create the overlay
                gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                overlay = gray_3channel.copy()
                mask = laplacian_binary > 0
                overlay[mask] = [0, 255, 0]  # Green edges
                
                # Stack frames for display
                combined_frame = np.hstack((frame, cv2.cvtColor(laplacian_binary, cv2.COLOR_GRAY2BGR), overlay))
                
                # Display the captured and processed frame
                cv2.imshow('Captured Frame and Edge Detection', combined_frame)
                cv2.setWindowTitle('Captured Frame and Edge Detection', 
                                   'Original | Edges | Overlay')
                
                # Save the processed images
                cv2.imwrite('captured_frame_original.jpg', frame)
                cv2.imwrite('captured_frame_edges.jpg', laplacian_binary)
                cv2.imwrite('captured_frame_overlay.jpg', overlay)
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
    Main function to run the capture on key press application.
    """
    try:
        # Parameters
        VIDEO_SOURCE = 0  # Use 0 for webcam or provide path to video file
        KERNEL_SIZE = 3   # Must be odd number (1, 3, 5, etc.)
        SCALE = 1        # Scaling factor for edge detection
        DELTA = 0        # Delta value added to results

        print("Starting live feed. Press 'C' to capture or 'Q' to quit.")
        
        capture_on_key_press(
            source=VIDEO_SOURCE,
            kernel_size=KERNEL_SIZE,
            scale=SCALE,
            delta=DELTA
        )
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
