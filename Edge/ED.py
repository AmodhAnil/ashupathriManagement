import cv2
import numpy as np

def process_video(source=0, kernel_size=3, scale=1, delta=0):
    """
    Perform real-time Laplacian edge detection on video input with overlay.
    
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

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer for three outputs
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('edge_detection_combined.avi', fourcc, fps, (frame_width*3, frame_height))

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply Laplacian edge detection
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size, scale=scale, delta=delta)
            
            # Convert back to uint8 and get absolute values
            laplacian = np.uint8(np.absolute(laplacian))
            
            # Apply threshold to make edges more visible
            _, laplacian_binary = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
            
            # Create overlay by combining grayscale and edges
            gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            edges_3channel = cv2.cvtColor(laplacian_binary, cv2.COLOR_GRAY2BGR)
            
            # Create overlay effect
            overlay = gray_3channel.copy()
            mask = laplacian_binary > 0
            overlay[mask] = [0, 255, 0]  # Green edges
            
            # Stack all three versions side by side
            combined_frame = np.hstack((frame, edges_3channel, overlay))
            
            # Write the frame to output video
            out.write(combined_frame)
            
            # Display the result
            cv2.imshow('Edge Detection (Press Q to quit)', combined_frame)
            cv2.setWindowTitle('Edge Detection (Press Q to quit)', 
                             'Original | Edges | Overlay')
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    finally:
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    """
    Main function with error handling and parameter settings.
    """
    try:
        # You can modify these parameters
        VIDEO_SOURCE = 0  # Use 0 for webcam or provide path to video file
        KERNEL_SIZE = 3   # Must be odd number (1, 3, 5, etc.)
        SCALE = 1        # Scaling factor for edge detection
        DELTA = 0        # Delta value added to results
        
        print("Starting edge detection...")
        print("Press 'Q' to quit the application")
        
        process_video(
            source=VIDEO_SOURCE,
            kernel_size=KERNEL_SIZE,
            scale=SCALE,
            delta=DELTA
        )
        
        print("Processing complete! Output saved as 'edge_detection_combined.avi'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()