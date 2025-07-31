import mss
import numpy as np
import cv2
import time
from .utils import get_primary_monitor_info

def main():
    """
    Main function to orchestrate the real-time screen capture and analysis.
    This runs in a single, tight loop.
    """
    print("Application started. Click on the 'Live Screen Feed' window and press 'q' to exit.")
    
    # Use our utility function to get the monitor region
    screen_region = get_primary_monitor_info()
    print(f"Capturing primary monitor with dimensions: {screen_region}")
    
    last_frame_time = time.time()
    
    # Create the named window outside the loop to make sure it exists
    window_name = "Live Screen Feed - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Keep track of the quit signal
    quit_requested = False

    try:
        with mss.mss() as sct:
            while not quit_requested:
                # 1. Capture the screen (this is the most blocking part)
                sct_img = sct.grab(screen_region)

                # 2. Convert to a NumPy array for OpenCV
                frame_np = np.array(sct_img)
                frame = cv2.cvtColor(frame_np, cv2.COLOR_BGRA2BGR)
                
                # 3. --- Your Custom Analysis Code Goes Here ---
                # This is where you would place your computer vision algorithms.
                # For now, let's just draw the FPS on the frame.
                current_time = time.time()
                fps = 1 / (current_time - last_frame_time)
                last_frame_time = current_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # 4. Display the frame
                cv2.imshow(window_name, frame)
                
                # 5. Handle GUI events and check for quit signal
                # Wait for 1ms and check for a keypress
                key = cv2.waitKey(1)
                
                # Check for 'q' keypress (ASCII 113)
                if key == ord('q'):
                    print("Quit signal received from keyboard.")
                    quit_requested = True
                
                # Also check for window close event (macOS specific)
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window was closed by user.")
                    quit_requested = True
                    
    except mss.exception.ScreenShotError as e:
        print(f"Screen capture failed: {e}")
        print("Please ensure your application has 'Screen Recording' permissions in macOS Settings.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
    finally:
        # Clean up OpenCV windows
        cv2.destroyAllWindows()
        print("Shutting down.")


if __name__ == "__main__":
    main()
