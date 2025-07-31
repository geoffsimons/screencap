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
    print("Application started. Press 'q' in the OpenCV window to exit.")
    
    # Use our utility function to get the monitor region
    screen_region = get_primary_monitor_info()
    print(f"Capturing primary monitor with dimensions: {screen_region}")
    
    last_frame_time = time.time()

    try:
        with mss.mss() as sct:
            while True:
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
                cv2.imshow("Live Screen Feed - Press 'q' to quit", frame)
                
                # 5. Handle GUI events and check for quit signal
                # The waitKey(1) call is crucial. It processes GUI events and
                # acts as a very short pause, allowing the OS to draw the window.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit signal received.")
                    break
    
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