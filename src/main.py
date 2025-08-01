import mss
import numpy as np
import cv2
import time
import threading
import queue

from .utils import get_primary_monitor_info

# Global flag and thread-safe queue for inter-thread communication
capture_queue = queue.Queue(maxsize=1)
quit_event = threading.Event()

def capture_thread_worker(screen_region):
    """
    Worker function for the capture thread. It grabs frames and puts them in a queue.
    """
    print("Capture thread started.")
    with mss.mss() as sct:
        while not quit_event.is_set():
            sct_img = sct.grab(screen_region)
            frame_np = np.array(sct_img)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_BGRA2BGR)
            
            try:
                capture_queue.put_nowait(frame_bgr)
            except queue.Full:
                try:
                    capture_queue.get_nowait()
                    capture_queue.put_nowait(frame_bgr)
                except queue.Empty:
                    capture_queue.put_nowait(frame_bgr)
            
            time.sleep(0.005)

    print("Capture thread shutting down.")


def main():
    """
    Main thread function. It handles the GUI and frame display.
    """
    global quit_event
    
    print("Application started. Click the native window's close button or press 'q' to exit.")
    
    screen_region = get_primary_monitor_info()
    print(f"Capturing primary monitor with dimensions: {screen_region}")
    
    last_frame_time = time.time()
    
    window_name = "Live Screen Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    capture_thread = threading.Thread(target=capture_thread_worker, args=(screen_region,), daemon=True)
    capture_thread.start()
    
    try:
        while not quit_event.is_set():
            try:
                frame = capture_queue.get_nowait()
            except queue.Empty:
                frame = None

            if frame is not None:
                current_time = time.time()
                fps = 1 / (current_time - last_frame_time)
                last_frame_time = current_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow(window_name, frame)

            # Process GUI events with a minimal delay
            key = cv2.waitKey(1)

            # Check for 'q' keypress
            if key == ord('q'):
                print("Quit signal received from keyboard.")
                quit_event.set()
                
            # Check for the native window close event
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Native window was closed by user.")
                quit_event.set()
                
    except Exception as e:
        print(f"An unexpected error occurred in the main thread: {e}")
        
    finally:
        if not quit_event.is_set():
            quit_event.set()
            
        cv2.destroyAllWindows()
        print("Main thread shutting down.")

if __name__ == "__main__":
    main()
