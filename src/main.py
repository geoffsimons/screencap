import mss
import numpy as np
import cv2
import time
import threading
import queue
import argparse
import os
import traceback

from .utils import get_primary_monitor_info, find_all_window_coordinates
from .analysis import analyze_frame_for_components, calculate_edge_change, frame_to_edges

capture_queue = queue.Queue(maxsize=1)
quit_event = threading.Event()
frame_buffer_size = 10

def capture_thread_worker(screen_region):
    print("Capture thread started.")
    with mss.mss() as sct:
        while not quit_event.is_set():
            sct_img = sct.grab(screen_region)
            frame_np = np.array(sct_img)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_BGRA2BGR)

            # Resize the frame by 50% to account for Retina scaling
            height, width, _ = frame_bgr.shape
            resized_frame = cv2.resize(frame_bgr, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

            try:
                capture_queue.put_nowait(resized_frame)
            except queue.Full:
                try:
                    capture_queue.get_nowait()
                    capture_queue.put_nowait(resized_frame)
                except queue.Empty:
                    capture_queue.put_nowait(resized_frame)

            time.sleep(0.005)

    print("Capture thread shutting down.")

def main():
    global quit_event

    parser = argparse.ArgumentParser(description="Real-time macOS screencapture and OCR application.")
    parser.add_argument('--window', type=str, help='The title of the window to capture.')
    parser.add_argument('--x', type=int, help='The x-coordinate of the capture box top-left corner.')
    parser.add_argument('--y', type=int, help='The y-coordinate of the capture box top-left corner.')
    parser.add_argument('--width', type=int, help='The width of the capture box.')
    parser.add_argument('--height', type=int, help='The height of the capture box.')
    parser.add_argument('--save', action='store_true', help='Capture a single frame and save it to a file, then exit.')
    parser.add_argument('--load_file', type=str, help='Load an image file from disk, analyze it, and display the results.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (e.g., show intermediate image processing steps)')

    args = parser.parse_args()

    # If --load_file is provided, we skip all capture logic and go straight to analysis.
    if args.load_file:
        if not os.path.exists(args.load_file):
            print(f"Error: The file '{args.load_file}' was not found.")
            return

        frame = cv2.imread(args.load_file)
        if frame is None:
            print(f"Error: Could not load image from '{args.load_file}'.")
            return

        print(f"Loading and analyzing image from '{args.load_file}'...")
        boxes, annotated_frame = analyze_frame_for_components(frame, args.debug)

        window_name = "Analysis Result"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, annotated_frame)

        print("Analysis complete. Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Determine the capture region for live mode or saving.
    if args.window:
        found_windows = find_all_window_coordinates(args.window)
        if found_windows:
            screen_region = found_windows[0]
            print(f"Capturing the first matching window '{screen_region['window_name']}' at coordinates: {screen_region}")
        else:
            print(f"Window with title '{args.window}' not found. Exiting.")
            return
    elif all([args.x, args.y, args.width, args.height]):
        screen_region = {'top': args.y, 'left': args.x, 'width': args.width, 'height': args.height}
        print(f"Capturing custom region: {screen_region}")
    else:
        screen_region = get_primary_monitor_info()
        print(f"No custom region or window specified. Capturing primary monitor with dimensions: {screen_region}")

    if args.save:
        # Create the 'captures' directory if it doesn't exist
        save_dir = "captures"
        os.makedirs(save_dir, exist_ok=True)

        with mss.mss() as sct:
            sct_img = sct.grab(screen_region)
            frame_np = np.array(sct_img)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_BGRA2BGR)

            # Resize the frame by 50% for Retina
            height, width, _ = frame_bgr.shape
            resized_frame = cv2.resize(frame_bgr, (width // 2, height // 2), interpolation=cv2.INTER_AREA)

            filename = os.path.join(save_dir, f"capture_{int(time.time())}.png")
            cv2.imwrite(filename, resized_frame)
            print(f"Saved capture to {os.path.abspath(filename)}")
            return

    # Store recent frames and their timestamps
    frames_buffer = []

    # Get the start time to calculate FPS
    start_time = time.time()

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
                first_frame = frames_buffer[0] if frames_buffer else {'timestamp': start_time}

                last_frame = frames_buffer[-1] if frames_buffer else {'timestamp': start_time}

                # print("Last frame:", last_frame)
                last_frame_time = last_frame['timestamp']
                first_frame_time = first_frame['timestamp']
                # Set the current time to the time of capture, before processing.
                current_time = time.time()

                # For unprocessed testing of frame buffer and base FPS
                # annotated_frame = frame
                # parsed_numbers, annotated_frame = analyze_frame_for_numbers(frame)
                annotated_frame = frame_to_edges(frame)

                # Add the current frame and its timestamp to the buffer
                # TODO: We should store processed frames so we are only processing them once before
                #       comparison with other frames in the buffer.
                frames_buffer.append({'frame': annotated_frame, 'timestamp': current_time})

                # Keep the buffer at a fixed size
                if len(frames_buffer) > frame_buffer_size:
                    frames_buffer.pop(0)

                fps = 1 / (current_time - last_frame_time)

                elapsed_time = current_time - first_frame_time
                avg_fps = len(frames_buffer) / elapsed_time

                change = calculate_edge_change(frames_buffer)

                cv2.putText(annotated_frame, f"FPS: {fps:.2f} AVG: {avg_fps:.2f} change: {change:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)

                cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                print("Quit signal received from keyboard.")
                quit_event.set()

    except Exception as e:
        print(f"An unexpected error occurred in the main thread: {e}")
        traceback.print_exc()

    finally:
        if not quit_event.is_set():
            quit_event.set()

        cv2.destroyAllWindows()
        print("Main thread shutting down.")

if __name__ == "__main__":
    main()
