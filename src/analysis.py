import cv2
import numpy as np

def analyze_frame_for_components(frame, debug_mode=False):
    """
    Analyzes a single frame to find the bounding boxes of all major components.

    This function uses a combination of grayscale conversion and edge detection
    to isolate potential regions of interest (ROIs) and returns a list of
    their bounding boxes.

    Args:
        frame: A numpy array representing the image frame.
        debug_mode: A boolean to control the display of intermediate debug windows.

    Returns:
        A tuple containing:
        - A list of bounding box tuples (x, y, w, h) for all detected components.
        - The original frame with detected boxes drawn on it for visualization.
    """
    # Create a blurred version of the image to smooth out noise.
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert the frame to grayscale for simpler processing
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection to find the outlines of components. This provides
    # a clean binary image of edges, which is perfect for contour finding.
    # The thresholds (100 and 200) are crucial and might need fine-tuning.
    canny_edges = cv2.Canny(gray, 100, 200)

    if debug_mode:
        cv2.imshow("Canny Edges", canny_edges)

    # Find contours in the Canny edge map. A contour is a curve joining
    # all continuous points (along the boundary) having the same color or intensity.
    contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list to store the detected bounding boxes
    detected_boxes = []
    annotated_frame = frame.copy()

    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Apply basic size filtering to remove noise (e.g., very small contours)
        # We also filter based on aspect ratio to potentially separate labels from other shapes.
        aspect_ratio = w / float(h)
        if w > 5 and h > 5 and (0.3 < aspect_ratio < 10.0):
            # To avoid duplicate boxes from a single component, we'll
            # use a check to see if a box is already in our list.
            box_tuple = (x, y, w, h)
            if box_tuple not in detected_boxes:
                # Draw a green rectangle around the contour on the annotated frame
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Add the bounding box to our list
                detected_boxes.append(box_tuple)

    # Sort the boxes for consistent output, e.g., by y then x coordinate
    detected_boxes.sort(key=lambda b: (b[1], b[0]))

    return detected_boxes, annotated_frame
