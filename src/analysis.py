import cv2
import numpy as np

def analyze_frame_for_components(frame):
    """
    Analyzes a single frame to find the bounding boxes of all major components.

    This function uses a combination of grayscale conversion, adaptive thresholding,
    and morphological operations to isolate potential regions of interest (ROIs)
    and returns a list of their bounding boxes. It is generalized to find any
    distinct shapes, not just numbers.

    Args:
        frame: A numpy array representing the image frame.

    Returns:
        A tuple containing:
        - A list of bounding box tuples (x, y, w, h) for all detected components.
        - The original frame with detected boxes drawn on it for visualization.
    """
    # Convert the frame to grayscale for simpler processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to handle varying lighting conditions and backgrounds.
    # This automatically determines the threshold for small regions.
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    # Create a kernel for morphological transformations to help connect
    # fragmented or broken shapes. A rectangular kernel is effective for text and grids.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))

    # Apply dilation to "fatten" up the shapes and close any small gaps.
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # For debugging, you can still view the effect of the dilation.
    cv2.imshow("Dilated Image", dilated)

    # Find contours in the dilated image. A contour is a curve joining
    # all continuous points (along the boundary) having the same color or intensity.
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list to store the detected bounding boxes
    detected_boxes = []
    annotated_frame = frame.copy()

    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Apply basic size filtering to remove noise (e.g., very small contours)
        if w > 10 and h > 10:
            # Draw a green rectangle around the contour on the annotated frame
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add the bounding box to our list
            detected_boxes.append((x, y, w, h))

    return detected_boxes, annotated_frame
