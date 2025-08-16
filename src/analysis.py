import cv2
import numpy as np

def analyze_frame_for_components(frame, debug_mode=False):
    """
    Analyzes a single frame to find the bounding boxes of all major components.

    This function uses a combination of grayscale conversion, adaptive thresholding,
    and morphological operations to isolate potential regions of interest (ROIs)
    and returns a list of their bounding boxes. It is generalized to find any
    distinct shapes, not just numbers.

    Args:
        frame: A numpy array representing the image frame.
        debug_mode: A boolean to control the display of intermediate debug windows.

    Returns:
        A tuple containing:
        - A list of bounding box tuples (x, y, w, h) for all detected components.
        - The original frame with detected boxes drawn on it for visualization.
    """
    # Convert the frame to grayscale for simpler processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to handle varying lighting conditions and backgrounds.
    # We use a small block size (21) and a constant (4) to get a detailed
    # black and white image.
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    # --- New logic to handle both light and dark backgrounds ---
    # Create the inverted threshold image
    thresh_inv = cv2.bitwise_not(thresh)
    # ----------------------------------------------------------

    # Create a kernel for morphological transformations. A 3x3 rectangular kernel
    # is a good starting point for a secondary dilation to connect fragmented shapes.
    # You can experiment with other kernel shapes like cv2.MORPH_ELLIPSE or cv2.MORPH_CROSS
    # for different effects.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Apply dilation to "fatten" up the shapes and close any small gaps.
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Also dilate the inverted image
    dilated_inv = cv2.dilate(thresh_inv, kernel, iterations=1)

    # --- New Edge Detection Examples (for debugging) ---
    if debug_mode:
        # First, apply a Gaussian blur to reduce noise. This helps the Canny
        # detector find cleaner, more significant edges.
        # blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # cv2.imshow("Blurred Image", blurred_gray)

        # Example 1: Canny edge detection on the blurred grayscale image.
        # The two thresholds (100 and 200) control what is considered an edge.
        # A pixel is considered an edge if its gradient is between these two values.
        # canny_edges = cv2.Canny(blurred_gray, 100, 200)
        # cv2.imshow("Blurred Canny Edges", canny_edges)

        # Example 2: Canny on unblurred image
        unblurred_canny_edges = cv2.Canny(gray, 100, 200)
        cv2.imshow("Canny Edges", unblurred_canny_edges)
    # --------------------------------------------------

    if debug_mode:
        cv2.imshow("Original Threshold", thresh)
        # cv2.imshow("Inverted Threshold", thresh_inv)
        cv2.imshow("Dilated Image", dilated)
        # cv2.imshow("Inverted Dilated Image", dilated_inv)

    # Combine the contours from both the dilated and inverted dilated images
    # to capture both light-on-dark and dark-on-light components.
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_inv, _ = cv2.findContours(dilated_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_contours = contours + contours_inv

    # Create a list to store the detected bounding boxes
    detected_boxes = []
    annotated_frame = frame.copy()

    # To avoid duplicate boxes from the dilated and inverted images, we'll
    # use a check to see if a box is already in our list.
    unique_boxes = set()

    for contour in all_contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Apply basic size filtering to remove noise (e.g., very small contours)
        # We also filter based on aspect ratio to potentially separate labels from other shapes.
        aspect_ratio = w / float(h)
        if w > 5 and h > 5 and (0.1 < aspect_ratio < 10.0):
            # Check for duplicates before adding the box
            box_tuple = (x, y, w, h)
            if box_tuple not in unique_boxes:
                # Draw a green rectangle around the contour on the annotated frame
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Add the bounding box to our list and unique set
                detected_boxes.append(box_tuple)
                unique_boxes.add(box_tuple)

    # Sort the boxes for consistent output, e.g., by y then x coordinate
    detected_boxes.sort(key=lambda b: (b[1], b[0]))

    return detected_boxes, annotated_frame
