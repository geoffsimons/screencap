import cv2
import numpy as np

def frame_to_edges(frame):
    """
    Convert the frame to it's edges using Canny
    """
    # Create a blurred version of the image to smooth out noise.
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert the frame to grayscale for simpler processing
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection to find the outlines of components. This provides
    # a clean binary image of edges, which is perfect for contour finding.
    # The thresholds (100 and 200) are crucial and might need fine-tuning.
    canny_edges = cv2.Canny(gray, 100, 200)

    return canny_edges

def calculate_edge_change(frame_buffer):
    """
    Calculates the average change in edges between consecutive frames in a buffer.

    Args:
        frame_buffer (list): A list of dictionaries, where each dictionary has
                             'frame' (a numpy array of canny edges) and 'timestamp' fields.

    Returns:
        float: A value from 0.0 to 1.0 expressing how much the edges are changing,
               where 0.0 means no change and 1.0 means the edges are completely
               different. Returns 0.0 if the buffer has fewer than two frames.
    """
    # Check if there are at least two frames to compare.
    if len(frame_buffer) < 2:
        print("Not enough frames in the buffer to calculate change. At least 2 are needed.")
        return 0.0

    total_change = 0.0
    num_comparisons = len(frame_buffer) - 1

    # Get the dimensions of the edge frames for normalization
    height, width = frame_buffer[0]['frame'].shape
    total_pixels = height * width

    # Get the first edge frame from the dictionary
    previous_edges = frame_buffer[0]['frame']

    # Iterate through the rest of the frames in the buffer
    for i in range(1, len(frame_buffer)):
        current_edges = frame_buffer[i]['frame']

        # Calculate the absolute difference between the two edge images.
        # Since Canny edges are binary (0 or 255), this will show where edges
        # appear or disappear.
        diff = cv2.absdiff(previous_edges, current_edges)

        # Sum up all the changes (the pixel values in the diff image).
        # We divide by 255 because Canny outputs 255 for edges, so we want
        # a count of changed pixels, not the sum of their values.
        change_score = np.sum(diff / 255)

        # Normalize the change score by the total number of pixels to get a value from 0 to 1.
        normalized_change = change_score / total_pixels

        # Add the normalized change to the total
        total_change += normalized_change

        # Update the previous_edges for the next iteration
        previous_edges = current_edges

    # Return the average change across all comparisons
    return total_change / num_comparisons

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
