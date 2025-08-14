import cv2
import numpy as np
import pytesseract

def analyze_frame_for_numbers(frame):
    """
    Analyzes a single frame to find and OCR numbers.
    Args:
        frame: A numpy array representing the image frame.
    Returns:
        A tuple containing:
        - A dictionary of parsed numbers.
        - The original frame with detected numbers and boxes drawn on it.
    """
    # Convert the frame to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to get a black and white image
    # The THRESH_OTSU flag automatically determines the optimal threshold value
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create a kernel for morphological transformations
    # A 15x3 rectangular kernel is good for connecting horizontal segments of numbers
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))

    # Apply dilation to fill in gaps in numbers (e.g., a broken '8')
    # This helps find a single contour for each number
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # --- New code to visualize the dilated image ---
    cv2.imshow("Dilated Image", dilated)
    # -----------------------------------------------

    # Find contours in the dilated image
    # RETR_EXTERNAL retrieves only the external contours, ignoring nested ones
    # CHAIN_APPROX_SIMPLE compresses horizontal and vertical segments
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list to store the detected numbers and their bounding boxes
    detected_numbers = []
    annotated_frame = frame.copy()

    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw a rectangle around the contour for visualization
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Print the box for debugging
        # Note: These coordinates are for the resized frame. For the original
        # screen, you would need to scale them up by 2 (e.g., x * 2, y * 2, etc.)
        print(f"box: {(x, y, w, h)}")

        # Extract the region of interest (ROI) for OCR
        roi = annotated_frame[y:y+h, x:x+w]

        # Use Tesseract to perform OCR on the ROI
        text = pytesseract.image_to_string(roi, config='--psm 6 outputbase digits').strip()

        if text.isdigit():
            detected_numbers.append({'number': int(text), 'box': (x, y, w, h)})
            # Optionally, you can draw the detected number on the frame as well
            cv2.putText(annotated_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return detected_numbers, annotated_frame
