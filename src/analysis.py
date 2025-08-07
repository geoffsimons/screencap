import cv2
import numpy as np
import pytesseract

def get_numbers_from_roi(roi_image):
    """
    Performs OCR on a pre-cropped ROI image.
    This is a simplified version of our previous function.
    """
    # Pre-process the image specifically for OCR
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Perform OCR, configured to recognize only digits
    ocr_result = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789')

    return ocr_result.strip()

def analyze_frame_for_numbers(frame):
    """
    Detects numeric ROIs in a frame and performs OCR on each.
    Returns a list of tuples: (parsed_text, x, y, w, h).
    """
    results = []

    # 1. Pre-process the frame for contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply a morphological operation to connect close characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(gray, kernel, iterations=1)

    # 2. Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size and aspect ratio to find potential text
    min_area = 100
    min_width = 10
    min_height = 10
    max_aspect_ratio = 5

    # 3. Loop over the contours and filter them
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        print(f"box: {(x,y,w,h)}")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if cv2.contourArea(contour) > min_area and w > min_width and h > min_height and w/h < max_aspect_ratio:
            # 4. Extract the ROI
            roi = frame[y:y+h, x:x+w]

            # 5. Perform OCR
            parsed_text = get_numbers_from_roi(roi)

            # 6. Add results if OCR was successful
            if parsed_text:
                results.append((parsed_text, x, y, w, h))

                # Optional: draw a green rectangle on the original frame to visualize the detected ROI
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return results, frame
