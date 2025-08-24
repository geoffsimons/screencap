import cv2
import numpy as np

# A list to store our data points, as this allows for more flexible
# time-based culling. Each element will be a tuple: (timestamp, value).
data_series = []

def create_graph_image(data_points, options):
    """
    Draws a real-time line graph on a NumPy array using OpenCV.
    This is much more efficient than using matplotlib for every frame.

    Args:
        data_points (list): A list of (timestamp, value) tuples.
        options (dict): A dictionary containing graph configuration,
                        including 'height', 'width', and 'seconds'.

    Returns:
        np.ndarray: A BGR-formatted image array suitable for OpenCV display.
    """
    # Create a blank black canvas for the graph.
    height = options['height']
    width = options['width']
    graph_img = np.zeros((height, width, 3), dtype=np.uint8)

    if not data_points:
        return graph_img

    # Normalize data to fit within the graph dimensions.
    # We need to find the min/max values and timestamps to scale the graph.
    timestamps = [point[0] for point in data_points]
    values = [point[1] for point in data_points]

    min_time = min(timestamps)
    max_time = max(timestamps)
    min_value = min(values)
    max_value = max(values)

    # Add some padding to the y-axis for better visualization.
    y_padding = (max_value - min_value) * 0.1
    min_value_padded = min_value - y_padding
    max_value_padded = max_value + y_padding

    # Create a list to hold the normalized pixel points.
    points = []

    for t, v in data_points:
        # Normalize the time (x-axis) to the graph width.
        if (max_time - min_time) == 0:
            x = 0
        else:
            x = int((t - min_time) / (max_time - min_time) * width)

        # Normalize the value (y-axis) to the graph height.
        if (max_value_padded - min_value_padded) == 0:
            y = 0
        else:
            y = int((v - min_value_padded) / (max_value_padded - min_value_padded) * height)

        # The y-axis in images goes from top to bottom, so we invert the y-coordinate.
        y = height - y

        points.append((x, y))

    # Convert the list of tuples to a NumPy array for cv2.polylines.
    points_np = np.array(points, np.int32).reshape((-1, 1, 2))

    # Draw the line graph on the canvas.
    cv2.polylines(graph_img, [points_np], isClosed=False, color=(255, 255, 0), thickness=2)

    # --- Draw axis lines and labels (optional, for clarity) ---
    # We can add text labels for the axes.
    cv2.putText(graph_img, "Value", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(graph_img, "Time (s)", (width - 100, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return graph_img
