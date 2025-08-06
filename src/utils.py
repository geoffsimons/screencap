import mss
import Quartz

def get_primary_monitor_info():
    """
    Returns the coordinates of the primary monitor.
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        return {
            'top': monitor['top'],
            'left': monitor['left'],
            'width': monitor['width'],
            'height': monitor['height']
        }

def find_all_window_coordinates(window_title):
    """
    Finds all windows with a given title and returns a list of their coordinates.
    Returns a list of dictionaries [{'top', 'left', 'width', 'height', 'window_name'}, ...]
    """
    window_list = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly,
        Quartz.kCGNullWindowID
    )

    found_windows = []
    for window in window_list:
        if 'kCGWindowName' in window and window['kCGWindowName'] == window_title:
            bounds = window['kCGWindowBounds']
            found_windows.append({
                'window_name': window['kCGWindowName'],
                'top': bounds['Y'],
                'left': bounds['X'],
                'width': bounds['Width'],
                'height': bounds['Height']
            })

    return found_windows
