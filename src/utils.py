import mss

def get_primary_monitor_info():
    """
    Returns the bounding box information for the primary monitor.
    mss.monitors[0] is a special dict for all monitors combined.
    mss.monitors[1] is the first actual monitor.
    """
    with mss.mss() as sct:
        if len(sct.monitors) > 1:
            return sct.monitors[1]
        else:
            # Fallback for systems with only one display
            return sct.monitors[0]