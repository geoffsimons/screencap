# macOS Screencapture Analysis App

A macOS application to capture screen content and perform real-time video analysis using Python, `mss`, and `OpenCV`.

## Features

- **Real-time Capture and Analysis:** Captures and processes screen frames in a single, high-performance loop.
- **Immediate Feedback:** Ensures the application is always working on the freshest available frame.
- **Configurable:** Easily select the screen region to capture (e.g., full screen, window, or specific area).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/macos-screencapture-app.git](https://github.com/yourusername/macos-screencapture-app.git)
    cd macos-screencapture-app
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python3 src/main.py
    ```

## Usage

- The application will open an `OpenCV` window displaying the captured screen content.
- Press `q` while the window is focused to quit the application.

## Directory Structure

- `src/`: Contains all the application's source code.
    - `main.py`: The single file containing the main capture and analysis loop.
    - `utils.py`: Helper functions, like getting monitor details.
- `requirements.txt`: Python package dependencies.
- `README.md`: This file.