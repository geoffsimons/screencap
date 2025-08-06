# Real-Time macOS Screencapture and OCR

This project is a foundation for real-time screen capture and analysis on macOS. It uses a multi-threaded architecture to capture a video feed from the screen and perform OCR on specific regions, which is especially useful for analyzing numeric displays in applications or games.

-----

### Key Features

  * **Multi-threaded Performance**: A dedicated thread handles screen capture to avoid freezing the main application and its display window.
  * **Dynamic Window Targeting**: The application can automatically find and capture a specific window by its title, such as "Google Chrome".
  * **Flexible Capture Regions**: You can choose to capture a specific window, a custom-defined area of the screen, or the entire primary monitor.
  * **Frame Saving for Analysis**: A command-line option allows you to save a single captured frame to a local file for focused testing and refinement of the OCR logic.
  * **Modular Analysis**: The OCR logic is separated into a dedicated `analysis.py` file, making it easy to swap out or improve the analysis process.

-----

### Prerequisites

To run this application, you'll need **Python 3** and the following dependencies. These are split into general requirements and platform-specific packages.

#### General Requirements (to be included in `requirements.txt`)

  * `mss`
  * `numpy`
  * `pytesseract`

#### Platform-Specific Requirements (sideloaded as needed)

  * **`opencv`**: While a Python package exists, it's highly recommended to install it via Homebrew on macOS for a stable GUI backend.
  * **`pyobjc-framework-Quartz`**: This is a macOS-specific library required for interacting with the windowing system.

-----

### Installation

1.  **Clone the repository and navigate to the project directory:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install core dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Install macOS-specific packages**:

      * **OpenCV**: If you don't have Homebrew installed, you'll need to install it first. Then, install OpenCV:
        ```bash
        brew install opencv
        ```
      * **PyObjC-Framework-Quartz**:
        ```bash
        pip install pyobjc-framework-Quartz
        ```

4.  **Grant Screen Recording Permission**: You **must** grant your terminal application (e.g., Terminal, iTerm2, VS Code) "Screen Recording" permission in **System Settings \> Privacy & Security \> Screen Recording**. Without this, the application will not be able to capture the screen.

-----

### Usage

The application uses command-line arguments to specify its behavior.

#### Live Capture of a Specific Window

Run the application with the `--window` flag and the exact title of the window you want to capture.

```bash
python3 src/main.py --window "Google Chrome"
```

#### Live Capture of a Custom Screen Region

Use the `--x`, `--y`, `--width`, and `--height` flags to define a custom capture box.

```bash
python3 src/main.py --x 100 --y 200 --width 400 --height 300
```

#### Save a Single Frame

Use the `--save` flag to capture a single frame from the specified window or region and save it to the `captures/` directory, then exit.

```bash
python3 src/main.py --window "Google Chrome" --save
```

This will create a new image file in the `captures/` directory. Be sure to add `captures/` to your `.gitignore` file to prevent these temporary files from being committed.

-----

### Project Structure

```
/project_root
├── .gitignore
├── README.md
├── requirements.txt
├── /src
│   ├── __init__.py
│   ├── main.py
│   ├── utils.py
│   └── analysis.py
└── /captures
    └── capture_1678886400.png  # (Example saved image)
```
