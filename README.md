# Satellite Image Download, Upscaling, and Animation Tool

This project fetches satellite images from the EUMETSAT Web Coverage Service (WCS), upscales them using a deep learning model, and creates smooth animations with optional Schlieren overlays. It is designed to work with satellite data to generate visually enhanced animations and analyze atmospheric or surface changes.

## Video Demonstration

### standard Satpic animation (control)

[![Watch the Video](https://img.youtube.com/vi/HgiFrnu-MsQ/0.jpg)](https://youtube.com/shorts/HgiFrnu-MsQ?feature=share)

### upscaled and smoothed

[![Watch the Video](https://img.youtube.com/vi/FmvXWt1_9tQ/0.jpg)](https://youtube.com/shorts/FmvXWt1_9tQ?feature=share)

### schlieren effect to identify Crowing and decaying Cu

[![Watch the Video](https://img.youtube.com/vi/3bEoKRruEMQ/0.jpg)](https://youtube.com/shorts/3bEoKRruEMQ?feature=share)

## Features

1. **Satellite Image Download**:
   - Downloads satellite images from the EUMETSAT WCS.
   - Supports fetching images at regular intervals within a specified time range and bounding box.

2. **Image Upscaling**:
   - Upscales images using the Enhanced Deep Residual Networks (LapSRN) model via OpenCV's `dnn_superres` module.

3. **Frame Interpolation**:
   - Interpolates between frames using optical flow to create smooth animations.

4. **Schlieren Visualization**:
   - Generates Schlieren-like visualizations of atmospheric changes by comparing consecutive frames.
   - Applies color mapping for enhanced visualization.

5. **Video Generation**:
   - Converts processed frames into high-quality videos.

## Requirements

- Python 3.8 or later
- Required Python packages:
  - `numpy`
  - `opencv-python`
  - `opencv-contrib-python`
  - `requests`

## Setup

### Installing Dependencies

Install the necessary Python packages using `pip`:
```bash
pip install numpy opencv-python opencv-contrib-python requests
```

### Usage

### Setting the Bounding Box and Time Range

To specify the geographic area and time range for downloading satellite images:

1. **Bounding Box**: 
   - Define the geographical region with latitude and longitude values in `[min_lon, min_lat, max_lon, max_lat]` format.
   - Example:
     ```python
     bbox = [17.5, -25, 20.5, -21]  # Bounding box for Namibia
     ```

2. **Time Range**:
   - Specify the start and end times in ISO 8601 format (`YYYY-MM-DDTHH:MM:SS`).
   - Example:
     ```python
     start_time = "2024-11-14T12:01:00"  # Start of the time range
     end_time = "2024-11-14T13:01:00"    # End of the time range
     ```

3. **Time Interval**:
   - Define the time interval between images in minutes:
     ```python
     interval = 10  # Time interval in minutes
     ```

Update these variables in the script before running it.

### Running the Script

1. Download images by running the script:
   ```bash
   python3 satvid.py
    ```




