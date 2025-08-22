# 3D Cube Pose Estimation using ArUco Markers
This project uses an Intel RealSense camera and ArUco markers to estimate the 3D position and orientation of a specific object (a cube) and its keypoints in real-time. It calculates the target marker's pose relative to a reference marker, enabling stable tracking regardless of the camera's position.

## Demo

A demo showing real-time tracking of the cube's center and its keypoints.

## Key Features

- **Relative Pose Estimation**: Calculates the target object's 3D position and orientation relative to the reference marker's coordinate system.

- **Keypoint Tracking**: Tracks the 3D coordinates of predefined keypoints on the cube's body, layers, etc.

- **Real-time Visualization**: Displays detected markers, coordinate axes, and keypoints on the live camera feed.

- **Easy Configuration**: Key parameters like marker IDs, sizes, and keypoint offsets can be easily changed within the code.

## Requirements
- Python 3.10

### Required Libraries:

- OpenCV (opencv-python)

- NumPy

- SciPy

- d435cam (Custom RealSense camera library)

You can install the necessary libraries with the following command:

```python
pip install opencv-python numpy scipy
```

## How to Use
### Hardware Setup

Connect the RealSense camera to your computer.

Place the reference marker (`REFERENCE_MARKER_ID`) in a fixed position and attach the target marker (`TARGET_MARKER_ID`) to the object you want to track.

### Code Configuration

Open the `get_cube_rel_center_keypoint_pose.py` file.

Modify the values for REFERENCE_MARKER_ID, TARGET_MARKER_ID, MARKER_LENGTH_M (marker size), and the cube/keypoint offsets (e.g., CUBE_SIDE_LENGTH) to match your setup.

### Run the Script

Execute the following command in your terminal:
```python
python get_cube_rel_center_keypoint_pose.py
```
Press the 'q' key in the display window to quit the program.
