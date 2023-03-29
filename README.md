# YOLO-Directional-People-Counter
An application that utilizes YOLOv4 object detection to count people and track their movement direction in a video stream, providing visualizations and count statistics.
![gif](https://user-images.githubusercontent.com/58664122/228600007-3171d86f-dc0b-4eea-ae65-37842fd976b0.gif)

## Installation
1. Clone the repository:
git clone `https://github.com/yourusername/yolov4-directional-people-counter.git`
2. Change to the project directory:
`cd yolov4-directional-people-counter`


## Usage

1. Download the YOLOv4 weights and configuration files, and place them in the project directory.

2. Update the `points` variable in the `main.py` file to match the polygon's corner points you want to use for your specific video.

3. Run the application with the following command:
`python3 main.py`

4. Press 'q' to stop the video stream and close the application.

## Important Note

This repository does not include the `video.mp4` and `yolov4.weights` files due to their size. Please download the appropriate files and place them in the project directory.

- For `yolov4.weights`, you can download the file from the official YOLOv4 repository: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
- For `video.mp4`, use a video file of your choice.

## Requirements
- Python 3.6 or later
- OpenCV
- NumPy
- 
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
