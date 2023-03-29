# YOLO-Directional-People-Counter
An application that utilizes YOLOv4 object detection to count people and track their movement direction in a video stream, providing visualizations and count statistics.

## Requirements
- Python 3.6 or later
- OpenCV
- NumPy
## Installation
1. Clone the repository:
git clone https://github.com/yourusername/yolov4-directional-people-counter.git
2. Change to the project directory:
cd yolov4-directional-people-counter
3. Install the required packages:
pip install -r requirements.txt

## Usage

1. Download the YOLOv4 weights and configuration files, and place them in the project directory.

2. Update the `points` variable in the `main.py` file to match the polygon's corner points you want to use for your specific video.

3. Run the application with the following command:
python main.py

4. Press 'q' to stop the video stream and close the application.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
