Pose Detection and Analysis System
This repository contains a Python-based system for detecting and analyzing human poses in video files using the MediaPipe library. The system calculates joint angles, classifies poses (e.g., yoga poses like Warrior Pose or Bhujangasana), computes the center of mass (CoM), evaluates stability, and tracks movement speeds. The codebase is modular, with separate files and classes for each functionality, ensuring maintainability and scalability.

Features
Pose Detection: Extracts landmarks from video frames using MediaPipe.
Angle Calculation: Computes angles for elbows, knees, waist, and shoulders with visual overlays.
Pose Classification: Identifies specific poses based on predefined angle thresholds.
Mass Distribution & CoM: Estimates body mass distribution and calculates the weighted CoM.
Stability Analysis: Assesses pose stability by checking CoM against a bounding box.
Speed Tracking: Measures the speed of key landmarks (e.g., wrists, nose) over time.
Video Processing: Processes input videos and saves annotated output.

Prerequisites
Python: Version 3.7 or higher.
Dependencies:
opencv-python: For video processing and visualization.
mediapipe: For pose detection.
numpy: For numerical computations.

Installation
Clone the repository:
git clone https://github.com/your-username/pose-detection-analysis.git
cd pose-detection-analysis

Install dependencies:
pip install opencv-python mediapipe numpy

Usage
Place your input video (e.g., warrior_pose.mp4) in the project directory.
Run the main script:
python main.py

The system will:
Process the video frame-by-frame.
Save the output as warrior_pose_output.mp4 in the same directory.

Customizing Input/Output
Modify main.py to change the input video path or output suffix:
video_path = "path/to/your/video.mp4"
output_path = "path/to/your/output_video.mp4"
processor = VideoProcessor(video_path, output_path)
processor.process()

File Structure
pose-detection-analysis/
├── utils.py                # Utility functions (e.g., angle calculation)
├── pose_detector.py        # Pose detection using MediaPipe
├── angle_calculator.py     # Joint angle computation and visualization
├── pose_classifier.py      # Pose classification based on angle thresholds
├── mass_distribution_calculator.py  # Mass distribution and CoM calculation
├── stability_analyzer.py   # Stability analysis with bounding box
├── speed_calculator.py     # Speed tracking for landmarks
├── video_processor.py      # Video processing pipeline
├── main.py                 # Entry point to run the system
└── README.md               # Project documentation

File Descriptions
utils.py: Contains calculate_angle for computing angles between three points.
pose_detector.py: Implements PoseDetector class for landmark extraction.
angle_calculator.py: Implements AngleCalculator class for joint angle calculations and drawing.
pose_classifier.py: Implements PoseClassifier class with pose-specific angle thresholds.
mass_distribution_calculator.py: Implements MassDistributionCalculator for weighted CoM computation.
stability_analyzer.py: Implements StabilityAnalyzer for balance assessment.
speed_calculator.py: Implements SpeedCalculator for tracking landmark speeds.
video_processor.py: Implements VideoProcessor to orchestrate the pipeline.
main.py: Launches the system with configurable input/output paths.

Example Output
For an input video warrior_pose.mp4:

Output: warrior_pose_output.mp4
Annotations: Joint angles, pose name, CoM, stability status, and speeds overlaid on each frame.

Adding New Poses
To add a new pose (e.g., Bhujangasana), update the pose_thresholds dictionary in pose_classifier.py:

"cobra_pose" : {
    "elbow_left": (140, 180),
    "elbow_right": (140, 180),
    "shoulder_left": (40, 65),
    "shoulder_right": (40, 65),
    "knee_left": (160, 180),
    "knee_right": (160, 180),
    "waist_left": (140, 160),
    "waist_right": (140, 160),
}
