import cv2
import mediapipe as mp
import time
from poseDetector import PoseDetector
from angleCalculator import AngleCalculator
from poseClassifier import PoseClassifier
from stabilityAnalyzer import StabilityAnalyzer
from speedCalculator import SpeedCalculator

class VideoProcessor:
    def __init__(self, video_path, output_path):
        """Initialize video capture, output writer, and processing classes."""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error opening video")
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   self.fps, (self.frame_width, self.frame_height))

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_detector = PoseDetector()
        self.angle_calculator = AngleCalculator()
        self.pose_classifier = PoseClassifier()
        self.stability_analyzer = StabilityAnalyzer()
        self.speed_calculator = SpeedCalculator()

    def process(self):
        """Process the video frame by frame."""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            landmarks = self.pose_detector.detect(frame)
            if landmarks:
                landmark_list = landmarks.landmark
                w, h = self.frame_width, self.frame_height
                current_time = time.time()

                elbow_angles = self.angle_calculator.calculate_elbow_angles(frame, landmark_list, w, h)
                knee_angles = self.angle_calculator.calculate_knee_angles(frame, landmark_list, w, h)
                waist_angles = self.angle_calculator.calculate_waist_angles(frame, landmark_list, w, h)
                shoulder_angles = self.angle_calculator.calculate_shoulder_angles(frame, landmark_list, w, h)

                angles = {
                    "elbow": elbow_angles,
                    "knee": knee_angles,
                    "waist": waist_angles,
                    "shoulder": shoulder_angles
                }

                pose = self.pose_classifier.classify(angles)
                self.stability_analyzer.analyze(pose, landmark_list, frame, w, h)
                self.speed_calculator.calculate_speeds(frame, landmark_list, w, h, current_time)

                cv2.putText(frame, pose, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_pose.POSE_CONNECTIONS)

            self.out.write(frame)
            cv2.imshow("Skeleton", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()