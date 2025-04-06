import mediapipe as mp
import cv2

class PoseDetector:
    def __init__(self):
        """Initialize the MediaPipe Pose detector."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2)

    def detect(self, frame):
        """
        Detect pose landmarks in the given frame.
        
        Args:
            frame (numpy.ndarray): Input video frame in BGR format.
        
        Returns:
            NormalizedLandmarkList: Detected pose landmarks, or None if not detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results.pose_landmarks