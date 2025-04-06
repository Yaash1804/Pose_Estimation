import mediapipe as mp
import cv2
import math

class SpeedCalculator:
    def __init__(self):
        """Initialize previous positions and time."""
        self.mp_pose = mp.solutions.pose
        self.prev_left_wrist = None
        self.prev_right_wrist = None
        self.prev_nose = None
        self.prev_time = None

    def calculate_speeds(self, frame, landmarks, w, h, current_time):
        """
        Calculate and display speeds of wrists and nose.
        
        Args:
            frame (numpy.ndarray): Video frame to draw on.
            landmarks (list): List of pose landmarks.
            w (int): Frame width.
            h (int): Frame height.
            current_time (float): Current timestamp.
        """
        left_wrist = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * w),
                      int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * h))
        right_wrist = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w),
                       int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h))
        nose = (int(landmarks[self.mp_pose.PoseLandmark.NOSE.value].x * w),
                int(landmarks[self.mp_pose.PoseLandmark.NOSE.value].y * h))

        if self.prev_time is None:
            self.prev_left_wrist = left_wrist
            self.prev_right_wrist = right_wrist
            self.prev_nose = nose
            self.prev_time = current_time
            return

        time_diff = current_time - self.prev_time
        if time_diff == 0:
            return

        left_distance = math.sqrt((left_wrist[0] - self.prev_left_wrist[0])**2 + (left_wrist[1] - self.prev_left_wrist[1])**2)
        left_speed = left_distance / time_diff
        cv2.putText(frame, f"{int(left_speed)} px/s", (left_wrist[0] - 50, left_wrist[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        right_distance = math.sqrt((right_wrist[0] - self.prev_right_wrist[0])**2 + (right_wrist[1] - self.prev_right_wrist[1])**2)
        right_speed = right_distance / time_diff
        cv2.putText(frame, f"{int(right_speed)} px/s", (right_wrist[0] - 50, right_wrist[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        nose_distance = math.sqrt((nose[0] - self.prev_nose[0])**2 + (nose[1] - self.prev_nose[1])**2)
        nose_speed = nose_distance / time_diff
        cv2.putText(frame, f"{int(nose_speed)} px/s", (nose[0] - 50, nose[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self.prev_left_wrist = left_wrist
        self.prev_right_wrist = right_wrist
        self.prev_nose = nose
        self.prev_time = current_time