import mediapipe as mp
import cv2
import math
from utils import calculate_angle

class AngleCalculator:
    def __init__(self):
        """Initialize the AngleCalculator with MediaPipe pose landmarks."""
        self.mp_pose = mp.solutions.pose

    def calculate_elbow_angles(self, frame, landmarks, w, h):
        """
        Calculate and draw elbow angles.
        
        Args:
            frame (numpy.ndarray): Video frame to draw on.
            landmarks (list): List of pose landmarks.
            w (int): Frame width.
            h (int): Frame height.
        
        Returns:
            list: [left_elbow_angle, right_elbow_angle]
        """
        left_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                         int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h))
        left_elbow = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w),
                      int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h))
        left_wrist = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * w),
                      int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * h))
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        cv2.putText(frame, str(int(left_elbow_angle)), (left_elbow[0] - 50, left_elbow[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        radius = 25
        start_angle = math.degrees(math.atan2(left_shoulder[1] - left_elbow[1], left_shoulder[0] - left_elbow[0]))
        end_angle = math.degrees(math.atan2(left_wrist[1] - left_elbow[1], left_wrist[0] - left_elbow[0]))
        cv2.ellipse(frame, left_elbow, (radius, radius), 0, start_angle, end_angle, (0, 255, 255), 1)

        right_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                          int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h))
        right_elbow = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w),
                       int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h))
        right_wrist = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w),
                       int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h))
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        cv2.putText(frame, str(int(right_elbow_angle)), (right_elbow[0] - 50, right_elbow[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        start_angle = math.degrees(math.atan2(right_shoulder[1] - right_elbow[1], right_shoulder[0] - right_elbow[0]))
        end_angle = math.degrees(math.atan2(right_wrist[1] - right_elbow[1], right_wrist[0] - right_elbow[0]))
        cv2.ellipse(frame, right_elbow, (radius, radius), 0, start_angle, end_angle, (0, 255, 255), 1)

        return [left_elbow_angle, right_elbow_angle]

    def calculate_knee_angles(self, frame, landmarks, w, h):
        """Calculate and draw knee angles."""
        left_hip = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                    int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h))
        left_knee = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w),
                     int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h))
        left_ankle = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w),
                      int(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h))
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        cv2.putText(frame, str(int(left_knee_angle)), (left_knee[0] - 50, left_knee[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        radius = 25
        start_angle = math.degrees(math.atan2(left_hip[1] - left_knee[1], left_hip[0] - left_knee[0]))
        end_angle = math.degrees(math.atan2(left_ankle[1] - left_knee[1], left_ankle[0] - left_knee[0]))
        cv2.ellipse(frame, left_knee, (radius, radius), 0, start_angle, end_angle, (255, 0, 255), 1)

        right_hip = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                     int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * h))
        right_knee = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w),
                      int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h))
        right_ankle = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w),
                       int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h))
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        cv2.putText(frame, str(int(right_knee_angle)), (right_knee[0] - 50, right_knee[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        start_angle = math.degrees(math.atan2(right_hip[1] - right_knee[1], right_hip[0] - right_knee[0]))
        end_angle = math.degrees(math.atan2(right_ankle[1] - right_knee[1], right_ankle[0] - right_knee[0]))
        cv2.ellipse(frame, right_knee, (radius, radius), 0, start_angle, end_angle, (255, 255, 0), 1)

        return [left_knee_angle, right_knee_angle]

    def calculate_waist_angles(self, frame, landmarks, w, h):
        """Calculate and draw waist angles."""
        left_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                         int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h))
        left_hip = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                    int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h))
        left_knee = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w),
                     int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h))
        left_waist_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        cv2.putText(frame, str(int(left_waist_angle)), (left_hip[0] - 50, left_hip[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        radius = 25
        start_angle = math.degrees(math.atan2(left_shoulder[1] - left_hip[1], left_shoulder[0] - left_hip[0]))
        end_angle = math.degrees(math.atan2(left_knee[1] - left_hip[1], left_knee[0] - left_hip[0]))
        cv2.ellipse(frame, left_hip, (radius, radius), 0, start_angle, end_angle, (255, 0, 255), 1)

        right_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                          int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h))
        right_hip = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                     int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * h))
        right_knee = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w),
                      int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h))
        right_waist_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        cv2.putText(frame, str(int(right_waist_angle)), (right_hip[0] - 50, right_hip[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        start_angle = math.degrees(math.atan2(right_shoulder[1] - right_hip[1], right_shoulder[0] - right_hip[0]))
        end_angle = math.degrees(math.atan2(right_knee[1] - right_hip[1], right_knee[0] - right_hip[0]))
        cv2.ellipse(frame, right_hip, (radius, radius), 0, start_angle, end_angle, (255, 0, 255), 1)

        return [left_waist_angle, right_waist_angle]

    def calculate_shoulder_angles(self, frame, landmarks, w, h):
        """Calculate and draw shoulder angles."""
        left_hip = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                    int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h))
        left_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                         int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h))
        left_elbow = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w),
                      int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h))
        left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
        cv2.putText(frame, str(int(left_shoulder_angle)), (left_shoulder[0] - 50, left_shoulder[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        radius = 25
        start_angle = math.degrees(math.atan2(left_hip[1] - left_shoulder[1], left_hip[0] - left_shoulder[0]))
        end_angle = math.degrees(math.atan2(left_elbow[1] - left_shoulder[1], left_elbow[0] - left_shoulder[0]))
        cv2.ellipse(frame, left_shoulder, (radius, radius), 0, start_angle, end_angle, (255, 0, 255), 1)

        right_hip = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                     int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * h))
        right_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                          int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h))
        right_elbow = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w),
                       int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h))
        right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
        cv2.putText(frame, str(int(right_shoulder_angle)), (right_shoulder[0] - 50, right_shoulder[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        start_angle = math.degrees(math.atan2(right_hip[1] - right_shoulder[1], right_hip[0] - right_shoulder[0]))
        end_angle = math.degrees(math.atan2(right_elbow[1] - right_shoulder[1], right_elbow[0] - right_shoulder[0]))
        cv2.ellipse(frame, right_shoulder, (radius, radius), 0, start_angle, end_angle, (255, 0, 255), 1)

        return [left_shoulder_angle, right_shoulder_angle]