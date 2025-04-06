import mediapipe as mp
import cv2
from massDistributionCalculator import MassDistributionCalculator

class StabilityAnalyzer:
    def __init__(self):
        """Initialize pose classes and mass calculator."""
        self.mp_pose = mp.solutions.pose
        self.pose_classes = {
            "standing_poses": {"standing_pose_side", "warrier_pose_side", "tree_pose_side", "standing_toe_front", "upward_salute"},
            "ground_contact_poses": {"sitting_toe", "triangle", "downward_dog", "child_pose", "left_leg_lunge"}
        }
        self.mass_calculator = MassDistributionCalculator()

    def analyze(self, pose, landmarks, frame, w, h):
        """
        Analyze stability and draw bounding box and CoM.
        
        Args:
            pose (str): Detected pose name.
            landmarks (list): List of pose landmarks.
            frame (numpy.ndarray): Video frame to draw on.
            w (int): Frame width.
            h (int): Frame height.
        """
        if pose == "Unknown Pose":
            return

        mass_distribution = self.mass_calculator.compute_mass_distribution(landmarks)
        com = self.mass_calculator.compute_com(landmarks)
        if not com:
            return

        if pose in self.pose_classes["standing_poses"]:
            color = (0, 255, 0)  # Green
            keypoints = [
                self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
                self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
            ]
        elif pose in self.pose_classes["ground_contact_poses"]:
            color = (255, 0, 0)  # Blue
            keypoints = [
                self.mp_pose.PoseLandmark.LEFT_ANKLE.value, self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
                self.mp_pose.PoseLandmark.LEFT_WRIST.value, self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                self.mp_pose.PoseLandmark.LEFT_KNEE.value, self.mp_pose.PoseLandmark.RIGHT_KNEE.value
            ]

        x_coords = [landmarks[point].x * w for point in keypoints if point < len(landmarks)]
        y_coords = [landmarks[point].y * h for point in keypoints if point < len(landmarks)]

        if x_coords and y_coords:
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            padding_x = 0.05 * w
            padding_y = 0.05 * h
            min_x, max_x = min_x - padding_x, max_x + padding_x
            min_y, max_y = min_y - padding_y, max_y + padding_y

            cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), color, 2)

            com_x, com_y = int(com[0] * w), int(com[1] * h)
            cv2.circle(frame, (com_x, com_y), 5, (0, 255, 255), -1)

            if pose in self.pose_classes["ground_contact_poses"]:
                stability_margin = 0.1 * h
                is_stable = (min_x <= com_x <= max_x) and ((min_y - stability_margin) <= com_y <= (max_y + stability_margin))
            else:
                is_stable = (min_x <= com_x <= max_x) and (min_y <= com_y <= max_y)

            stability_text = "Stable" if is_stable else "Unstable"
            cv2.putText(frame, stability_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            with open("mass_distribution.txt", "a") as f:
                f.write(f"{mass_distribution}\n")