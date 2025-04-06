class PoseClassifier:
    def __init__(self):
        """Initialize pose thresholds."""
        self.pose_thresholds = {
            "standing_pose_side": {
                "elbow_left": (128, 189), "elbow_right": (146, 207),
                "shoulder_left": (71, 132), "shoulder_right": (72, 132),
                "knee_left": (132, 193), "knee_right": (136, 197),
                "waist_left": (12, 73), "waist_right": (14, 75),
            },
            "warrier_pose_side": {
                "elbow_left": (149, 210), "elbow_right": (146, 207),
                "shoulder_left": (65, 126), "shoulder_right": (79, 140),
                "knee_left": (144, 205), "knee_right": (92, 153),
                "waist_left": (124, 185), "waist_right": (85, 146),
            },
            # Add other poses similarly...
            "upward_salute": {
                "elbow_left": (129, 190), "elbow_right": (131, 192),
                "shoulder_left": (141, 202), "shoulder_right": (139, 200),
                "knee_left": (145, 206), "knee_right": (148, 209),
                "waist_left": (139, 200), "waist_right": (143, 204),
            }
        }

    def classify(self, angles):
        """
        Classify the pose based on detected angles.
        
        Args:
            angles (dict): Dictionary with angle lists (e.g., {'elbow': [left, right], ...}).
        
        Returns:
            str: Name of the detected pose or "Unknown Pose".
        """
        detected_angles = {
            "elbow_left": angles["elbow"][0], "elbow_right": angles["elbow"][1],
            "knee_left": angles["knee"][0], "knee_right": angles["knee"][1],
            "waist_left": angles["waist"][0], "waist_right": angles["waist"][1],
            "shoulder_left": angles["shoulder"][0], "shoulder_right": angles["shoulder"][1]
        }
        for pose, thresholds in self.pose_thresholds.items():
            match = all(
                thresholds[joint][0] <= detected_angles[joint] <= thresholds[joint][1]
                for joint in thresholds
            )
            if match:
                return pose
        return "Unknown Pose"