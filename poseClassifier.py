class PoseClassifier:
    def __init__(self):
        """Initialize pose thresholds."""
        self.pose_thresholds = {
        "standing_pose_side": {
            "elbow_left": (128, 189),
            "elbow_right": (146, 207),
            "shoulder_left": (71, 132),
            "shoulder_right": (72, 132),
            "knee_left": (132, 193),
            "knee_right": (136, 197),
            "waist_left": (12, 73),
            "waist_right": (14, 75),
        },
        "warrier_pose_side": {
            "elbow_left": (149, 210),
            "elbow_right": (146, 207),
            "shoulder_left": (65, 126),
            "shoulder_right": (79, 140),
            "knee_left": (144, 205),
            "knee_right": (92, 153),
            "waist_left": (124, 185),
            "waist_right": (85, 146),
        },
        "tree_pose_side": {
            "elbow_left": (129, 190),
            "elbow_right": (134, 195),
            "shoulder_left": (145, 206),
            "shoulder_right": (149, 210),
            "knee_left": (146, 207),
            "knee_right": (22, 83),
            "waist_left": (149, 210),
            "waist_right": (106, 167),
        },
        "sitting_toe": {
            "elbow_left": (137, 198),
            "elbow_right": (131, 192),
            "shoulder_left": (66, 127),
            "shoulder_right": (63, 124),
            "knee_left": (141, 202),
            "knee_right": (145, 206),
            "waist_left": (22, 83),
            "waist_right": (21, 82),
        },
        "standing_toe_front": {
            "elbow_left": (115, 176),
            "elbow_right": (147, 208),
            "shoulder_left": (130, 191),
            "shoulder_right": (126, 187),
            "knee_left": (140, 201),
            "knee_right": (130, 191),
            "waist_left": (0, 53),
            "waist_right": (0, 44),
        },
        "triangle": {
            "elbow_left": (141, 202),
            "elbow_right": (132, 193),
            "shoulder_left": (148, 209),
            "shoulder_right": (149, 210),
            "knee_left": (39, 100),
            "knee_right": (95, 156),
            "waist_left": (46, 107),
            "waist_right": (106, 167),
        },
        "downward_dog": {
            "elbow_left": (147, 208),
            "elbow_right": (148, 209),
            "shoulder_left": (128, 189),
            "shoulder_right": (125, 186),
            "knee_left": (148, 209),
            "knee_right": (149, 210),
            "waist_left": (21, 82),
            "waist_right": (18, 79),
        },
        "child_pose": {
            "elbow_left": (121, 182),
            "elbow_right": (123, 184),
            "shoulder_left": (115, 176),
            "shoulder_right": (117, 178),
            "knee_left": (3, 64),
            "knee_right": (0, 60),
            "waist_left": (0, 58),
            "waist_right": (0, 57),
        },
        "left_leg_lunge": {
            "elbow_left": (139, 200),
            "elbow_right": (139, 200),
            "shoulder_left": (65, 126),
            "shoulder_right": (74, 135),
            "knee_left": (145, 206),
            "knee_right": (131, 192),
            "waist_left": (101, 162),
            "waist_right": (17, 78),
        },
        "upward_salute": {
            "elbow_left": (129, 190),
            "elbow_right": (131, 192),
            "shoulder_left": (141, 202),
            "shoulder_right": (139, 200),
            "knee_left": (145, 206),
            "knee_right": (148, 209),
            "waist_left": (139, 200),
            "waist_right": (143, 204),
        },

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