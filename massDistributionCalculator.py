import mediapipe as mp

class MassDistributionCalculator:
    def __init__(self):
        """Initialize body mass distribution."""
        self.mp_pose = mp.solutions.pose
        self.body_mass_distribution = {
            self.mp_pose.PoseLandmark.NOSE: 0.08,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER: 0.12,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER: 0.12,
            self.mp_pose.PoseLandmark.LEFT_ELBOW: 0.03,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW: 0.03,
            self.mp_pose.PoseLandmark.LEFT_WRIST: 0.02,
            self.mp_pose.PoseLandmark.RIGHT_WRIST: 0.02,
            self.mp_pose.PoseLandmark.LEFT_HIP: 0.15,
            self.mp_pose.PoseLandmark.RIGHT_HIP: 0.15,
            self.mp_pose.PoseLandmark.LEFT_KNEE: 0.08,
            self.mp_pose.PoseLandmark.RIGHT_KNEE: 0.08,
            self.mp_pose.PoseLandmark.LEFT_ANKLE: 0.05,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE: 0.05,
        }

    def compute_mass_distribution(self, landmarks):
        """
        Compute mass distribution for visible landmarks.
        
        Args:
            landmarks (list): List of pose landmarks.
        
        Returns:
            dict: Mass distribution with landmark coordinates and masses.
        """
        mass_distribution = {}
        for landmark, mass in self.body_mass_distribution.items():
            lm = landmarks[landmark.value]
            if lm.visibility > 0.5:
                x, y, z = lm.x, lm.y, lm.z
                mass_distribution[str(landmark)] = (x, y, z, mass)
        return mass_distribution

    def compute_com(self, landmarks):
        """
        Compute the weighted center of mass.
        
        Args:
            landmarks (list): List of pose landmarks.
        
        Returns:
            tuple: (com_x, com_y) or None if no valid landmarks.
        """
        total_mass = 0
        weighted_sum_x = 0
        weighted_sum_y = 0
        for landmark, mass in self.body_mass_distribution.items():
            lm = landmarks[landmark.value]
            if lm.visibility > 0.5:
                weighted_sum_x += lm.x * mass
                weighted_sum_y += lm.y * mass
                total_mass += mass
        if total_mass > 0:
            com_x = weighted_sum_x / total_mass
            com_y = weighted_sum_y / total_mass
            return (com_x, com_y)
        return None