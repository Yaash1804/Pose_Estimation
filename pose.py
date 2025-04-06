import cv2
import mediapipe as mp
import numpy as np
import math
import time
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def visualize_mass_distribution(landmarks):
    """Computes the mass distribution in 3D without visualization."""
    mass_distribution = {}  # Dictionary to store mass distribution

    for landmark, mass in BODY_MASS_DISTRIBUTION.items():
        lm = landmarks[landmark.value]
        if lm.visibility > 0.5:  # Consider only visible landmarks
            x, y, z = lm.x, lm.y, lm.z
            mass_distribution[str(landmark)] = (x, y, z, mass)  # Store in dictionary

    return mass_distribution  # Return computed mass distribution



def pose_detection(elbow_angles,knee_angles,waise_angles,shoulder_angles):

    pose_thresholds = {
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
    }

    
}

    # Create a detected_angles dictionary
    detected_angles = {
        "elbow_left": elbow_angles[0],
        "elbow_right": elbow_angles[1],
        "knee_left": knee_angles[0],
        "knee_right": knee_angles[1],
        "waist_left": waist_angles[0],
        "waist_right": waist_angles[1],
        "shoulder_left": shoulder_angles[0],
        "shoulder_right": shoulder_angles[1]
    }


    # Default pose
    detected_pose = "Unknown Pose"

    mass_distributions = []  # List to store mass distributions for each valid pose
    pose_frames = []  # List to store valid pose frames

    for pose, thresholds in pose_thresholds.items():
        match = all(
            thresholds[joint][0] <= detected_angles[joint] <= thresholds[joint][1]
            for joint in thresholds
        )

        if match:
            mass_distribution = visualize_mass_distribution(landmarks)  # Compute mass distribution
            mass_distributions.append(mass_distribution)
            pose_frames.append(landmarks)  # Store landmarks for reference
            detected_pose = pose

    # Save only the middle frame's mass distribution
    if mass_distributions:
        middle_index = len(mass_distributions) // 2
        middle_mass_distribution = mass_distributions[middle_index]

        # Optionally, save to a file
        with open("mass_distribution.txt", "a") as f:
            f.write(f"{middle_mass_distribution}\n")



    # Write the detected pose on the video frame
    cv2.putText(frame, detected_pose, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


def shoulder_angle():
    left_hip_coor = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
    left_shoulder_coor = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                           int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
    left_elbow_coor = (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w),
                        int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h))

    left_shoulder_angle = calculateAngle(left_hip_coor, left_shoulder_coor, left_elbow_coor)

    cv2.putText(frame, str(int(left_shoulder_angle)), (left_shoulder_coor[0] - 50, left_shoulder_coor[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    left_radius = 25  
    left_start_angle = math.degrees(math.atan2(left_hip_coor[1] - left_shoulder_coor[1], 
                                               left_hip_coor[0] - left_shoulder_coor[0]))
    left_end_angle = math.degrees(math.atan2(left_elbow_coor[1] - left_shoulder_coor[1], 
                                             left_elbow_coor[0] - left_shoulder_coor[0]))

    cv2.ellipse(frame, left_shoulder_coor, (left_radius, left_radius), 0, 
                left_start_angle, left_end_angle, (255, 0, 255), 1)
    
    right_hip_coor = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                      int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
    right_shoulder_coor = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
    right_elbow_coor = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w),
                         int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h))

    right_shoulder_angle = calculateAngle(right_hip_coor, right_shoulder_coor, right_elbow_coor)

    cv2.putText(frame, str(int(right_shoulder_angle)), (right_shoulder_coor[0] - 50, right_shoulder_coor[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    right_radius = 25
    right_start_angle = math.degrees(math.atan2(right_hip_coor[1] - right_shoulder_coor[1],
                                                 right_hip_coor[0] - right_shoulder_coor[0]))
    right_end_angle = math.degrees(math.atan2(right_elbow_coor[1] - right_shoulder_coor[1],
                                              right_elbow_coor[0] - right_shoulder_coor[0]))

    cv2.ellipse(frame, right_shoulder_coor, (right_radius, right_radius), 0,
                right_start_angle, right_end_angle, (255, 0, 255), 1)

    return [left_shoulder_angle, right_shoulder_angle]

def elbow_angle():
    left_shoulder_coor = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
    left_elbow_coor = (int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w),int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h))
    left_wrist_coor = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w),int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h))

    left_elbow_angle = calculateAngle(left_shoulder_coor, left_elbow_coor, left_wrist_coor)

    cv2.putText(frame, str(int(left_elbow_angle)), (left_elbow_coor[0] - 50, left_elbow_coor[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw arc
    left_radius = 25  # Set radius of the arc
    left_start_angle = math.degrees(math.atan2(left_shoulder_coor[1] - left_elbow_coor[1], 
                                            left_shoulder_coor[0] - left_elbow_coor[0]))
    left_end_angle = math.degrees(math.atan2(left_wrist_coor[1] - left_elbow_coor[1], 
                                            left_wrist_coor[0] - left_elbow_coor[0]))

    cv2.ellipse(frame, left_elbow_coor, (left_radius, left_radius), 0, 
                left_start_angle, left_end_angle, (0, 255, 255), 1)
    
    right_shoulder_coor = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
    right_elbow_coor = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w),
                        int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h))
    right_wrist_coor = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w),
                        int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * h))

    right_elbow_angle = calculateAngle(right_shoulder_coor, right_elbow_coor, right_wrist_coor)

    cv2.putText(frame, str(int(right_elbow_angle)), (right_elbow_coor[0] - 50, right_elbow_coor[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    right_radius = 25
    right_start_angle = math.degrees(math.atan2(right_shoulder_coor[1] - right_elbow_coor[1],
                                                right_shoulder_coor[0] - right_elbow_coor[0]))
    right_end_angle = math.degrees(math.atan2(right_wrist_coor[1] - right_elbow_coor[1],
                                            right_wrist_coor[0] - right_elbow_coor[0]))

    cv2.ellipse(frame, right_elbow_coor, (right_radius, right_radius), 0,
                right_start_angle, right_end_angle, (0, 255, 255), 1)

    return [left_elbow_angle,right_elbow_angle]

def knee_angle():
    # Left Knee
    left_hip_coor = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
    left_knee_coor = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h))
    left_ankle_coor = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h))

    left_knee_angle = calculateAngle(left_hip_coor, left_knee_coor, left_ankle_coor)

    cv2.putText(frame, str(int(left_knee_angle)), (left_knee_coor[0] - 50, left_knee_coor[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    left_knee_radius = 25
    left_knee_start_angle = math.degrees(math.atan2(left_hip_coor[1] - left_knee_coor[1],
                                                    left_hip_coor[0] - left_knee_coor[0]))
    left_knee_end_angle = math.degrees(math.atan2(left_ankle_coor[1] - left_knee_coor[1],
                                                left_ankle_coor[0] - left_knee_coor[0]))

    cv2.ellipse(frame, left_knee_coor, (left_knee_radius, left_knee_radius), 0,
                left_knee_start_angle, left_knee_end_angle, (255, 0, 255), 1)

    # Right Knee
    right_hip_coor = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
    right_knee_coor = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * h))
    right_ankle_coor = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w),
                        int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h))

    right_knee_angle = calculateAngle(right_hip_coor, right_knee_coor, right_ankle_coor)

    cv2.putText(frame, str(int(right_knee_angle)), (right_knee_coor[0] - 50, right_knee_coor[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    right_knee_radius = 25
    right_knee_start_angle = math.degrees(math.atan2(right_hip_coor[1] - right_knee_coor[1],
                                                    right_hip_coor[0] - right_knee_coor[0]))
    right_knee_end_angle = math.degrees(math.atan2(right_ankle_coor[1] - right_knee_coor[1],
                                                right_ankle_coor[0] - right_knee_coor[0]))

    cv2.ellipse(frame, right_knee_coor, (right_knee_radius, right_knee_radius), 0,
                right_knee_start_angle, right_knee_end_angle, (255, 255, 0), 1)

    return [left_knee_angle,right_knee_angle]
    
def waist_angle():
    # Left Side (Already defined)
    left_hip_coor = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
    left_shoulder_coor = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                          int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
    left_knee_coor = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w),
                      int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h))

    # Calculate Left Waist Angle
    left_waist_angle_value = calculateAngle(left_shoulder_coor, left_hip_coor, left_knee_coor)
    cv2.putText(frame, str(int(left_waist_angle_value)), 
                (left_hip_coor[0] - 50, left_hip_coor[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw Left Waist Angle Arc
    left_waist_radius = 25
    left_waist_start_angle = math.degrees(math.atan2(left_shoulder_coor[1] - left_hip_coor[1],
                                                     left_shoulder_coor[0] - left_hip_coor[0]))
    left_waist_end_angle = math.degrees(math.atan2(left_knee_coor[1] - left_hip_coor[1],
                                                   left_knee_coor[0] - left_hip_coor[0]))

    cv2.ellipse(frame, left_hip_coor, (left_waist_radius, left_waist_radius), 0,
                left_waist_start_angle, left_waist_end_angle, (255, 0, 255), 1)

    # Right Side (New addition)
    right_hip_coor = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                      int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
    right_shoulder_coor = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                           int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
    right_knee_coor = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w),
                       int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * h))

    # Calculate Right Waist Angle
    right_waist_angle_value = calculateAngle(right_shoulder_coor, right_hip_coor, right_knee_coor)
    cv2.putText(frame, str(int(right_waist_angle_value)), 
                (right_hip_coor[0] - 50, right_hip_coor[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw Right Waist Angle Arc
    right_waist_radius = 25
    right_waist_start_angle = math.degrees(math.atan2(right_shoulder_coor[1] - right_hip_coor[1],
                                                      right_shoulder_coor[0] - right_hip_coor[0]))
    right_waist_end_angle = math.degrees(math.atan2(right_knee_coor[1] - right_hip_coor[1],
                                                    right_knee_coor[0] - right_hip_coor[0]))

    cv2.ellipse(frame, right_hip_coor, (right_waist_radius, right_waist_radius), 0,
                right_waist_start_angle, right_waist_end_angle, (255, 0, 255), 1)

    return [left_waist_angle_value,right_waist_angle_value]

def wrist_speed(left_wrist,prev_left_wrist,right_wrist,prev_right_wrist,current_time,prev_time):
    time_diff = current_time - prev_time
    left_distance = math.sqrt((left_wrist[0] - prev_left_wrist[0])**2 + (left_wrist[1] - prev_left_wrist[1])**2)

    left_speed = left_distance/time_diff

    cv2.putText(frame, str(int(left_speed)) + " px/s", (left_wrist[0] - 50, left_wrist[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 0), 1)


    right_distance = math.sqrt((right_wrist[0] - prev_right_wrist[0])**2 + (right_wrist[1] - prev_right_wrist[1])**2)
    right_speed = right_distance / time_diff

    # Display right wrist speed
    cv2.putText(frame, str(int(right_speed)) + " px/s", (right_wrist[0] - 50, right_wrist[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 0), 1)  # Bright Blue Color
    

def nose_speed(curr_nose,prev_nose,current_time,prev_time):
    time_diff = current_time - prev_time
    distance = math.sqrt((curr_nose[0] - prev_nose[0])**2 + (curr_nose[1]-prev_nose[1])**2)
    speed = distance/time_diff
    cv2.putText(frame, str(int(speed)) + " px/s", (curr_nose[0] - 50, curr_nose[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def calculateAngle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba,bc)/(np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

video_path = "childposeSide.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
  print("error opening the video")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = False, model_complexity = 2)
mp_drawing = mp.solutions.drawing_utils

output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
prev_left_wrist = 0
prev_right_wrist = 0
prev_nose = 0
# left_wrist = 0
# right_wrist = 0
# nose = 0

BODY_MASS_DISTRIBUTION = {
    mp_pose.PoseLandmark.NOSE: 0.08,
    mp_pose.PoseLandmark.LEFT_SHOULDER: 0.12,
    mp_pose.PoseLandmark.RIGHT_SHOULDER: 0.12,
    mp_pose.PoseLandmark.LEFT_ELBOW: 0.03,
    mp_pose.PoseLandmark.RIGHT_ELBOW: 0.03,
    mp_pose.PoseLandmark.LEFT_WRIST: 0.02,
    mp_pose.PoseLandmark.RIGHT_WRIST: 0.02,
    mp_pose.PoseLandmark.LEFT_HIP: 0.15,
    mp_pose.PoseLandmark.RIGHT_HIP: 0.15,
    mp_pose.PoseLandmark.LEFT_KNEE: 0.08,
    mp_pose.PoseLandmark.RIGHT_KNEE: 0.08,
    mp_pose.PoseLandmark.LEFT_ANKLE: 0.05,
    mp_pose.PoseLandmark.RIGHT_ANKLE: 0.05,
}


while cap.isOpened():
    ret,frame = cap.read()

    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)



    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        h,w,_ = frame.shape

        current_time = time.time()

        elbow_angles = elbow_angle()
        knee_angles = knee_angle()
        waist_angles = waist_angle()
        shoulder_angles = shoulder_angle()

        pose_detection(elbow_angles,knee_angles,waist_angles,shoulder_angles)

        left_wrist = (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w),
                  int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h))
        right_wrist = (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h))
        
        if prev_left_wrist:
            wrist_speed(left_wrist,prev_left_wrist,right_wrist,prev_right_wrist,current_time,prev_time)

        nose = (int(landmarks[mp_pose.PoseLandmark.NOSE.value].x * w),
                int(landmarks[mp_pose.PoseLandmark.NOSE.value].y * h))
        
        if prev_nose:
            nose_speed(nose,prev_nose,current_time,prev_time)

        mp_drawing.draw_landmarks(frame,results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        prev_left_wrist = left_wrist
        prev_right_wrist = right_wrist
        prev_nose = nose
        prev_time = current_time


    cv2.imshow("Skeleton", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()