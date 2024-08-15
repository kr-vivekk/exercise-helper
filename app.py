import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe pose and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Counters and stages for left and right arm repetitions
left_counter = 0
left_stage = None

right_counter = 0
right_stage = None

def calculate_angles(a, b, c):
    """
    Calculate the angle between three points.
    
    Parameters:
    a (list): Coordinates of the first point.
    b (list): Coordinates of the second point (the vertex of the angle).
    c (list): Coordinates of the third point.
    
    Returns:
    float: The calculated angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

# Start MediaPipe pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Perform pose detection
        result = pose.process(image)
        
        # Convert the image back to BGR for OpenCV rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            # Extract landmarks
            landmarks = result.pose_landmarks.landmark
            
            # Right arm coordinates
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Calculate the angle of the right arm
            right_angle = calculate_angles(right_shoulder, right_elbow, right_wrist)
            cv2.putText(image, str(right_angle), 
                        tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Left arm coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate the angle of the left arm
            left_angle = calculate_angles(left_shoulder, left_elbow, left_wrist)
            cv2.putText(image, str(left_angle), 
                        tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Determine the stage and count for right arm
            if right_angle > 160:
                right_stage = "down"
            if right_angle < 30 and right_stage == 'down':
                right_stage = "up"
                right_counter += 1
            
            # Determine the stage and count for left arm
            if left_angle > 160:
                left_stage = "down"
            if left_angle < 30 and left_stage == 'down':
                left_stage = "up"
                left_counter += 1
            
        except:
            pass
        
        # Display the repetition count and stage for both arms
        cv2.rectangle(image, (0, 0), (260, 73), (255, 255, 255), -1)
        cv2.rectangle(image, (370, 0), (800, 73), (255, 255, 255), -1)
        
        # Right arm information
        cv2.putText(image, 'R REPS', (15, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(right_counter), 
                    (10, 60), 
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 51, 153), 2, cv2.LINE_AA)
        cv2.putText(image, 'R STAGE', (90, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(right_stage), 
                    (90, 60), 
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 51, 153), 1, cv2.LINE_AA)
        
        # Left arm information
        cv2.putText(image, 'L REPS', (395, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(left_counter), 
                    (390, 60), 
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 51, 153), 2, cv2.LINE_AA)
        cv2.putText(image, 'L STAGE', (470, 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(left_stage), 
                    (470, 60), 
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 51, 153), 1, cv2.LINE_AA)
        
        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        
        # Display the image
        cv2.imshow('MediaPipe Feed', image)
        
        # Break the loop on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
