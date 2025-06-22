import cv2
import numpy as np

def create_test_video(filename, duration_seconds=3):
    """Create a simple test video with a moving figure"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    width, height = 640, 480
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    total_frames = duration_seconds * fps
    
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        center_x = int(320 + 100 * np.sin(i * 0.1))
        center_y = int(240 + 50 * np.cos(i * 0.1))
        
        cv2.circle(frame, (center_x, center_y - 60), 20, (255, 255, 255), -1)
        
        cv2.line(frame, (center_x, center_y - 40), (center_x, center_y + 40), (255, 255, 255), 3)
        
        arm_angle = np.sin(i * 0.2) * 0.5
        arm_x = int(30 * np.cos(arm_angle))
        arm_y = int(30 * np.sin(arm_angle))
        cv2.line(frame, (center_x, center_y - 20), (center_x - arm_x, center_y - 20 - arm_y), (255, 255, 255), 3)
        cv2.line(frame, (center_x, center_y - 20), (center_x + arm_x, center_y - 20 + arm_y), (255, 255, 255), 3)
        
        leg_angle = np.sin(i * 0.15) * 0.3
        leg_x = int(25 * np.sin(leg_angle))
        leg_y = int(25 * np.cos(leg_angle))
        cv2.line(frame, (center_x, center_y + 40), (center_x - leg_x, center_y + 40 + leg_y), (255, 255, 255), 3)
        cv2.line(frame, (center_x, center_y + 40), (center_x + leg_x, center_y + 40 + leg_y), (255, 255, 255), 3)
        
        out.write(frame)
    
    out.release()
    print(f'Test video created: {filename}')

if __name__ == "__main__":
    create_test_video('teacher_test.mp4', 3)
    create_test_video('student_test.mp4', 3)
