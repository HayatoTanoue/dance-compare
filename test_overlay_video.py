import cv2
import sys

def test_overlay_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {video_path}")
        return False
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {width}x{height}, {frame_count} frames, {fps} fps")
    
    frames_tested = 0
    non_black_frames = 0
    
    for i in range(min(10, frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_tested += 1
        if frame.mean() > 1:  # If average pixel value > 1, it's not black
            non_black_frames += 1
            print(f"Frame {i}: Non-black content detected (mean={frame.mean():.2f})")
        else:
            print(f"Frame {i}: Black or nearly black (mean={frame.mean():.2f})")
    
    cap.release()
    
    print(f"Summary: {non_black_frames}/{frames_tested} frames have visible content")
    return non_black_frames > 0

if __name__ == "__main__":
    video_path = "/tmp/tmplecmhrni_teacher_overlay.mp4"
    test_overlay_video(video_path)
