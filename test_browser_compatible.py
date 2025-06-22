import cv2

def test_browser_compatible_video():
    video_path = 'overlay_browser_compatible.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return False
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Browser-compatible video info: {width}x{height}, {frame_count} frames, {fps} fps")
    
    frames_tested = 0
    non_black_frames = 0
    
    for i in range(min(5, frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_tested += 1
        mean_val = frame.mean()
        if mean_val > 1:
            non_black_frames += 1
            print(f"Frame {i}: Non-black content detected (mean={mean_val:.2f})")
        else:
            print(f"Frame {i}: Black or nearly black (mean={mean_val:.2f})")
    
    cap.release()
    
    print(f"Summary: {non_black_frames}/{frames_tested} frames have visible content")
    return non_black_frames > 0

if __name__ == "__main__":
    test_browser_compatible_video()
