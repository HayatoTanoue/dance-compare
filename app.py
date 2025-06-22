import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class PoseAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_poses(self, video_path):
        """動画から骨格情報を抽出"""
        cap = cv2.VideoCapture(video_path)
        poses = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                poses.append(landmarks)
            else:
                poses.append([0] * (33 * 4))  # 33個のランドマーク × 4次元
            
            frame_count += 1
        
        cap.release()
        return np.array(poses), frame_count
    
    def draw_pose_on_frame(self, frame, pose_landmarks):
        """フレームに骨格を描画"""
        if pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame

def main():
    st.set_page_config(
        page_title="Dance Compare",
        page_icon="💃",
        layout="wide"
    )
    
    st.title("💃 Dance Compare")
    st.markdown("ダンスの練習が一人でできるようなインターフェースと解析機能")
    
    st.sidebar.header("動画アップロード")
    
    teacher_video = st.sidebar.file_uploader(
        "教師動画をアップロード",
        type=['mp4', 'avi', 'mov'],
        key="teacher"
    )
    
    student_video = st.sidebar.file_uploader(
        "生徒動画をアップロード", 
        type=['mp4', 'avi', 'mov'],
        key="student"
    )
    
    col1, col2 = st.columns(2)
    
    analyzer = PoseAnalyzer()
    
    with col1:
        st.header("教師動画")
        if teacher_video is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(teacher_video.read())
                teacher_path = tmp_file.name
            
            st.video(teacher_path)
            
            if st.button("教師動画の骨格推定を実行", key="teacher_analyze"):
                with st.spinner("骨格推定中..."):
                    poses, frame_count = analyzer.extract_poses(teacher_path)
                    st.session_state.teacher_poses = poses
                    st.session_state.teacher_frame_count = frame_count
                    st.success(f"骨格推定完了！ {frame_count}フレーム処理しました")
            
            if st.checkbox("骨格重畳表示", key="teacher_overlay"):
                if 'teacher_poses' in st.session_state:
                    st.info("骨格重畳表示機能は実装中です")
                else:
                    st.warning("まず骨格推定を実行してください")
    
    with col2:
        st.header("生徒動画")
        if student_video is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(student_video.read())
                student_path = tmp_file.name
            
            st.video(student_path)
            
            if st.button("生徒動画の骨格推定を実行", key="student_analyze"):
                with st.spinner("骨格推定中..."):
                    poses, frame_count = analyzer.extract_poses(student_path)
                    st.session_state.student_poses = poses
                    st.session_state.student_frame_count = frame_count
                    st.success(f"骨格推定完了！ {frame_count}フレーム処理しました")
            
            if st.checkbox("骨格重畳表示", key="student_overlay"):
                if 'student_poses' in st.session_state:
                    st.info("骨格重畳表示機能は実装中です")
                else:
                    st.warning("まず骨格推定を実行してください")
    
    st.header("比較分析")
    
    if ('teacher_poses' in st.session_state and 
        'student_poses' in st.session_state):
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("時間同期設定")
            teacher_start = st.slider(
                "教師動画開始時刻 (秒)",
                0.0, 
                float(st.session_state.teacher_frame_count / 30),  # 30fps仮定
                0.0,
                0.1
            )
            student_start = st.slider(
                "生徒動画開始時刻 (秒)",
                0.0,
                float(st.session_state.student_frame_count / 30),  # 30fps仮定
                0.0,
                0.1
            )
        
        with col4:
            st.subheader("分析結果")
            if st.button("比較分析を実行"):
                with st.spinner("分析中..."):
                    teacher_poses = st.session_state.teacher_poses
                    student_poses = st.session_state.student_poses
                    
                    min_frames = min(len(teacher_poses), len(student_poses))
                    teacher_sync = teacher_poses[:min_frames]
                    student_sync = student_poses[:min_frames]
                    
                    similarities = []
                    for i in range(min_frames):
                        t_pose = np.array(teacher_sync[i])
                        s_pose = np.array(student_sync[i])
                        
                        if np.linalg.norm(t_pose) > 0 and np.linalg.norm(s_pose) > 0:
                            similarity = np.dot(t_pose, s_pose) / (np.linalg.norm(t_pose) * np.linalg.norm(s_pose))
                            similarities.append(max(0, similarity))  # 負の値は0にクリップ
                        else:
                            similarities.append(0)
                    
                    avg_similarity = np.mean(similarities)
                    st.metric("平均類似度", f"{avg_similarity:.3f}")
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(similarities)
                    ax.set_xlabel('フレーム')
                    ax.set_ylabel('類似度')
                    ax.set_title('ポーズ類似度の時系列変化')
                    ax.grid(True)
                    st.pyplot(fig)
    
    else:
        st.info("比較分析を行うには、両方の動画で骨格推定を実行してください")

if __name__ == "__main__":
    main()
