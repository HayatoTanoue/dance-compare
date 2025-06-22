import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

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
        
        self.joint_groups = {
            "上半身": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],  # 肩、肘、手首、指
            "下半身": [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],  # 腰、膝、足首、足
            "体幹": [11, 12, 23, 24],  # 肩と腰
            "左腕": [11, 13, 15, 17, 19, 21],  # 左肩から左手まで
            "右腕": [12, 14, 16, 18, 20, 22],  # 右肩から右手まで
            "左脚": [23, 25, 27, 29, 31],  # 左腰から左足まで
            "右脚": [24, 26, 28, 30, 32]   # 右腰から右足まで
        }
    
    def extract_poses(self, video_path):
        """動画から骨格情報を抽出"""
        cap = cv2.VideoCapture(video_path)
        poses = []
        frames = []
        pose_results = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            frames.append(frame.copy())
            pose_results.append(results)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                poses.append(landmarks)
            else:
                poses.append([0] * (33 * 4))
            
            frame_count += 1
        
        cap.release()
        return np.array(poses), frame_count, frames, pose_results
    
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
    
    def create_skeleton_overlay_video(self, frames, pose_results, output_path):
        """骨格重畳表示動画を作成"""
        if not frames or not pose_results:
            return None
            
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        for frame, results in zip(frames, pose_results):
            overlay_frame = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    overlay_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            out.write(overlay_frame)
        
        out.release()
        return output_path
    
    def calculate_joint_similarities(self, teacher_poses, student_poses, joint_group="全体"):
        """関節レベルでの類似度計算"""
        if joint_group == "全体":
            joint_indices = list(range(33))
        else:
            joint_indices = self.joint_groups.get(joint_group, list(range(33)))
        
        min_frames = min(len(teacher_poses), len(student_poses))
        similarities = []
        
        for i in range(min_frames):
            t_pose = np.array(teacher_poses[i]).reshape(33, 4)
            s_pose = np.array(student_poses[i]).reshape(33, 4)
            
            t_joints = t_pose[joint_indices, :3].flatten()  # x,y,z座標のみ
            s_joints = s_pose[joint_indices, :3].flatten()
            
            if np.linalg.norm(t_joints) > 0 and np.linalg.norm(s_joints) > 0:
                similarity = np.dot(t_joints, s_joints) / (np.linalg.norm(t_joints) * np.linalg.norm(s_joints))
                similarities.append(max(0, similarity))
            else:
                similarities.append(0)
        
        return similarities
    
    def create_karaoke_style_chart(self, similarities, interval_seconds=1, fps=30):
        """カラオケ風スコア表示チャート作成"""
        interval_frames = int(interval_seconds * fps)
        intervals = []
        scores = []
        
        for i in range(0, len(similarities), interval_frames):
            end_idx = min(i + interval_frames, len(similarities))
            interval_score = np.mean(similarities[i:end_idx])
            intervals.append(f"{i//fps:.1f}-{end_idx//fps:.1f}s")
            scores.append(interval_score * 100)  # パーセンテージに変換
        
        colors = []
        for score in scores:
            if score >= 80:
                colors.append('#FFD700')  # 金色
            elif score >= 60:
                colors.append('#32CD32')  # 緑
            elif score >= 40:
                colors.append('#FFA500')  # オレンジ
            else:
                colors.append('#FF6347')  # 赤
        
        fig = go.Figure(data=[
            go.Bar(
                x=intervals,
                y=scores,
                marker_color=colors,
                text=[f'{score:.1f}%' for score in scores],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="カラオケ風ダンススコア",
            xaxis_title="時間区間",
            yaxis_title="類似度スコア (%)",
            yaxis=dict(range=[0, 100]),
            showlegend=False
        )
        
        return fig

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
                    poses, frame_count, frames, pose_results = analyzer.extract_poses(teacher_path)
                    st.session_state.teacher_poses = poses
                    st.session_state.teacher_frame_count = frame_count
                    st.session_state.teacher_frames = frames
                    st.session_state.teacher_pose_results = pose_results
                    st.success(f"骨格推定完了！ {frame_count}フレーム処理しました")
            
            if st.checkbox("骨格重畳表示", key="teacher_overlay"):
                if 'teacher_poses' in st.session_state:
                    with st.spinner("骨格重畳表示動画を作成中..."):
                        overlay_path = tempfile.mktemp(suffix='_teacher_overlay.mp4')
                        result_path = analyzer.create_skeleton_overlay_video(
                            st.session_state.teacher_frames,
                            st.session_state.teacher_pose_results,
                            overlay_path
                        )
                        if result_path:
                            st.video(result_path)
                            st.success("骨格重畳表示完了！")
                        else:
                            st.error("骨格重畳表示の作成に失敗しました")
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
                    poses, frame_count, frames, pose_results = analyzer.extract_poses(student_path)
                    st.session_state.student_poses = poses
                    st.session_state.student_frame_count = frame_count
                    st.session_state.student_frames = frames
                    st.session_state.student_pose_results = pose_results
                    st.success(f"骨格推定完了！ {frame_count}フレーム処理しました")
            
            if st.checkbox("骨格重畳表示", key="student_overlay"):
                if 'student_poses' in st.session_state:
                    with st.spinner("骨格重畳表示動画を作成中..."):
                        overlay_path = tempfile.mktemp(suffix='_student_overlay.mp4')
                        result_path = analyzer.create_skeleton_overlay_video(
                            st.session_state.student_frames,
                            st.session_state.student_pose_results,
                            overlay_path
                        )
                        if result_path:
                            st.video(result_path)
                            st.success("骨格重畳表示完了！")
                        else:
                            st.error("骨格重畳表示の作成に失敗しました")
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
                float(st.session_state.teacher_frame_count / 30),
                0.0,
                0.1
            )
            student_start = st.slider(
                "生徒動画開始時刻 (秒)",
                0.0,
                float(st.session_state.student_frame_count / 30),
                0.0,
                0.1
            )
            
            st.subheader("分析設定")
            analysis_level = st.selectbox(
                "分析レベル",
                ["全体", "上半身", "下半身", "体幹", "左腕", "右腕", "左脚", "右脚"]
            )
            
            interval_seconds = st.slider(
                "カラオケ風スコア区間 (秒)",
                0.5, 3.0, 1.0, 0.5
            )
        
        with col4:
            st.subheader("分析結果")
            if st.button("比較分析を実行"):
                with st.spinner("分析中..."):
                    teacher_poses = st.session_state.teacher_poses
                    student_poses = st.session_state.student_poses
                    
                    teacher_start_frame = int(teacher_start * 30)
                    student_start_frame = int(student_start * 30)
                    
                    teacher_sync = teacher_poses[teacher_start_frame:]
                    student_sync = student_poses[student_start_frame:]
                    
                    similarities = analyzer.calculate_joint_similarities(
                        teacher_sync, student_sync, analysis_level
                    )
                    
                    if similarities:
                        avg_similarity = np.mean(similarities)
                        max_similarity = np.max(similarities)
                        min_similarity = np.min(similarities)
                        
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("平均スコア", f"{avg_similarity*100:.1f}%")
                        with col_m2:
                            st.metric("最高スコア", f"{max_similarity*100:.1f}%")
                        with col_m3:
                            st.metric("最低スコア", f"{min_similarity*100:.1f}%")
                        
                        st.subheader("🎤 カラオケ風ダンススコア")
                        karaoke_fig = analyzer.create_karaoke_style_chart(
                            similarities, interval_seconds
                        )
                        st.plotly_chart(karaoke_fig, use_container_width=True)
                        
                        st.subheader("📊 詳細分析")
                        
                        progress_fig = go.Figure()
                        
                        time_axis = np.arange(len(similarities)) / 30  # 30fps仮定
                        
                        progress_fig.add_trace(go.Scatter(
                            x=time_axis,
                            y=[s*100 for s in similarities],
                            mode='lines+markers',
                            name='類似度スコア',
                            line=dict(width=3),
                            marker=dict(size=4)
                        ))
                        
                        progress_fig.add_hline(y=80, line_dash="dash", line_color="gold", 
                                             annotation_text="優秀 (80%)")
                        progress_fig.add_hline(y=60, line_dash="dash", line_color="green", 
                                             annotation_text="良好 (60%)")
                        progress_fig.add_hline(y=40, line_dash="dash", line_color="orange", 
                                             annotation_text="要改善 (40%)")
                        
                        progress_fig.update_layout(
                            title=f"ダンス類似度の時系列変化 ({analysis_level})",
                            xaxis_title="時間 (秒)",
                            yaxis_title="類似度スコア (%)",
                            yaxis=dict(range=[0, 100]),
                            height=400
                        )
                        
                        st.plotly_chart(progress_fig, use_container_width=True)
                        
                        st.subheader("📋 分析サマリー")
                        
                        score_ranges = {
                            "優秀 (80%以上)": sum(1 for s in similarities if s >= 0.8),
                            "良好 (60-80%)": sum(1 for s in similarities if 0.6 <= s < 0.8),
                            "普通 (40-60%)": sum(1 for s in similarities if 0.4 <= s < 0.6),
                            "要改善 (40%未満)": sum(1 for s in similarities if s < 0.4)
                        }
                        
                        total_frames = len(similarities)
                        for range_name, count in score_ranges.items():
                            percentage = (count / total_frames) * 100 if total_frames > 0 else 0
                            st.write(f"**{range_name}**: {count}フレーム ({percentage:.1f}%)")
                    
                    else:
                        st.error("類似度の計算に失敗しました")
    
    else:
        st.info("比較分析を行うには、両方の動画で骨格推定を実行してください")

if __name__ == "__main__":
    main()
