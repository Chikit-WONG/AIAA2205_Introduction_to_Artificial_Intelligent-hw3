import os
import subprocess

# 视频文件夹路径
video_folder = "./data/hw3_videos/hw3_videos/videos"
output_folder = "./data/hw3_16fpv/hw3_16fpv"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历所有视频文件
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(video_folder, video_file)
        output_path = os.path.join(output_folder, video_file.replace(".mp4", ""))
        os.makedirs(output_path, exist_ok=True)

        # 使用 ffmpeg 提取帧（每秒 30 帧）
        command = [
            "ffmpeg",
            "-i",
            video_path,
            "-vf",
            "fps=30",
            os.path.join(output_path, "frame_%05d.jpg"),
        ]
        subprocess.run(command)
