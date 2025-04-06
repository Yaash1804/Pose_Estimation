from videoProcessor import VideoProcessor
import os

if __name__ == "__main__":
    video_path = "inputVideos/Ameya_Bhujangasana.mp4"

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"outputVideos/{base_name}_output.mp4"
    processor = VideoProcessor(video_path, output_path)
    processor.process()