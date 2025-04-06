from videoProcessor import VideoProcessor

if __name__ == "__main__":
    video_path = "warrior_pose.mp4"
    output_path = "output_video.mp4"
    processor = VideoProcessor(video_path, output_path)
    processor.process()