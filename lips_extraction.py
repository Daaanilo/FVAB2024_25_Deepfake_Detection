import os
import cv2
import shutil

def extract_frames_from_videos(input_dir, output_dir, folder_name):
    """
    Extract frames from videos in a given directory and save them to an output directory.

    Args:
        input_dir (str): Path to the directory containing videos.
        output_dir (str): Path to the directory to save extracted frames.
        folder_name (str): Subdirectory name to split real and fake videos.
    """
    # Create subdirectory for the specified folder_name (real or fake)
    split_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(split_output_dir, exist_ok=True)

    for video_file in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_file)

        # Skip non-video files
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Skipping non-video file: {video_file}")
            continue
        
        # Create a folder for the current video
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(split_output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        # Read the video
        cap = cv2.VideoCapture(video_path)

        frame_number = 0
        saved_frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save every frame (assuming 20 FPS for all videos)
            frame_filename = f"frame_lips{saved_frame_count + 1:03d}.jpg"
            frame_path = os.path.join(video_output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1

            frame_number += 1

        cap.release()
        print(f"Extracted {saved_frame_count} frames from {video_file} to {video_output_dir}")

# Paths for "real" and "fake" datasets
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
celeb_real_path = os.path.join(desktop_path, "celeb_real")  # Path to your "real" folder
celeb_fake_path = os.path.join(desktop_path, "celeb_fake")  # Path to your "fake" folder
output_path = os.path.join(desktop_path, "frames")          # Path to your output folder

# Extract frames from both datasets and split into "real" and "fake" folders
extract_frames_from_videos(celeb_real_path, output_path, folder_name="real")
extract_frames_from_videos(celeb_fake_path, output_path, folder_name="fake")

# Zip the output folder for easier handling
zip_output_path = f"{output_path}.zip"
shutil.make_archive(output_path, 'zip', output_path)
print(f"Output folder zipped as {zip_output_path}")