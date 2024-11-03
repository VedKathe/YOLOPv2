import cv2
import os

def video_to_all_images(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Save every frame as an image
        image_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(image_filename, frame)
        frame_count += 1

    video_capture.release()
    print(f"Saved {frame_count} images to {output_folder}")
    
# Example usage
video_path = "demo.mov"
output_folder = "input_img"
frame_rate = 1  # Capture one image per second
video_to_all_images(video_path, output_folder)
