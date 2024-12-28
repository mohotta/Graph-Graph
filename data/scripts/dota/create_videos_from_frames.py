import argparse
import os
from sklearn.model_selection import train_test_split
import cv2

"""
    create videos from frames given in original DoTA dataset
"""

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', dest='in_dir', help='The directory to the Dataset', type=str, default="data/dota/frames")
    parser.add_argument('--out_dir', dest='out_dir', help='The directory to the output files.', type=str, default="data/dota/videos")

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


def create_video(image_folder, output_video, fps=10):

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    length = len(images)
    images.sort()

    if len(images) > 0:

        # Get size from first image
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, _ = frame.shape

        # Define the video codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # Write each image as a frame in the video
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        video.release()

    return length


def generate_videos(videos, input_base_path, output_base_path):

    os.makedirs(output_base_path, exist_ok=True)
    lengths = []

    for video in videos:

        print("processing:", video)

        length = create_video(os.path.join(input_base_path, video, 'images'), os.path.join(output_base_path, f'{video}.avi'))
        lengths.append(length)

    return lengths


if __name__ == "__main__":

    args = parse_args()
    input_dir = args.in_dir
    output_dir = args.out_dir

    all_videos = os.listdir(input_dir)
    # image list have parent folder
    all_videos.remove('frames')

    train_videos, test_videos = train_test_split(all_videos, test_size=0.2, random_state=42)

    train_lengths = generate_videos(train_videos, input_dir, os.path.join(output_dir, "training"))
    test_lengths = generate_videos(test_videos, input_dir, os.path.join(output_dir, "testing"))

    # print(max(train_lengths), min(train_lengths))
    # print(max(test_lengths), min(test_lengths))
