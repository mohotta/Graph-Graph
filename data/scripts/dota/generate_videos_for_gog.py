import argparse
import os
import json
import cv2

"""
    process videos to output videos of given frame count (50) and 
    save time to accident according to that
"""

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', dest='video_dir', help='The directory to the Dataset', type=str, default="data/dota/videos")
    parser.add_argument('--label_dir', dest='label_dir', help='The directory to the Dataset', type=str, default="data/dota/annotations")
    parser.add_argument('--out_dir', dest='out_dir', help='The directory to the output files.', type=str, default="data/dota/new_videos")
    parser.add_argument('--out_label_dir', dest='out_label_dir', help='The directory to the output files.', type=str, default="data/dota/toas")

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

def read_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def write_video(capture, video_writer, starting_frame):
    capture.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
    for frame_num in range(starting_frame, starting_frame + 50):
        ret, frame = capture.read()
        if ret:
            video_writer.write(frame)
        else:
            break


if __name__ == "__main__":

    args = parse_args()
    video_dir = args.video_dir
    annotation_dir = args.label_dir
    output_dir = args.out_dir
    toa_dir = args.out_label_dir

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "training", "positive"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "training", "negative"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "testing"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "testing", "positive"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "testing", "negative"), exist_ok=True)
    os.makedirs(toa_dir, exist_ok=True)

    positive_count = 0
    negative_count = 0

    for root, _, files in os.walk(video_dir):
        for file in files:

            print("processing:", file)

            phase = root.split('/')[-1]

            file_path = os.path.join(root, file)
            json_file = os.path.join(annotation_dir, file[:-4] + ".json")
            output_file = os.path.join(output_dir, phase, "positive", file)

            if os.path.exists(json_file):

                data = read_json(json_file)
                length = int(data["num_frames"])
                accident_frame = int(data["anomaly_start"])
                accident_end = int(data["anomaly_end"])

                print("length:", length)
                print("toa:", accident_frame)
                print("accident end", accident_end)

                if length >= 50:

                    cap = cv2.VideoCapture(file_path)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    fps = 10
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

                    ## writing positive videos
                    if length == 50:
                        toa = accident_frame
                        start_frame = 0
                    elif accident_frame < 30:
                        toa = accident_frame
                        start_frame = 0
                    else:
                        if (accident_frame - 30 + 50) < length:
                            start_frame = accident_frame - 30
                            toa = 30
                        else:
                            start_frame = (length - 50 - 1)
                            toa = accident_frame - start_frame

                    print("start:", start_frame)

                    write_video(cap, video, start_frame)

                    with open(os.path.join(toa_dir, file[:-4] + ".txt"), "w") as f:
                        f.write(str(toa))

                    print("saved:", output_file)
                    positive_count += 1

                    video.release()

                    ## writing negative videos
                    if accident_frame >= 50:
                        for i in range(accident_frame // 50):
                            video = cv2.VideoWriter(
                                os.path.join(output_dir, phase, "negative", file + "-neg-" + str(i)),
                                fourcc,
                                fps,
                                (width, height)
                            )
                            start_frame  = 0 + i * 50
                            print("found negative1:", start_frame, start_frame+50)
                            write_video(cap, video, start_frame)
                            print("saved negative1:", os.path.join(output_dir, phase, "negative", file[:-4] + "-neg-" + str(i) + ".avi"))
                            negative_count += 1
                            video.release()
                    if (length - accident_end) >= 50:
                        for i in range((length-accident_end) // 50):
                            video = cv2.VideoWriter(
                                os.path.join(output_dir, phase, "negative", file[:-4] + "-neg-2-" + str(i) + ".avi"),
                                fourcc,
                                fps,
                                (width, height)
                            )
                            start_frame = accident_end + i * 50
                            print("found negative2:", start_frame, start_frame + 50)
                            write_video(cap, video, start_frame)
                            print("saved negative2:", os.path.join(output_dir, phase, "negative", file + "-neg-2-" + str(i)))
                            negative_count += 1
                            video.release()

                    cap.release()
                    #cv2.destroyAllWindows()
    with open("meta.txt", "w") as f:
        f.write("\n".join([f"positive: {positive_count}", f"negative: {negative_count}"]))
    print("positive:", positive_count, "|", "negative:", negative_count)
    print("done!")
