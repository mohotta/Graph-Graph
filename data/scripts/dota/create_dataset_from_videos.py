import argparse
import os
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import numpy as np
import json
import cv2
import os.path as osp
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image

"""
    create a dataset for graph-graph using any videoset 
    need
        - videoset seperate in two folders positive and negative (anormaly happen in positive)
        - time to anormaly for positive videos in a text file as the same name of the videos
"""

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', dest='video_dir', help='The directory to the Dataset', type=str, default="data/dota/videos")
    parser.add_argument('--toa_dir', dest='toa_dir', help='The directory to the output files.', type=str, default="data/dota/toas")
    parser.add_argument('--output_dir', dest='output_dir', help='The directory to the output files.', type=str, default="data/dota")
    args = parser.parse_args()
    return args

## --------------------------------------------------------------------------------------------
## detections

args = parse_args()

video_dir = args.video_dir
toa_dir = args.toa_dir
output_dir = args.output_dir

print('started detection!')

det_dir = osp.join(output_dir, 'detections')
os.makedirs(det_dir, exist_ok=True)

model = YOLO('yolov8s.pt')

# saving anno file
names = list(model.names.values())
with open(osp.join(output_dir, "obj_idx_to_labels.json"), "w") as f:
    json.dump(names, f)

shape_needed = (50, 19, 6)

for root, _, files in os.walk(video_dir):
    for file in files:

        file_path = osp.join(root, file)
        output_file = osp.join(det_dir, file[:-4] + ".npy")

        print("processing:", file_path)

        results = model(file_path)
        detections = []

        if len(results) > 0:
            arr = map(lambda x: x.boxes.data.cpu().numpy().astype(np.float16), results)
            for item in arr:
                if len(item) > shape_needed[1]:
                    item = item[:shape_needed[1]]
                else:
                    item = np.pad(item, ((0, shape_needed[1]-item.shape[0]), (0, 0)), mode='constant', constant_values=0)

                detections.append(item)
            detections = np.array(detections)
        else:
            detections = np.zeros(shape_needed)

        print(detections.shape)
        assert detections.shape == shape_needed

        np.save(output_file, detections)

        print("saved:", output_file)

## -----------------------------------------------------------------------------------------------
## frames_stats

def get_frames_stats(video_path):

    print("processing: ", video_path)

    frame_stats = []

    capture = cv2.VideoCapture(video_path)
    frame_index = 0

    while capture.isOpened():

        ret, frame = capture.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frame_stats.append(np.array([height, width]))
        frame_index += 1

    capture.release()

    frame_stats = np.array(frame_stats)

    print(frame_stats.shape)

    return frame_stats

print('started frames_stats!')

frames_dir = osp.join(output_dir, 'frames_stats')

os.makedirs(frames_dir, exist_ok=True)
os.makedirs(osp.join(frames_dir, "training"), exist_ok=True)
os.makedirs(osp.join(frames_dir, "training", "negative"), exist_ok=True)
os.makedirs(osp.join(frames_dir, "training", "positive"), exist_ok=True)
os.makedirs(osp.join(frames_dir, "testing"), exist_ok=True)
os.makedirs(osp.join(frames_dir, "testing", "negative"), exist_ok=True)
os.makedirs(osp.join(frames_dir, "testing", "positive"), exist_ok=True)

for root, _, files in os.walk(video_dir):
    for file in files:
        frame_stats = get_frames_stats(osp.join(root, file))
        out_file = osp.join(frames_dir, root.split("/")[-2], root.split("/")[-1], file[:-4] + ".npy")
        print("save to:", out_file)
        np.save(out_file, frame_stats)

## -----------------------------------------------------------------------------------------------
## obj_feat

CLASSES = ('__background__', 'Car', 'Pedestrian', 'Cyclist')

n_frames=50
n_boxes=19
dim_feat=4096

def get_video_frames(video_file, n_frames=50):
    # get the video data
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    video_data = []
    counter = 0
    while (ret):
        video_data.append(frame)
        ret, frame = cap.read()
        counter += 1
    assert counter == n_frames
    return video_data


def bbox_to_imroi(bboxes, image):
    """
    bboxes: (n, 4), ndarray
    image: (H, W, 3), ndarray
    """
    imroi_data = []
    for bbox in bboxes:
        imroi = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        imroi = transform(Image.fromarray(imroi))  # (3, 224, 224), torch.Tensor
        imroi_data.append(imroi)
    imroi_data = torch.stack(imroi_data)
    return imroi_data


def get_boxes(dets_all, im_size):
    bboxes = []
    for bbox in dets_all:
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        x1 = min(max(0, x1), im_size[1] - 1)  # 0<=x1<=W-1
        y1 = min(max(0, y1), im_size[0] - 1)  # 0<=y1<=H-1
        x2 = min(max(x1, x2), im_size[1] - 1)  # x1<=x2<=W-1
        y2 = min(max(y1, y2), im_size[0] - 1)  # y1<=y2<=H-1
        h = y2 - y1 + 1
        w = x2 - x1 + 1
        if h > 2 and w > 2:  # the area is at least 9
            bboxes.append([x1, y1, x2, y2])
    bboxes = np.array(bboxes, dtype=np.int32)
    return bboxes


def extract_features(detections_path, video_path, dest_path, phase):

    for root, _, files in os.walk(video_path):
        for file in files:
            video_file = os.path.join(root, file)
            print("processing:", video_file)
            video_frames = get_video_frames(video_file, n_frames=n_frames)
            detections_file = os.path.join(detections_path, file[:-4] + ".npy")
            detections = np.load(detections_file)
            label = np.array([0,1]) if root.split("/")[-1] == "positive" else np.array([1,0])
            feat_file = os.path.join(dest_path, file[:-4] + ".npz")

            features_vgg16 = np.zeros((n_frames, n_boxes + 1, dim_feat), dtype=np.float32)


            for i, frame in tqdm(enumerate(video_frames), total=len(video_frames)):
                bboxes = get_boxes(detections[i], frame.shape)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                with torch.no_grad():
                    image = transform(Image.fromarray(frame))
                    ims_frame = torch.unsqueeze(image, dim=0).float().to(device=device)
                    feature_frame = torch.squeeze(feat_extractor(ims_frame))
                    features_vgg16[i, 0, :] = feature_frame.cpu().numpy() if feature_frame.is_cuda else feature_frame.detach().numpy()

                    # extract object feature
                    if len(bboxes) > 0:
                        # bboxes to roi data
                        ims_roi = bbox_to_imroi(bboxes, frame)  # (n, 3, 224, 224)
                        ims_roi = ims_roi.float().to(device=device)
                        feature_roi = torch.squeeze(torch.squeeze(feat_extractor(ims_roi), dim=-1), dim=-1)  # (4096,)
                        features_vgg16[i, 1:len(bboxes) + 1, :] = feature_roi.cpu().numpy() if feature_roi.is_cuda else feature_roi.detach().numpy()

                    np.savez_compressed(feat_file, data=features_vgg16, det=detections, labels=label, ID=file[:-4])


def run(detections_path, video_path, dest_path):
    # prepare the result paths
    train_path = osp.join(dest_path, 'training')
    if not osp.exists(train_path):
        os.makedirs(train_path)
    test_path = osp.join(dest_path, 'testing')
    if not osp.exists(test_path):
        os.makedirs(test_path)

    # process training set
    extract_features(detections_path, osp.join(video_path, 'training'), train_path, 'training')
    # process testing set
    extract_features(detections_path, osp.join(video_path, 'testing'), test_path, 'testing')

print('start obj_feat!')

feat_extractor = models.vgg16(pretrained=True)
feat_extractor.classifier = nn.Sequential(*list(feat_extractor.classifier.children())[:-1])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
feat_extractor = feat_extractor.to(device=device)
feat_extractor.eval()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()]
)

detections_path = osp.join(output_dir, 'detections')
run(detections_path, video_dir, osp.join(output_dir, 'obj_feat'))

## -----------------------------------------------------------------------------------------------
## i3d_feat (actually using vgg16)

feat_dir = osp.join(output_dir, 'obj_feat')
i3d_dir = osp.join(output_dir, "i3d_feat")

os.makedirs(i3d_dir, exist_ok=True)
os.makedirs(os.path.join(i3d_dir, "training"), exist_ok=True)
os.makedirs(os.path.join(i3d_dir, "training", "negative"), exist_ok=True)
os.makedirs(os.path.join(i3d_dir, "training", "positive"), exist_ok=True)
os.makedirs(os.path.join(i3d_dir, "testing"), exist_ok=True)
os.makedirs(os.path.join(i3d_dir, "testing", "negative"), exist_ok=True)
os.makedirs(os.path.join(i3d_dir, "testing", "positive"), exist_ok=True)

for root, _, files in os.walk(video_dir):
    for file in files:
        print("processing", file)
        features = np.load(os.path.join(feat_dir, root.split("/")[-2], file[:-4] + ".npz"))["data"]
        out_file = os.path.join(i3d_dir, root.split("/")[-2], root.split("/")[-1], file[:-4] + ".npy")
        np.save(out_file, features[:, 0, :].reshape(50, 4096))
        print("saved", out_file)

## -----------------------------------------------------------------------------------------------
## splits_dota

print('start split_dota!')

base_dir = osp.join(output_dir, 'obj_feat')
# split_dir = osp.join(output_dir, 'splits_dota')
split_dir = 'splits_dota'

os.makedirs(split_dir, exist_ok=True)

for _, dirs, _ in os.walk(base_dir):
    if len(dirs) > 0:
        for dir in dirs:
            print("processing", dir)
            with open(osp.join(split_dir, f"{'train' if dir == 'training' else 'test'}_split.txt"), "w") as f:
                for _, _, files in os.walk(os.path.join(base_dir, dir)):
                    f.write("\n".join(files))

## ------------------------------------------------------------------------------------------------

print("done!")
