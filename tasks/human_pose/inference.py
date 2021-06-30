import os, time, sys
import cv2
import json
import argparse
import logging
import PIL.Image

import torch
import torch2trt
from torch2trt import TRTModule
import torchvision.transforms as transforms

from utils.utils import TrtPose, get_keypoint, execute

import trt_pose.coco
import trt_pose.models
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects


def parse_arguments():
    #trt_pose
    parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
    parser.add_argument('--mode', default='video', help='Choose how you wanna test. Two modes available : video and webcam')
    parser.add_argument('--json', default='human_pose.json', help='json file for pose estimation')
    parser.add_argument('--size', default=256, type=int, help='image size (WxH) as model input (You can also check this in the filename of pretrained model file.)')
    parser.add_argument('--trt_model', default='pretrained/densenet121_baseline_att_256x256_B_epoch_160.pth', help='trt pose model')

    parser.add_argument('--video_input', default='street_walk_4.mp4', help='testing video input')
    return parser

class Camera():
    def start(self, args):
        self.args = args
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        self.output_path = os.path.join('outputs','output_of'+args.video_input)
        #inference type 
        if self.args.mode == 'video':
            self.cap = cv2.VideoCapture(args.video_input)
        elif self.args.mode == 'webcam':
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 840)
        else:
            print('Wrong mode')
        self.video_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.out_video = cv2.VideoWriter(self.output_path, self.fourcc, self.cap.get(cv2.CAP_PROP_FPS), self.video_size)
        return self.cap

    def write(self, dst):
        cv2.imshow('Output Window', dst)
        self.out_video.write(dst)

    def destory(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.out_video.release()
        logging.info("Vidoe write to outputs/"+self.output_path)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    parser = parse_arguments()
    args = parser.parse_args()
    WIDTH, HEIGHT = args.size, args.size

    #trtpose
    pose = TrtPose(args)   
    #cv2 setup
    cam = Camera()
    cap = cam.start(args)

    while(cap.isOpened()):
        ret, dst = cap.read()
        if ret:
            img = cv2.resize(dst, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            execute(pose, img, dst)
            cam.write(dst)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cam.destory()

if __name__ == '__main__':
    main()