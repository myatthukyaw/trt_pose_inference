import json
import logging
import argparse
import torch
import torch2trt
from torch2trt import TRTModule

import trt_pose.models

def parse_args():
    parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
    parser.add_argument('--json', default='human_pose.json', help='json file for pose estimation')
    parser.add_argument('--size', default=256,type=int, help='image size (WxH) as model input (You can also check this in the filename of pretrained model file.)')
    parser.add_argument('--model', default='pretrained/densenet121_baseline_att_256x256_B_epoch_160.pth', help='output trt pose model')
    return parser

def main():
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    parser = parse_args()
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        human_pose = json.load(f)

    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    width, height = args.size, args.size

    if args.model.split('/')[1].split('_')[0][0:8] == 'densenet':
        model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    elif args.model.split('/')[1].split('_')[0][0:6] == 'resnet':
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    else:
        print('Only two pretrained models: densenet121 and resnet18 are supported')

    model.load_state_dict(torch.load(args.model))
    data = torch.zeros((1, 3, height, width)).cuda()
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)


    OPTIMIZED_MODEL = args.model.split('.')[0]+'_trt.pth'

    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
    logging.info('Trt model saved to '+ OPTIMIZED_MODEL)

if __name__ == '__main__':
    main()
