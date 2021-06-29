import os
import cv2
import json
import torch
import PIL.Image
import argparse
import numpy as np
from PIL import Image

import torch2trt
from torch2trt import TRTModule
import torchvision.transforms as transforms

import trt_pose.coco
import trt_pose.models
from trt_pose import models, coco
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects


class TrtPose:
    
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.args = args
        
        self.width, self.height = self.args.size, self.args.size
        
        # load humanpose json data
        self.human_pose = self.load_json(args.json)
        
        # load trt model
        self.trt_model  = self._load_trt_model(args.trt_model)
        self.topology = coco.coco_category_to_topology(self.human_pose)
        self.parse_objects = ParseObjects(self.topology)    #, cmap_threshold=0.08, link_threshold=0.08
        self.draw_objects = DrawObjects(self.topology)

        # transformer
        # self.transforms = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ])
        
        
    @staticmethod
    def load_json(json_file):
        with open(json_file, 'r') as f:
            human_pose = json.load(f)
        return human_pose

    def _load_trt_model(self, MODEL):
        """
        load converted tensorRT model  
        """
        num_parts = len(self.human_pose['keypoints'])
        num_links = len(self.human_pose['skeleton'])

        if MODEL.split('_')[-1] == 'trt.pth':
            model = TRTModule()
            model.load_state_dict(torch.load(MODEL))
            model.eval()
        else:
            if MODEL.split('/')[1].split('_')[0][0:8] == 'densenet':
                model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
            elif MODEL.split('/')[1].split('_')[0][0:6] == 'resnet':
                model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
            model.load_state_dict(torch.load(MODEL))

        return model


    def predict(self, image: np.ndarray):
        """
        predict pose estimationkeypoints
        *Note - image need to be RGB numpy array format
        """
        data = self.preprocess(image)
        cmap, paf = self.trt_model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf) # cmap threhold=0.15, link_threshold=0.15
        return counts, objects, peaks
    
    def preprocess(self, image):
        """
        resize image and transform to tensor image
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    def draw_objs(self,dst, counts, objects, peaks):
        self.draw_objects(dst, counts, objects, peaks)


def get_keypoint(humans, hnum, peaks):
    kpoint = []
    pid = hnum
    key_points = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            # there is a joint : 0
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:heigh
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            key_points.append(peak[2])
            key_points.append(peak[1])
            #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:    
            # there is no joint : -1
            peak = (j, 0, 0)
            kpoint.append(peak)
            key_points.append(peak[2])
            key_points.append(peak[1])
            #print('index:%d : None'%(j) )
    return pid, key_points

def execute(pose, img, dst):
    """
    {people : [ {'person_id':'',pose_keypoints_2d:''} ]}
    """
    # people = {'people':[]}
    # person = {}
    # key_points = []
    
    counts, objects, peaks = pose.predict(img)

    # for i in range(counts[0]):
    #     pid, key_points = get_keypoint(objects, i, peaks)
    #     person['person_id'] = pid
    #     person['pose_keypoints_2d'] = key_points
    #     people['people'].append(person)
    #     person = {}
    pose.draw_objs(dst, counts, objects, peaks)
    #return img, people
    

       
        
    
        
                        