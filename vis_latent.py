import os
import cv2
import cmapy
import time
import argparse
import numpy as np
import torch.autograd
from skimage import io
from skimage.exposure import rescale_intensity
from torchvision.transforms import functional as transF
from collections import OrderedDict


################## Model ##################
from models.SAM_CD import SAM_CD as Net
#from models.ResNet_CD import ResNet_CD as Net
NET_NAME = 'SAM_CD_latent'

from datasets import Levir_CD as Data
DATA_NAME = 'Levir_CD'
#from datasets import WHU_CD as Data
#DATA_NAME = 'WHU_CD'
################## Model ##################

class PredOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        
    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--crop_size', required=False, default=(1024, 1024), help='cropping size')
        parser.add_argument('--T', required=False, default=3.0, help='Test time augmentation')
        parser.add_argument('--test_dir', required=False, default=os.path.join(Data.root, 'test'), help='directory to test images')
        parser.add_argument('--pred_dir', required=False, default=os.path.join(working_path, 'eval', DATA_NAME, NET_NAME), help='directory to output masks')
        parser.add_argument('--chkpt_path', required=False, default='xxx/SAM_CD/checkpoints/Levir_CD/xxx.pth' )
        parser.add_argument('--dev_id', required=False, default=0, help='Device id')
        self.initialized = True
        return parser
        
    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt

COLORMAP = [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0], [0,0,128]]

def Index2Color(pred):
    colormap = np.asarray(COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    net = Net()
    state_dict = torch.load(opt.chkpt_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict
    net.load_state_dict(new_state_dict)  
    net.to(torch.device('cuda', int(opt.dev_id))).eval()
    
    predict(net, opt)
    time_use = time.time() - begin_time
    print('Total time: %.2fs'%time_use)

def predict(net, opt):
    imgA_dir = os.path.join(opt.test_dir, 'A')
    imgB_dir = os.path.join(opt.test_dir, 'B')
    
    if not os.path.exists(opt.pred_dir): os.makedirs(opt.pred_dir)
    pred_mA_dir = os.path.join(opt.pred_dir, 'A')
    pred_mB_dir = os.path.join(opt.pred_dir, 'B')
    if not os.path.exists(pred_mA_dir): os.makedirs(pred_mA_dir)
    if not os.path.exists(pred_mB_dir): os.makedirs(pred_mB_dir)
    
    data_list = os.listdir(imgA_dir)
    valid_list = []
    for it in data_list:
        if (it[-4:]=='.png'): valid_list.append(it)
    
    for it in valid_list:
        imgA_path = os.path.join(imgA_dir, it)
        imgB_path = os.path.join(imgB_dir, it)
        imgA = io.imread(imgA_path)
        imgB = io.imread(imgB_path)
        imgA = Data.normalize_image(imgA)
        imgB = Data.normalize_image(imgB)
                
        with torch.no_grad():
              tensorA = transF.to_tensor(imgA).unsqueeze(0).to(torch.device('cuda', int(opt.dev_id))).float()
              tensorB = transF.to_tensor(imgB).unsqueeze(0).to(torch.device('cuda', int(opt.dev_id))).float()            
              output, outputs_A, outputs_B = net(tensorA, tensorB)
              
              mapA = outputs_A.squeeze().detach().cpu().numpy()
              mapB = outputs_B.squeeze().detach().cpu().numpy()
              
              latent_num = mapA.shape[0]
              for idx in range(latent_num):
                  latentA = rescale_intensity(mapA[idx], out_range=(0,255)).astype(np.uint8)
                  latentB = rescale_intensity(mapB[idx], out_range=(0,255)).astype(np.uint8)
                  latentA_color = cv2.applyColorMap(latentA, cmapy.cmap('jet'))
                  latentB_color = cv2.applyColorMap(latentB, cmapy.cmap('jet'))
                  
                  pred_pathA = os.path.join(pred_mA_dir, it[:-4]+'_'+str(idx)+'.png')
                  pred_pathB = os.path.join(pred_mB_dir, it[:-4]+'_'+str(idx)+'.png')
                  io.imsave(pred_pathA, latentA_color)
                  io.imsave(pred_pathB, latentB_color)


if __name__ == '__main__':
    main()