import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import FSPNet_model
import dataset

from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from imageio import imwrite
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from einops import rearrange
from fvcore.nn import FlopCountAnalysis, parameter_count_table


if __name__ =='__main__':
    batch_size = 1
    net = FSPNet_model.Model(None, img_size=512).cuda()

    ckpt=['model_152_loss_0.50682.pth']

    Dirs=[
           "/mnt/ssd1/sunhao/data/test/CAMO",
           "/mnt/ssd1/sunhao/data/test/COD10K",
           "/mnt/ssd1/sunhao/data/test/NC4K",
           "/mnt/ssd1/sunhao/data/test/CHAMELEON"
          ]

    result_save_root="result_8N/"

    for m in ckpt:
        print(m)
        ckpt_root="ckpt_save_1/"
        ckpt_file=""
        pretrained_dict = torch.load(ckpt_root+ckpt_file+m)

        net_dict = net.state_dict()
        pretrained_dict={k[7:]: v for k, v in pretrained_dict.items() if k[7:] in net_dict }
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
        net.eval()
        for i in range(len(Dirs)):
            Dir = Dirs[i]
            if not os.path.exists(result_save_root):
                os.mkdir(result_save_root)
            if not os.path.exists(os.path.join(result_save_root, Dir.split("/")[-1])):
                os.mkdir(os.path.join(result_save_root, Dir.split("/")[-1]))
            Dataset = dataset.TestDataset(Dir, 512)
            Dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=batch_size*2)
            count=0
            for data in Dataloader:
                count+=1
                img, label = data['img'].cuda(), data['label'].cuda()
                name = data['name'][0].split("/")[-1]
                with torch.no_grad():
                    out = net(img)
                    out = out[2]
                filename = data['name'][0][37:-4] + '.jpg'
                B,C,H,W = label.size()
                o = F.interpolate(out, (H,W), mode='bilinear', align_corners=True).detach().cpu().numpy()[0,0]
                o =(o-o.min())/(o.max()-o.min()+1e-8)
                o = (o*255).astype(np.uint8)
                imwrite(result_save_root+Dir.split("/")[-1]+"/"+name, o)
    
    print("Test finished!")


