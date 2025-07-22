import os
import random
import imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.cuda.amp import autocast

from generative.inferers import ControlNetDiffusionInferer, DiffusionInferer
from generative.networks.schedulers import DDPMScheduler

from monai import transforms
from monai.data import (
    Dataset,
    CacheDataset,
    DataLoader,
    pad_list_data_collate
    )

from models.controlnet import ControlNet

random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_images_ours_ag(channel_name='flair'):   
    diffusion_model = torch.load('/path/to/weights/{}/diffusion.pth'.format(channel_name)) #, map_location=torch.device('cpu')))
    controlnet_model = torch.load('/path/to/weights/brainmrdiff_{}/brainmrdiff.pth'.format(channel_name)) #, map_location=torch.device('cpu')))

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    controlnet_inferer = ControlNetDiffusionInferer(scheduler)

    from PIL import Image
    import torchvision.transforms as transforms

    tumor = Image.open('/path/to/temp/tumor.png')
    brain = Image.open('/path/to/temp/brain.png')

    wmt = Image.open('/path/to/temp/structure_wmt.png')
    cgm = Image.open('/path/to/temp/structure_cgm.png')
    lv = Image.open('/path/to/temp/structure_lv.png')

    transform = transforms.ToTensor()

    tumor = transform(tumor)
    brain = transform(brain)
    wmt = transform(wmt)
    cgm = transform(cgm)
    lv = transform(lv)

    masks = torch.cat((brain, wmt, cgm, lv), dim=0).to(device)

    with torch.no_grad():
        with autocast(enabled=True):
            noise = torch.randn((1, 1, 128, 128)).to(device)
            sample = controlnet_inferer.sample(
                input_noise = noise,
                diffusion_model = diffusion_model,
                controlnet = controlnet_model,
                cn_cond = (tumor.unsqueeze(0).to(device), masks.unsqueeze(0)),
                scheduler = scheduler,
            )

    mask = sample[0, 0].cpu().numpy()

    if mask.dtype == np.float32 or mask.dtype == np.float64:
        mask = (255 * (mask - mask.min()) / (mask.max() - mask.min())).astype(np.uint8)
    imageio.imwrite('/path/to/{}.png'.format(channel_name), mask)

if __name__ == "__main__":
    flag = 0

    # generate_images_ours_ag(channel_name='flair')
    # generate_images_ours_ag(channel_name='t1')
    # generate_images_ours_ag(channel_name='t1ce')
    # generate_images_ours_ag(channel_name='t2')