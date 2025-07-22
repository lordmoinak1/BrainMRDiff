import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from monai import transforms
from monai.utils import first
from monai.data import (
    CacheDataset,
    DataLoader,
    pad_list_data_collate
    )

from generative.inferers import ControlNetDiffusionInferer, DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

from loss import *
from models.controlnet import ControlNet

random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model(channel_name):
    diffusion_model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(256, 512, 512),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=512,
    )

    diffusion_model.to(device)

    diffusion_model.load_state_dict(torch.load('/path/to/weights/{}/statedict_diffusion_128.pth'.format(channel_name)))

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    inferer = DiffusionInferer(scheduler)

    controlnet = ControlNet(
        spatial_dims=2,
        in_channels=1,
        num_channels=(256, 512, 512),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=(0, 512, 512),
        conditioning_embedding_in_channels=1,#5, #### 2 for combined and 5 for combined_structure ####
        conditioning_embedding_num_channels=(16,),
    )
    controlnet.to(device)

    for p in diffusion_model.parameters():
        p.requires_grad = False

    # optimizer = torch.optim.Adam(params=controlnet.parameters(), lr=2.5e-5)
    optimizer = torch.optim.AdamW(params=controlnet.parameters(), lr=1e-5)

    controlnet_inferer = ControlNetDiffusionInferer(scheduler)

    return diffusion_model, controlnet, optimizer, inferer, controlnet_inferer, scheduler

class ConvertToMultiChannelBasedOnBratsClassesd(transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []

            # # merge label 2 and label 3 to construct TC
            # result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # # merge labels 1, 2 and 3 to construct WT
            # result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # # label 2 is ET
            # result.append(d[key] == 2)

            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))

            # result.append(d[key] == 1)
            # result.append(d[key] == 2)
            # result.append(d[key] == 3)

            d[key] = torch.stack(result, axis=0).float()
            # print(d[key].shape)
        return d

class SelectClassesStructureWMTd(transforms.MapTransform):
    """
    Convert labels to multi channels based on SynthSeg classes:
    label 2 and label 41 is the WMT
    label 3 and label 42 is the CGM
    label 4 and label 43 is the LV

    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(torch.logical_or(d[key] == 2, d[key] == 41))
            d[key] = torch.stack(result, axis=0).float()
        return d
    
class SelectClassesStructureCGMd(transforms.MapTransform):
    """
    Convert labels to multi channels based on SynthSeg classes:
    label 2 and label 41 is the WMT
    label 3 and label 42 is the CGM
    label 4 and label 43 is the LV

    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(torch.logical_or(d[key] == 3, d[key] == 42))
            d[key] = torch.stack(result, axis=0).float()
        return d

class SelectClassesStructureLVd(transforms.MapTransform):
    """
    Convert labels to multi channels based on SynthSeg classes:
    label 2 and label 41 is the WMT
    label 3 and label 42 is the CGM
    label 4 and label 43 is the LV

    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(torch.logical_or(d[key] == 4, d[key] == 43))
            d[key] = torch.stack(result, axis=0).float()
        return d
    
class CombineTwoKeysTransform(transforms.MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        
    def __call__(self, data):
        d = dict(data)
        key1, key2 = self.keys
        # Custom combination logic (e.g., adding image and label arrays)
        combined_data = torch.stack([d[key1], d[key2]], axis=0).float() #d[key1] + d[key2]  # Example operation
        combined_data = torch.squeeze(combined_data, axis=1)
        d["combined"] = combined_data  # Add combined result to data dict
        return d
    
class CombineFourKeysTransform(transforms.MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        
    def __call__(self, data):
        d = dict(data)
        key1, key2, key3, key4 = self.keys
        # Custom combination logic (e.g., adding image and label arrays)
        combined_data = torch.stack([d[key1], d[key2], d[key3], d[key4]], axis=0).float() #d[key1] + d[key2]  # Example operation
        combined_data = torch.squeeze(combined_data, axis=1)
        d["combined_structure"] = combined_data  # Add combined result to data dict
        return d
    
def dataset(channel_name):
    print(channel_name+"_struc")
    all_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=[channel_name, "wmt", "cgm", "lv", "seg"]),
            transforms.EnsureChannelFirstd(keys=[channel_name]),
            transforms.EnsureTyped(keys=[channel_name, "wmt", "cgm", "lv", "seg"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="seg"),
            SelectClassesStructureWMTd(keys="wmt"),
            SelectClassesStructureCGMd(keys="cgm"),
            SelectClassesStructureLVd(keys="lv"),
            transforms.Orientationd(keys=[channel_name, "wmt", "cgm", "lv", "seg"], axcodes="RAS"),

            transforms.Spacingd(keys=[channel_name, "wmt", "cgm", "lv", "seg"], pixdim=(1.5, 1.5, 1.0), mode="bilinear"),
            transforms.CenterSpatialCropd(keys=[channel_name, "wmt", "cgm", "lv", "seg"], roi_size=(128, 128, 64)),
            transforms.ScaleIntensityRangePercentilesd(keys=channel_name, lower=0, upper=99.5, b_min=0, b_max=1),
            transforms.RandSpatialCropd(keys=[channel_name, "wmt", "cgm", "lv", "seg"], roi_size=(128, 128, 1), random_size=False),
            
            transforms.Lambdad(keys=[channel_name, "wmt", "cgm", "lv", "seg"], func=lambda x: x.squeeze(-1)),
            transforms.CopyItemsd(keys=[channel_name], times=1, names=["mask"]),
            transforms.Lambdad(keys=["mask"], func=lambda x: torch.where(x > 0.1, 1, 0)),
            transforms.FillHolesd(keys=["mask"]),
            transforms.CastToTyped(keys=["mask"], dtype=np.float32),
            CombineTwoKeysTransform(keys=["mask", "seg"]),
            transforms.CastToTyped(keys=["combined"], dtype=np.float32),

            transforms.CastToTyped(keys=["wmt", "cgm", "lv"], dtype=np.float32),

            CombineFourKeysTransform(keys=["mask", "wmt", "cgm", "lv"]),
            transforms.CastToTyped(keys=["combined_structure"], dtype=np.float32),
        ]
    )

    def generate_splits(data_path):
        subjects = []
        for i in os.listdir(data_path):
            subject = {
                't1': os.path.join(data_path+i, i+'_t1.nii.gz'),
                't1ce': os.path.join(data_path+i, i+'_t1ce.nii.gz'),
                't2': os.path.join(data_path+i, i+'_t2.nii.gz'),
                'flair': os.path.join(data_path+i, i+'_flair.nii.gz'),
                'seg': os.path.join(data_path+i, i+'_seg.nii.gz'),
                'wmt': os.path.join('/path/to/synthseg/', i+'_flair.nii.gz'),
                'cgm': os.path.join('/path/to/synthseg/', i+'_flair.nii.gz'),
                'lv': os.path.join('/path/to/synthseg/', i+'_flair.nii.gz'),
                }
            subjects.append(subject)
        return subjects
    
    train_subjects = generate_splits('/path/to/train/')
    val_subjects = generate_splits('/path/to/val/')

    dataset = CacheDataset(data=train_subjects, transform=all_transforms, cache_num=24, cache_rate=1, num_workers=2)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, collate_fn=pad_list_data_collate, drop_last=True)

    dataset = CacheDataset(data=val_subjects, transform=all_transforms, cache_num=24, cache_rate=1, num_workers=2)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, collate_fn=pad_list_data_collate, drop_last=True)

    data_example = dataset[2]
    print(data_example["seg"].shape)

    return train_loader, val_loader

def train_controlnet(diffusion_model, controlnet, optimizer, inferer, controlnet_inferer, scheduler, train_loader, val_loader, channel_name):
    n_epochs = 200
    val_interval = 50
    epoch_loss_list = []
    val_epoch_loss_list = []

    loss_sat = SATopologicalLoss()

    scaler = GradScaler()
    for epoch in range(n_epochs):
        controlnet.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch[channel_name].to(device)

            brain = batch["mask"].to(device)
            tumor = batch["seg"].to(device)
            wmt = batch["wmt"].to(device)
            cgm = batch["cgm"].to(device)
            lv = batch["lv"].to(device)

            combined_structure = batch["combined_structure"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                # Generate random noise
                noise = torch.randn_like(images).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                noise_pred = controlnet_inferer(
                    inputs=images,
                    diffusion_model=diffusion_model,
                    controlnet=controlnet,
                    noise=noise,
                    timesteps=timesteps,
                    cn_cond=(tumor, combined_structure),#(brain, tumor, wmt, cgm, lv),
                )

                if epoch < 75:
                    loss = F.mse_loss(noise_pred.float(), noise.float())
                else:
                    loss_mse = F.mse_loss(noise_pred.float(), noise.float())
                    loss_topo = loss_sat(noise_pred.float(), noise.float(), tumor.float(), brain.float(), wmt.float(), cgm.float(), lv.float())
                    loss = loss_mse + 0.001*loss_topo

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_loss_list.append(epoch_loss / (step + 1))

        if (epoch + 1) % val_interval == 0:
            controlnet.eval()
            val_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                images = batch[channel_name].to(device)
                
                brain = batch["mask"].to(device)
                tumor = batch["seg"].to(device)
                wmt = batch["wmt"].to(device)
                cgm = batch["cgm"].to(device)
                lv = batch["lv"].to(device)

                combined_structure = batch["combined_structure"].to(device)

                with torch.no_grad():
                    with autocast(enabled=True):
                        noise = torch.randn_like(images).to(device)
                        timesteps = torch.randint(
                            0, controlnet_inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                        ).long()

                        noise_pred = controlnet_inferer(
                            inputs=images,
                            diffusion_model=diffusion_model,
                            controlnet=controlnet,
                            noise=noise,
                            timesteps=timesteps,
                            cn_cond=(tumor, combined_structure),#(brain, tumor, wmt, cgm, lv),
                        )
                        
                        if epoch < 75:
                            val_loss = F.mse_loss(noise_pred.float(), noise.float())
                        else:
                            val_loss_mse = F.mse_loss(noise_pred.float(), noise.float())
                            val_loss_topo = loss_sat(noise_pred.float(), noise.float(), tumor.float(), brain.float(), wmt.float(), cgm.float(), lv.float())
                            val_loss = val_loss_mse + 0.001*val_loss_topo

                val_epoch_loss += val_loss.item()
                progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

            torch.save(controlnet, '/path/to/weights/{}/ours_epoch_{}.pth'.format(channel_name, epoch))
            torch.save(controlnet.state_dict(), '/path/to/weights/{}/statedict_ours_epoch_{}.pth'.format(channel_name, epoch))

if __name__ == "__main__":
    flag = 0

    import argparse

    parser = argparse.ArgumentParser(description='select mode')
    parser.add_argument('--channel_name', type=str, required=True)

    args = parser.parse_args()
    print('Channel Name:', args.channel_name)

    train_loader, val_loader = dataset(args.channel_name)
    diffusion_model, controlnet, optimizer, inferer, controlnet_inferer, scheduler = model(args.channel_name)
    train_controlnet(diffusion_model, controlnet, optimizer, inferer, controlnet_inferer, scheduler, train_loader, val_loader, args.channel_name)

    # CUDA_VISIBLE_DEVICES=0 python3 train.py --channel_name flair
    # CUDA_VISIBLE_DEVICES=1 python3 train.py --channel_name t1
    # CUDA_VISIBLE_DEVICES=2 python3 train.py --channel_name t1ce
    # CUDA_VISIBLE_DEVICES=3 python3 train.py --channel_name t2 