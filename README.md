# BrainMRDiff: A Diffusion Model for Anatomically Consistent Brain MRI Synthesis

Under review.
# Datasets
Download [BraTS-AG](https://www.synapse.org/Synapse:syn64153130/wiki/631053) (BraTS Glioma Segmentation on Pre- and Post-treatment MRI: Pre only) and [BraTS-Met](https://www.synapse.org/Synapse:syn64153130/wiki/631058) (Segmentation of Pre- and Post-Treatment Brain Metastases: Pre only)

# Train
```
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --channel_name flair
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --channel_name t1
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --channel_name t1ce
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --channel_name t2 
```

## Citation
If you find this repository useful, please consider giving a star :star: and cite the following
```
TBD
```
