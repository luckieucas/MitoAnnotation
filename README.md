# MitoAnnotation
## Use nnUNet predict with probabilities
```
nnUNetv2_predict -i /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset022_MitoSegBetaSeg/imagesTs -o /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset022_MitoSegBetaSeg/imagesTs_pred -d 22 -c 3d_fullres -f all  --save_probabilities
```
## Run BC watershed
```
python bc_watershed.py -i /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset022_MitoSegBetaSeg/imagesTs_pred -o /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset022_MitoSegBetaSeg/imagesTs_pred_waterz --save-tiff
```