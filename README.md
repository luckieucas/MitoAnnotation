# MitoAnnotation
 ## Resize image or mask to target shape
 ```
 python resize_img.py --input /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset022_MitoSegBetaSeg/labelsTr/high_c2_im_300-522.tiff --output /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset022_MitoSegBetaSeg/labelsTr/high_c2_im_300-522_resized.tiff --size 443 606 870 --mode mask
 ```
 ## Generate boundary from label for nnUNet training
 ```
 python generate_contour.py -i /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset026_MitoSegF536/labelsTr -o /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset026_MitoSegF536/labelsTr_Contour -w 5
 ```
 
 ## Train nnUNet
 ```
 sbatch scripts/train_nnunet.sl nnunet_task_id
 ```

 ## Use nnUNet predict with probabilities
```
nnUNetv2_predict -i /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset022_MitoSegBetaSeg/imagesTs -o /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset022_MitoSegBetaSeg/imagesTs_pred -d 22 -c 3d_fullres -f all  --save_probabilities
```
 ## Run BC watershed
```
python bc_watershed.py -i /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset022_MitoSegBetaSeg/imagesTs_pred -o /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset022_MitoSegBetaSeg/imagesTs_pred_waterz --save-tiff
```

##  Evaluate results (if have GT)
```
python evaluate_res.py --gt_file /path/to/gt_tiff --pred_file /path/to/pred_tiff
```