# MitoAnnotation - Mitochondria Segmentation and Annotation Toolkit

This is a comprehensive toolkit for mitochondria segmentation, annotation, and analysis, including data processing, model training, prediction, post-processing, and evaluation functionalities.

---
## **Install**

## **Model Training**

### **Train mitoNet**

raw:
```bash
python src/training/mitoNet_finetune.py -d 1  --mode train --model_dir ./checkpoints/mitonet_ft_d1_test --model_name MitoNet_FT_d1 --iterations 2000 --test_after_training
```
submit to slurm:
```bash
python submit_slurm.py --command "python src/training/mitoNet_finetune.py -d 6  --mode train --model_dir ./checkpoints/mitonet_ft_d6 --model_name MitoNet_FT_d6 --iterations 2000 --test_after_training --skip_conversion"  -t 12:00:00 -p short -n mitoNet_FT_6 -e mitoem2
```

### **Train microSAM**

raw:
```bash
python src/training/micro_sam_finetune.py --dataset_path /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset006_ME2-Pyra --model_type vit_b_em_organelles  --n_epochs 10  --batch_size 1
```
submit to slurm:
```bash
python submit_slurm.py -t 12:00:00 -e mitoem2 -n Pyra_microsam_FT -p short --command "python src/training/micro_sam_finetune.py --dataset_path /projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset006_ME2-Pyra --model_type vit_b_em_organelles  --n_epochs 10  --batch_size 1"
```
### **Train nnUNetBC**
raw:
```bash
nnUNetv2_train 6 3d_fullres 0
```
submit to slurm:
```bash
python submit_slurm.py -t 12:00:00 -e mitoem2 -n nnUNetBC -p short --command "nnUNetv2_train 6 3d_fullres 0"
```


## **Model Inference**

### **mitoNet Inference**
```bash
python src/inference/MitoNet_baseline.py  -d 7 -c checkpoints/Dataset007_MitoHardKedarf536/Kedarf536.yaml --use_gpu --axes xy
```
submit to slurm:
```bash
python submit_slurm.py -t 12:00:00 -e mitoem2 -n mitoNet_pred_mossy -p short --command "python src/inference/MitoNet_baseline.py  -d 7 -c checkpoints/Dataset007_MitoHardKedarf536/Kedarf536.yaml --use_gpu --axes xy"
```

### **microSAM Inference**
```bash
python src/inference/micro_sam_baseline.py  -d 7
```

```bash
python submit_slurm.py -t 12:00:00 -e mitoem2 -n microsam_FT_pred_mossy -p short --command "python src/inference/micro_sam_baseline.py  -d 7"
```


### **nnUNetBC Inference**
```bash
nnUNetv2_predict -i [input_folder] -o [output_folder] -d 6 -f 0 -tr nnUNetTrainerV2
```
submit to slurm:
```bash
 python submit_slurm.py -t 12:00:00 -e mitoem2 -n nnUNetBC_pred_mossy -p short --command "nnUNetv2_predict -i [input_folder] -o [output_folder] -d 6 -f 0 -tr nnUNetTrainerV2"
```



## **Post-Processing**
