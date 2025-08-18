# MitoAnnotation

## Resize image or mask to target shape
```bash
python resize_img.py --input <input_path> --output <output_path> --size <width> <height> <depth> --mode <mode>
```

Example:
```bash
python resize_img.py --input /path/to/input.tiff --output /path/to/output.tiff --size 443 606 870 --mode mask
```

## Generate boundary from label for nnUNet training
```bash
python generate_contour.py -i <input_dir> -o <output_dir> -w <width>
```

Example:
```bash
python generate_contour.py -i /path/to/labelsTr -o /path/to/labelsTr_Contour -w 5
```

## Train nnUNet
```bash
sbatch scripts/train_nnunet.sl <nnunet_task_id>
```

## Use nnUNet predict with probabilities
```bash
nnUNetv2_predict -i <input_dir> -o <output_dir> -d <dataset_id> -c <configuration> -f <fold> --save_probabilities
```

Example:
```bash
nnUNetv2_predict -i /path/to/imagesTs -o /path/to/imagesTs_pred -d 22 -c 3d_fullres -f all --save_probabilities
```

## Run BC watershed
```bash
python bc_watershed.py -i <input_dir> -o <output_dir> --save-tiff
```

Example:
```bash
python bc_watershed.py -i /path/to/imagesTs_pred -o /path/to/imagesTs_pred_waterz --save-tiff
```

## Find the false merge
```bash
python code/get_labels_size_sort.py -i <mask_file>
```

Example:
```bash
python code/get_labels_size_sort.py -i /path/to/mask.tiff
```

## Use micro SAM to predict boundary to reduce the false merge
```bash
python predict_boundary.py --img_path <image_path> --mask_path <mask_path> --merge_axes <axes>
```

Example:
```bash
python predict_boundary.py --img_path /path/to/image.tif --mask_path /path/to/mask.tif --merge_axes z
```

## Evaluate results (if have GT)
```bash
python evaluate_res.py --gt_file <gt_file> --pred_file <pred_file>
```

Example:
```bash
python evaluate_res.py --gt_file /path/to/gt_tiff --pred_file /path/to/pred_tiff
```

## Note
Replace all `<parameter>` placeholders with your actual file paths and parameters. This helps maintain security and avoid exposing sensitive directory structures.