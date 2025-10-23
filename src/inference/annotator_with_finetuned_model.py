import imageio.v3 as imageio

from micro_sam.sam_annotator import annotator_2d


def run_annotator_with_finetuned_model():
    """Run the 2d anntator with a custom (finetuned) model.

    Here, we use the model that is produced by `finetuned_hela.py` and apply it
    for an image from the validation set.
    """
    # take the last frame, which is part of the val set, so the model was not directly trained on it
    im = imageio.imread("/projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset003_MitoHardJurkat/2d_slices/val/jrc_jurkat-1_recon-1_test1/images/slice_0000.tif")

    # set the checkpoint and the path for caching the embeddings
    checkpoint = "checkpoints/sam_jurkat/best.pt"
    embedding_path = "/projects/weilab/liupeng/nnUNet/DATASET/nnUNet_raw/Dataset003_MitoHardJurkat/imagesTs_microsam_FT/embeddings/embeddings-finetuned.zarr"

    # Adapt this if you finetune a different model type, e.g. vit_h.
    model_type = "vit_b"  # We finetune a vit_b in the example script.

    # Run the 2d annotator with the custom model.
    annotator_2d(im, model_type=model_type, embedding_path=embedding_path, checkpoint_path=checkpoint)


if __name__ == "__main__":
    run_annotator_with_finetuned_model()
