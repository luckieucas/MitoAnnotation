from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
import numpy as np
import argparse
import tifffile as tiff
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import sys
sys.path.append("/mmfs1/data/liupen/project/MitoSeg/MitoVerse")

from post_processing import remove_single_slice_labels,filter_small_connected_components, filter_3d_segmentation_per_slice

def init_predictor():
    
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset002_MitoSeg/nnUNetTrainer__nnUNetPlans__2d'),
        use_folds=('all',),
        checkpoint_name='checkpoint_final.pth',
    )
    
    return predictor

def my_iterator(predictor, list_of_input_arrs, list_of_input_props):
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=predictor.verbose)
    for a, p in zip(list_of_input_arrs, list_of_input_props):
        data, seg = preprocessor.run_case_npy(a,
                                                None,
                                                p,
                                                predictor.plans_manager,
                                                predictor.configuration_manager,
                                                predictor.dataset_json)
        yield {'data': torch.from_numpy(data).contiguous().pin_memory(), 'data_properties': p, 'ofile': None}

def pred_mito(predictor, img, save_name):
    """
    predict mitochondria from 2d nnUNet model on a tiff image
    """
    slice_list = [img[np.newaxis,i:i+1, :, :] for i in range(img.shape[0])]
    ret = predictor.predict_from_data_iterator(my_iterator(predictor,slice_list,[{'spacing': [1.0, 1.0, 1.0]}]*len(slice_list)),
                                               save_probabilities=False, num_processes_segmentation_export=3)
    print(f"ret len: {len(ret)}, ret[0] shape: {ret[0].shape}")
    # convert ret to a 3d image
    pred = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    for i in range(img.shape[0]):
        pred[i] = np.squeeze(ret[i][0])
    
    # convert prob to a 3d image
    # prob = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
    # for i in range(img.shape[0]):
    #     prob[i] = np.squeeze(ret[i][1][1])

    # do post processing
    tiff.imwrite(save_name, pred)
    pred = filter_3d_segmentation_per_slice(pred)
    tiff.imwrite(save_name, pred)
    pred = filter_small_connected_components(pred)
    pred = remove_single_slice_labels(pred, 10)
    # save the prediction
    if pred.max() > 256:
        pred = pred.astype(np.uint16)
    if pred.max() < 256:
        pred = pred.astype(np.uint8)
    tiff.imwrite(save_name, pred)
    #tiff.imwrite(save_name.replace('.tif', '_prob.tif'), prob)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    # load the tiff image
    img = tiff.imread(args.input)
    # initiate the nnUNetPredictor
    predictor = init_predictor()

    # predict mitochondria
    pred_mito(predictor, img, args.output)
if __name__ == '__main__':
    main()



