# import zipfile

# nnUNet_results = "/Users/duongphamminhdung/Documents/MacAirM2/GitHub/signal-processing-ecg/results"
# model_zip_path = "/Users/duongphamminhdung/Downloads/signal_ecg_1.zip"

# def install_model_from_zip_file(zip_file: str):
#     with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#         zip_ref.extractall(nnUNet_results)

# install_model_from_zip_file(model_zip_path)
import glob
import cv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs
from tqdm import tqdm
import numpy as np
import torch
import os

def cvt_mask(img):
    img[img == 1] = 255
    
    return img

def convert_mask_from_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.png"))
    for file_path in tqdm(files):
        i = cvt_mask(cv2.imread(file_path))
        cv2.imwrite(file_path, i)
    
def predict_1_file(path):
    pass

def predict(input_folder, output_folder, model_folder, fold='all', step_size=0.5, disable_tta=False, verbose=False,
                                    save_probabilities=False, continue_prediction=False, chk='checkpoint_final.pth', npp=3,
                                    nps=3, prev_stage_predictions=None, device='cuda', disable_progress_bar=False):

    if not isdir(output_folder):
        maybe_mkdir_p(output_folder)

    assert device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=step_size,
                                use_gaussian=True,
                                use_mirroring=not disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=verbose,
                                allow_tqdm=not disable_progress_bar,
                                verbose_preprocessing=verbose)
    predictor.initialize_from_trained_model_folder(model_folder, fold, chk)
    predictor.predict_from_files(input_folder, output_folder, save_probabilities=save_probabilities,
                                 overwrite=not continue_prediction,
                                 num_processes_preprocessing=npp,
                                 num_processes_segmentation_export=nps,
                                 folder_with_segs_from_prev_stage=prev_stage_predictions,
                                 num_parts=1, part_id=0)
    convert_mask_from_folder(output_folder)

# if __name__ == "__main__":
#     predict()
# !nnUNet_results="/content/drive/MyDrive/AI_TRAINING/signal-ecg/results" \
# nnUNetv2_predict \
# -i input_folder \
# -o output_folder \
# -d dataset_id \
# -c 2d \
# -f all