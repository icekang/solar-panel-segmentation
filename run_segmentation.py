import os
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
# from patchify import patchify, unpatchify

from solarnet.datasets.utils import normalize

from solarnet.models import Segmenter

import torch


def get_model() -> Segmenter:
    model = Segmenter()
    model.load_state_dict(torch.load('data/models/segmenter.model', map_location=torch.device('cpu')))
    model.eval()
    return model


def _load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


# def ____segment_image(image_path: str, model: Segmenter, model_input_size=224, original_resolution=0.075, target_resolution=0.3) -> np.ndarray:
#     image = _load_image(image_path)
#     # Resize the image to the target resolution
#     image = cv2.resize(image, (0, 0), fx=original_resolution / target_resolution, fy=original_resolution / target_resolution)

#     original_shape = image.shape

#     # Pad the iamge to be divisible by model_input_size
#     top = 0
#     bottom = model_input_size - (image.shape[0] % model_input_size)
#     left = 0
#     right = model_input_size - (image.shape[1] % model_input_size)
#     image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)

#     # Crop image to squares
#     with torch.no_grad():
#         predictions = []
#         for offset_y in range(0, image.shape[0], model_input_size):
#             prediction_row = []
#             for offset_x in range(0, image.shape[1], model_input_size):
#                 image_crop = image[offset_y:offset_y+model_input_size, offset_x:offset_x+model_input_size, :]
#                 assert image_crop.shape == (model_input_size, model_input_size, 3), 'Cropped image is not square'
#                 image_crop = np.moveaxis(image_crop, -1, 0)
#                 image_crop = normalize(image_crop)

#                 input_tensor = torch.as_tensor(image_crop).unsqueeze(0).float()
#                 prediction = model(input_tensor)
#                 prediction_row.append(prediction.squeeze(0).squeeze(0))

#             row = torch.cat(prediction_row, dim=1)
#             predictions.append(row)
#         predictions = torch.cat(predictions, dim=0)

#     # Crop the predictions to the original image size
#     predictions = predictions[:original_shape[0], :original_shape[1]]
#     predictions = predictions.numpy() * 255
#     predictions = cv2.resize(predictions, (0, 0), fx=target_resolution / original_resolution, fy=target_resolution / original_resolution)
#     predictions = predictions.astype(np.uint8)
#     cv2.imwrite(image_path.replace('.tiff', '_pred.png'), predictions)
#     mask_colored = cv2.applyColorMap(predictions, cv2.COLORMAP_JET)
#     cv2.imwrite(image_path.replace('.tiff', '_pred_color.png'), mask_colored)


# def segment_image(image_path: str, model: Segmenter, model_input_size=224, original_resolution=0.075, target_resolution=0.3):
#     image = _load_image(image_path)
#     # Resize the image to the target resolution
#     image = cv2.resize(image, (0, 0), fx=original_resolution / target_resolution, fy=original_resolution / target_resolution)

#     original_shape = image.shape

#     # Pad the iamge to be divisible by model_input_size
#     top = 0
#     bottom = model_input_size - (image.shape[0] % model_input_size)
#     left = 0
#     right = model_input_size - (image.shape[1] % model_input_size)
#     image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
#     segmentation = np.zeros((image.shape[0], image.shape[1]))
#     segmentation_weight = np.zeros((image.shape[0], image.shape[1]))

#     # Crop image to squares
#     with torch.no_grad():
#         predictions = []
#         for offset_y in range(0, image.shape[0], model_input_size // 2):
#             prediction_row = []
#             for offset_x in range(0, image.shape[1], model_input_size // 2):
#                 image_crop = image[offset_y:offset_y+model_input_size, offset_x:offset_x+model_input_size, :]
#                 if image_crop.shape != (model_input_size, model_input_size, 3):
#                     continue
#                 image_crop = np.moveaxis(image_crop, -1, 0)
#                 image_crop = normalize(image_crop)

#                 input_tensor = torch.as_tensor(image_crop).unsqueeze(0).float()
#                 prediction = model(input_tensor)
#                 prediction_row.append(prediction.squeeze(0).squeeze(0))
#                 segmentation[offset_y:offset_y+model_input_size, offset_x:offset_x+model_input_size] += prediction.squeeze(0).squeeze(0).numpy()
#                 segmentation_weight[offset_y:offset_y+model_input_size, offset_x:offset_x+model_input_size] += 1


#             row = torch.zeros((model_input_size, image.shape[1]))
#             row[:, :model_input_size] = prediction_row[0]
#             for i, tile in enumerate(prediction_row[1:], 1):
#                 row[:, i * model_input_size // 2: i * model_input_size // 2 + model_input_size] += tile
#                 row[:, i * model_input_size // 2: i * model_input_size // 2 + model_input_size // 2] /= 2
#             predictions.append(row)
#         predictions = torch.cat(predictions, dim=0)
#         segmentation /= segmentation_weight


#     # Crop the predictions to the original image size
#     predictions = predictions[:original_shape[0], :original_shape[1]]
#     predictions = predictions.numpy() * 255
#     predictions = cv2.resize(predictions, (0, 0), fx=target_resolution / original_resolution, fy=target_resolution / original_resolution)
#     predictions = predictions.astype(np.uint8)
#     cv2.imwrite(image_path.replace('.tiff', '_pred_overlap.png'), predictions)
#     mask_colored = cv2.applyColorMap(predictions, cv2.COLORMAP_JET)
#     cv2.imwrite(image_path.replace('.tiff', '_pred_color_overlap.png'), mask_colored)
#     cv2.imwrite(image_path.replace('.tiff', '_pred_overlap.png'), segmentation)


def segment_image(image_path: str, model: Segmenter, model_input_size=224, original_resolution=0.075, target_resolution=0.3):
    image = _load_image(image_path)
    # Resize the image to the target resolution
    image = cv2.resize(image, (0, 0), fx=original_resolution / target_resolution, fy=original_resolution / target_resolution)

    original_shape = image.shape

    # Pad the iamge to be divisible by model_input_size
    top = 0
    bottom = model_input_size - (image.shape[0] % model_input_size)
    left = 0
    right = model_input_size - (image.shape[1] % model_input_size)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    segmentation = np.zeros((image.shape[0], image.shape[1]))
    segmentation_weight = np.zeros((image.shape[0], image.shape[1]))

    # Crop image to squares
    with torch.no_grad():
        predictions = []
        for offset_y in range(0, image.shape[0], model_input_size // 4):
            prediction_row = []
            for offset_x in range(0, image.shape[1], model_input_size // 4):
                image_crop = image[offset_y:offset_y+model_input_size, offset_x:offset_x+model_input_size, :]
                if image_crop.shape != (model_input_size, model_input_size, 3):
                    continue
                image_crop = np.moveaxis(image_crop, -1, 0)
                image_crop = normalize(image_crop)

                input_tensor = torch.as_tensor(image_crop).unsqueeze(0).float()
                prediction = model(input_tensor)
                segmentation[offset_y:offset_y+model_input_size, offset_x:offset_x+model_input_size] += prediction.squeeze(0).squeeze(0).numpy()
                segmentation_weight[offset_y:offset_y+model_input_size, offset_x:offset_x+model_input_size] += 1


        segmentation /= segmentation_weight


    # Crop the predictions to the original image size
    segmentation = segmentation[:original_shape[0], :original_shape[1]]
    segmentation = segmentation * 255
    segmentation = cv2.resize(segmentation, (0, 0), fx=target_resolution / original_resolution, fy=target_resolution / original_resolution)
    segmentation = segmentation.astype(np.uint8)
    mask_colored = cv2.applyColorMap(segmentation, cv2.COLORMAP_JET)
    cv2.imwrite(image_path.replace('.tiff', '_pred_color_overlap_wavg.png'), mask_colored)



# def segment_image(image_path: str, model: Segmenter, model_input_size=224, original_resolution=0.075, target_resolution=0.3):
#     image = _load_image(image_path)
#     # Resize the image to the target resolution
#     image = cv2.resize(image, (0, 0), fx=original_resolution / target_resolution, fy=original_resolution / target_resolution)

#     original_shape = image.shape

#     # Pad the iamge to be divisible by model_input_size
#     top = 0
#     bottom = model_input_size - (image.shape[0] % model_input_size)
#     left = 0
#     right = model_input_size - (image.shape[1] % model_input_size)
#     image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
#     patches = patchify(image=image, patch_size=(model_input_size, model_input_size, 3), step=model_input_size // 2)
#     print(image.shape, patches.shape)
#     segmentation_mask = np.zeros((patches.shape[0], patches.shape[1], model_input_size, model_input_size))

#     with torch.no_grad():
#         # Crop image to squares
#         for i in range(patches.shape[0]):
#             for j in range(patches.shape[1]):
#                 image_crop = patches[i, j, 0, :, :, :]
                
#                 assert image_crop.shape == (model_input_size, model_input_size, 3), f'Cropped image is not {(model_input_size, model_input_size, 3)}, instead got {image_crop.shape}'
#                 image_crop = np.moveaxis(image_crop, -1, 0)
#                 image_crop = normalize(image_crop)

#                 input_tensor = torch.as_tensor(image_crop).unsqueeze(0).float()
#                 prediction = model(input_tensor)
#                 segmentation_mask[i, j, :, :] = prediction.squeeze(0).squeeze(0)
    
#     print('before segmentation_mask.shape', segmentation_mask.shape)
#     segmentation_mask = unpatchify(segmentation_mask, image.shape[:2])
#     print('after segmentation_mask.shape', segmentation_mask.shape)
#     segmentation_mask = segmentation_mask[:original_shape[0], :original_shape[1]]
#     segmentation_mask = segmentation_mask * 255
#     segmentation_mask = cv2.resize(segmentation_mask, (0, 0), fx=target_resolution / original_resolution, fy=target_resolution / original_resolution)
#     segmentation_mask = segmentation_mask.astype(np.uint8)
#     # cv2.imwrite(image_path.replace('.tiff', '_pred_overlap.png'), predictions)
#     mask_colored = cv2.applyColorMap(segmentation_mask, cv2.COLORMAP_JET)
#     cv2.imwrite(image_path.replace('.tiff', '_pred_color_patchified.png'), mask_colored)

if __name__ == '__main__':
    gic_data_dir = Path('../Solar Panels Dataset - GeoTIFF/Solar Panels Dataset - GeoTIFF/')

    for filename in os.listdir(gic_data_dir):
        break
        if filename.endswith('.tiff'):
            segment_image(str(gic_data_dir / filename), get_model())