import os
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from osgeo import gdal
import torchvision

from solarnet.datasets.utils import normalize

from solarnet.models import Segmenter

import torch


def _load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def _check_resolution(image_path: str) -> float:
    dataset = gdal.Open(image_path)
    geotransform = dataset.GetGeoTransform()
    pixel_width = abs(geotransform[1])
    # pixel_height = abs(geotransform[5])
    return round(pixel_width, 3)


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
    model.eval() # batch norm will not work properly otherwise
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
                prediction = model(input_tensor)['out']
                prediction = torch.sigmoid(prediction)
                segmentation[offset_y:offset_y+model_input_size, offset_x:offset_x+model_input_size] += prediction.squeeze(0).squeeze(0).numpy()
                segmentation_weight[offset_y:offset_y+model_input_size, offset_x:offset_x+model_input_size] += 1


        segmentation /= segmentation_weight


    # Crop the predictions to the original image size
    segmentation = segmentation[:original_shape[0], :original_shape[1]]
    segmentation = segmentation * 255
    segmentation = cv2.resize(segmentation, (0, 0), fx=target_resolution / original_resolution, fy=target_resolution / original_resolution)
    segmentation = segmentation.astype(np.uint8)
    mask_colored = cv2.applyColorMap(segmentation, cv2.COLORMAP_JET)
    cv2.imwrite(image_path.replace('.tiff', '_pred_deeplabv3_224_bc.png'), mask_colored)


def get_deeplab():
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True)
    model.classifier[-1] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load('data/models/segmenter.model', map_location=torch.device('cpu')))
    return model


if __name__ == '__main__':
    gic_data_dir = Path('../Solar Panels Dataset - GeoTIFF/Solar Panels Dataset - GeoTIFF/')

    for filename in os.listdir(gic_data_dir):
        if filename.endswith('.tiff'):
            original_resolution = _check_resolution(str(gic_data_dir / filename))
            segment_image(str(gic_data_dir / filename), get_deeplab(), model_input_size=224, original_resolution=original_resolution, target_resolution=0.3)