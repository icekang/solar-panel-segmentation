import os
from pathlib import Path
from osgeo import gdal # use gdal conda env


if __name__ == '__main__':
    gic_data_dir = Path('../Solar Panels Dataset - GeoTIFF/Solar Panels Dataset - GeoTIFF/')
    for filename in os.listdir(gic_data_dir):
        if filename.endswith('.tiff'):
            dataset = gdal.Open(str(gic_data_dir / filename))
            geotransform = dataset.GetGeoTransform()
            pixel_width = abs(geotransform[1])
            pixel_height = abs(geotransform[5])
            print(pixel_width, pixel_height)