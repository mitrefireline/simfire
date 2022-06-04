import csv
import glob
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal


def read_csv(filename):
    with open(filename) as f:
        file_data = csv.reader(f)
        d = {headers[0]: (headers[5], headers[6], headers[7]) for headers in file_data}
        # remove header because I'm lazy and don't want to read the csv correctly
        del d["Value"]
        d.update((k, [float(vals) for vals in v]) for k, v in d.items())
        return d


def preprocess():
    """
    As described:
        http://pyrologix.com/reports/Contemporary-Wildfire-Hazard-Across-California.pdf

    TIF files in GEoTIFF format at approximately 30m

    Fuelscape will consist of:
        LANDFIRE 2016-2020 Remap 2.0.0 (LF Remap)
            - Surface Fuel Model (FM40, FM13) --> ONLY INTERESTED IN THIS FOR NOW
            - Fuel Vegetation Cover (FVC)
            - Canopy Cover (CC)
            - Canopy Height (CH)
            - Canopy Bulk Density (CBD)
            - Canopy Base Height (CBH)

    Projection mapping:
        Original Projection: Albers Equal Conical Area
        Mapping for Topography: NAVD88

        Solution: Albers -> WSG84

    """
    tile_path = Path("/nfs/lslab2/fireline/data/fuel/30m/")
    tiles = glob.glob(str(tile_path / "*"))
    fuel_model = "FBFM13"
    fuel = "F13"

    fm_csv = Path(f"/nfs/lslab2/fireline/data/fuel/LF16_{fuel}_200.csv")
    fm_dict = read_csv(str(fm_csv))

    for tile in tiles:
        fm_file = glob.glob(str(Path(f"{tile}/*_{fuel_model}_*/*.tif")))
        filename = fm_file[0].split(".")
        out_fm_file = filename[0] + "_projected." + filename[1]
        out_fm_file_whitespace_removed = filename[0] + "_projected_no_whitespace"
        out_rgb_file = filename[0] + "_projected_rgb"

        if os.path.exists(
            str(out_fm_file_whitespace_removed) + ".npy"
        ) and os.path.exists(str(out_rgb_file) + ".npy"):
            pass
        else:
            # this might need to be on line
            os.system(
                f"gdalwarp {str(fm_file[0])} "
                f'{str(out_fm_file)} -t_srs "+proj=longlat +ellps=WGS84"'
            )

            tif_data = gdal.Open(str(out_fm_file))
            tif_band = tif_data.GetRasterBand(1)
            tif_arr = tif_band.ReadAsArray()

            new_arr = tif_arr[~(tif_arr == -32768).all(axis=1)]
            idx = np.argwhere(np.all(new_arr[..., :] == -32768, axis=0))
            projected_array = np.delete(new_arr, idx, axis=1)

            # use nearest neighbor interpolation to pad image to be standard
            # 30m resolution: (3612, 3612)
            resized_projected = cv2.resize(
                projected_array, (3612, 3612), 0, 0, interpolation=cv2.INTER_NEAREST
            )

            np.save(str(out_fm_file_whitespace_removed), resized_projected)

            lat = resized_projected.shape[0]
            long = resized_projected.shape[1]

            rgb_arr = []
            for i in range(lat):
                for j in range(long):
                    rgb_arr.append(tuple(fm_dict[str(resized_projected[i, j])]))

            rgb_arr = np.asarray(rgb_arr)
            rgb_arr = rgb_arr.reshape((lat, long, 3))

            plt.imsave(f"images/{tile[-7:]}.png", rgb_arr)
            plt.show()
            np.save(str(out_rgb_file), rgb_arr)


if __name__ == "__main__":
    preprocess()
