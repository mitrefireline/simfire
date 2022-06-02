from pathlib import Path
from osgeo import gdal
import csv
import os
import glob


def produceSlices(tif):
    """
    Slices up the input tif into equal resolution .tif files

    Arguments:
        tif.  An input tif file

    Returns:
        None.
    """
    gt = tif.GetGeoTransform()
    xmin = gt[0]
    ymax = gt[3]
    res = gt[1]
    xlen = res * tif.RasterXSize
    ylen = res * tif.RasterYSize

    x_slices = int(xlen)
    y_slices = int(ylen)
    # xsize = xlen/x_slices
    # ysize = ylen/y_slices
    xsize = 1
    ysize = 1

    xsteps = [xmin + xsize * i for i in range(x_slices + 1)]
    ysteps = [ymax - ysize * i for i in range(y_slices + 1)]

    for i in range(x_slices):
        for j in range(y_slices):
            xmin = xsteps[i]
            xmax = xsteps[i + 1]
            ymax = ysteps[j]
            ymin = ysteps[j + 1]

            west = str(int(abs(xmin)))
            north = str(int(abs(ymax)))

            gdal.Warp(
                f"/nfs/lslab2/fireline/data/risk/30m/n{north}w{west}.tif",
                tif,
                outputBounds=(xmin, ymin, xmax, ymax),
            )


def preprocess():
    """
    Opens a BP_CA_ALIGNED.tif file which is a file that is projected to WGS84 and cropped to (-125,43),(-113, 32)
    within the /nfs/lslab2/fireline/data/risk/CA/ folder

    If BP_CA_ALIGNED.tif is missing you must run the following commands in order below within the terminal:

    1) gdalwarp BP_CA.tif BP_CA_WGS84.tif -overwrite -t_srs "+proj=longlat +ellps=WGS84"
    2) gdalwarp BP_CA_WGS84.tif BP_CA_ALIGNED.tif -overwrite -t_srs "+proj=longlat +ellps=WGS84" -te -125.0 32.0 -113.0 43.0

    Arguments:
        None

    Returns:
        None.  Instead, produces 30m x 30m resolution slices of BP_CA_ALIGNED.tif into the
        /nfs/lslab2/fireline/data/risk/CA/30m/ folder
    """
    tif = gdal.Open("/nfs/lslab2/fireline/data/risk/CA/BP_CA_ALIGNED.tif")
    produceSlices(tif)


if __name__ == "__main__":
    preprocess()
