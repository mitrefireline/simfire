import glob

import matplotlib.pyplot as plt
from osgeo import gdal, osr


def GetExtent(ds):
    """Return list of corner coordinates from a gdal Dataset"""
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel

    return (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)


def ReprojectCoords(coords, src_srs, tgt_srs):
    """Reproject a list of x,y coordinates."""
    trans_coords = []
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    for x, y in coords:
        x, y, z = transform.TransformPoint(x, y)
        trans_coords.append([x, y])
    return trans_coords


def getCardinalCoordinates(tif):
    ext = GetExtent(tif)
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(tif.GetProjection())
    tgt_srs = src_srs.CloneGeogCS()

    geo_ext = ReprojectCoords(ext, src_srs, tgt_srs)
    print(geo_ext)


def createSubTif(tif):
    gdal.Translate("BP_CA_WGS84_ALIGNED.tif", tif, projWin=(-125, 41, -115, 33))


def mass_preprocess():
    for item in glob.glob("gdalwarps/*"):
        tif = gdal.Open(item)
        name = item.replace("gdalwarps/", "")
        name = name.replace(".tif", "")
        array = tif.GetRasterBand(1).ReadAsArray()
        plt.imshow(array)
        plt.savefig(f"gdalwarps_pngs/{name}.png")


def preprocess():
    tif = gdal.Open("gdalwarps/n39w122.tif")
    array = tif.GetRasterBand(1).ReadAsArray()
    plt.imshow(array)
    plt.savefig("n39w122.png")


if __name__ == "__main__":
    # mass_preprocess()
    preprocess()
