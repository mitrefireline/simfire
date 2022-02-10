import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from array import array
from typing import Tuple, List
from pathlib import Path
import math


# Developing a function to round to a multiple
def round_up_to_multiple(number, multiple):
    return multiple * math.ceil(number / multiple)


def round_down_to_multiple(num, divisor):
    return divisor * math.floor(num / divisor)


class TopographyGen():
    def __init__(self, latitude: Tuple[float], longitude: Tuple[float]) -> None:
        '''

        This class of methods will get initialized with the config.
        It will ingest lat/long cordinates and get corresponding topographic data

        Arguments:
            latitude: Tuple(float)
                A tuple(min, max) of latitude cooordinates

            longitude: Tuple(float)
                A tuple(min, max) of longitude coordinates

        Return:
            topograpghy: np.ndarray
                The corresponding topography that approximates given lat/long
                    coordinates


        '''
        # preset these for southern california
        self.lat = latitude
        self.long = longitude
        self.datapath = Path('/nfs/lslab2/fireline/topographic/')

        self.output_dems = self._get_nearest_tile()
        self.flt_filenames, self.tif_filenames = self._get_dems()

    def _get_dems(self) -> List[List[Path]]:
        '''
        This method will use the outputed tiles and return the correct dem files

        '''
        flt_filenames = []
        tif_filenames = []
        if len(self.output_dems) >= 4:
            for dem in self.output_dems:
                (five_deg_n, five_deg_w) = dem[0]

                tif_data_region = Path(f'n{five_deg_n}w{five_deg_w}_dem.tif')
                flt_data_region = Path(f'n{five_deg_n}w{five_deg_w}_dem.flt')
                self.tif_file = self.datapath / tif_data_region
                self.flt_file = self.datapath / flt_data_region
                flt_filenames.append(self.flt_file)
                tif_filenames.append(self.tif_filenames)

        else:
            (five_deg_n, five_deg_w) = self.output_dems
            tif_data_region = Path(f'n{five_deg_n}w{five_deg_w}_dem.tif')
            flt_data_region = Path(f'n{five_deg_n}w{five_deg_w}_dem.flt')
            self.tif_file = self.datapath / tif_data_region
            self.flt_file = self.datapath / flt_data_region
            flt_filenames = self.flt_file
            tif_filenames = self.tif_file
        flt_filenames = [flt_filenames]
        tif_filenames = [tif_filenames]
        return flt_filenames, tif_filenames

    def _get_nearest_tile(self) -> Tuple[Tuple[int]]:
        '''
        This method will take the lat/long tuples and retrieve the nearest dem.

        NOTE: Only works if lat/long are split across 2 DEMs
                2+ DEMs is NOT implemented

        Always want the lowest (closest to equator and furthest from center divide) bound:
            n30w120 --> N30-N35, W120-W115
            N30-N35, W120-W115 --> (N29.99958333-N34.99958333,W120.0004167-W115.0004167)

        For simplicity, assume we are in upper hemisphere (N)
            and left of center divide (W)

        Arguments:
            None

        Returns:
            Tuple[Tuple[int]]
                The coordinates needed for loading the correct DEM

        '''
        # round up on latitdue
        five_deg_north_min = self.lat[0]
        five_deg_north_min_min = round_up_to_multiple(five_deg_north_min, 5)
        if round(five_deg_north_min_min - five_deg_north_min, 2) <= 0.01:
            min_max = round_up_to_multiple(five_deg_north_min, 5)
        else:
            min_max = round_down_to_multiple(five_deg_north_min, 5)

        # five_deg_north_min_max = round_up_to_multiple(five_deg_north_min, 5)

        five_deg_north_max = self.lat[1]
        five_deg_north_max_min = round_down_to_multiple(five_deg_north_max, 5)
        if round(five_deg_north_max - five_deg_north_max_min, 2) <= 0.01:
            max_min = round_up_to_multiple(five_deg_north_max, 5)
        else:
            max_min = round_down_to_multiple(five_deg_north_max, 5)

        self.five_deg_north_min = min_max
        self.five_deg_north_max = max_min

        # round down on longitude (w is negative)
        five_deg_west_max = abs(self.long[0])
        five_deg_west_max_min = round_down_to_multiple(five_deg_west_max, 5)
        if round(five_deg_west_max - five_deg_west_max_min, 2) <= 0.01:
            max_min = round_down_to_multiple(five_deg_west_max, 5)
        else:
            max_min = round_up_to_multiple(five_deg_west_max, 5)

        five_deg_west_min = abs(self.long[1])
        five_deg_west_min_max = round_up_to_multiple(five_deg_west_min, 5)
        if round(five_deg_west_min_max - five_deg_west_min, 2) <= 0.01:
            min_max = round_down_to_multiple(five_deg_west_min, 5)
        else:
            min_max = round_up_to_multiple(five_deg_west_min, 5)

        self.five_deg_west_min = min_max
        self.five_deg_west_max = max_min

        if self.five_deg_north_min == self.five_deg_north_max and \
                self.five_deg_west_max == self.five_deg_west_min:
            return ((self.five_deg_north_min, self.five_deg_west_max))

        else:
            return ((self.five_deg_north_min, self.five_deg_west_max),
                    (self.five_deg_north_max, self.five_deg_west_min))

    def _generate_contour_map(self) -> np.ndarray:
        '''
        Elevation in (m)

        TODO: only get data within lat/long range

        '''
        gdal_data = gdal.Open(str(self.tif_file))
        gdal_band = gdal_data.GetRasterBand(1)
        nodataval = gdal_band.GetNoDataValue()
        # convert to a numpy array
        data_array = gdal_data.ReadAsArray().astype(np.float)

        # replace missing values if necessary
        if np.any(data_array == nodataval):
            data_array[data_array == nodataval] = np.nan
        fig = plt.figure(figsize=(12, 8))
        fig.add_subplot(111)
        plt.contour(data_array, cmap='viridis')  # levels = list(range(0, 6000, 500))
        plt.title('Elevation Contours')
        # cbar = plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(f'img_{self.tif_file.stem}.png')

        return data_array

    def _read_flt(self) -> Tuple[Tuple[int]]:
        '''
        .flt file holds values for a single numeric measure, a value for each cell
            in the rectangular grid.
        The numeric values are in IEEE floating-point 32-bit (aka single-precision)
            signed binary format.
        The first number in the .flt file corresponds to the top left cell of the
            raster/grid.

        <filename>.hdr contains the header info:
            NCOLS                   6000
            NROWS                   6000
            XLLCENTER
            YLLCENTER
            CELLSIZE
            NODATA_VALUE
            BYTEORDER LSBFIRST


        Arguments:
            None

        Returns:
            Tuple[int]
                The indices of the lat/long coordinates ((x1, y1), (x2, y2), )
        '''
        for dem_file in self.flt_filenames:
            fid = open(dem_file, 'rb').read()
            arr = array('i')
            arr.frombytes(fid)
            # decdeg_calc = (2**32 / 360)
            # for bit in arr:
            # dec_degrees = bit / decdeg_calc
