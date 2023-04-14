from osgeo import gdal
from matplotlib import pyplot as plt
from typing import DefaultDict
import numpy as np

def display_tif_bands(file, title, bounds=None):
    if bounds:
        y0, y1, x0, x1 = bounds
        directory = "images/trees/"
    else:
        directory = "images/"

    img = gdal.Open(file)
    num_bands = img.RasterCount
    bands: DefaultDict = {}
    for i in range(num_bands):
        if bounds:
            b = img.GetRasterBand(i + 1)
            band = b.ReadAsArray()
            bands[f'b_{i}'] = band[y0:y1+1, x0:x1+1]
        else:
            b = img.GetRasterBand(i + 1)
            bands[f'b_{i}'] = b.ReadAsArray()
        plt.title(f'{title}_band_{i}')
        plt.imshow(bands[f'b_{i}'])
        plt.savefig(f'{directory}{title}_band_{i}.png')
        plt.show()

    return bands

def display_tif_rgb(bands, title):
    stacked_bands = []
    for band in bands.keys():
        band_array = bands[band]
        stacked_bands.append(band_array)
    img = np.dstack(stacked_bands)
    plt.title(f'{title}_stacked_rgb')
    plt.imshow(img)
    plt.savefig(f'images/trees/{title}_stacked_rgb.png')
    plt.show()

if __name__ == "__main__":
    may_dir = "source/may/"
    dec_dir = "source/dec/"

    dec_RE = dec_dir + "result_RedEdge.tif"
    # may_RE = may_dir + "raw_10May2021_90deg43m85pct255deg_result_RedEdge (1).tif"
    #
    dec_NIR = dec_dir + "result_NIR.tif"
    # may_NIR = may_dir + "raw_10May2021_90deg43m85pct255deg_result_NIR.tif"
    #

    # display_tif(may_RE, "may_RE")
    #
    # display_tif(dec_NIR,"dec_NIR")
    # display_tif(may_NIR, "may_NIR")

    dec_rgb = dec_dir + "raw_18Dec2020_result.tif"
    may_rgb = may_dir + "raw_10May2021_90deg43m85pct255deg_result.tif"
    # display_tif(dec_rgb, "dec_rgb")
    # display_tif(may_rgb, "may_rgb")

    # # test to debug why the mean is 255
    # # bands = display_tif_bands(dec_rgb,"Falcataria_Moluccana_0", bounds=[2187,3018,1841,2541])
    # # display_tif_rgb(bands, "Falcataria_Moluccana_0")
    # 
    # '''
    # Ficus Variegata|2422|2962|812|1348
    # Spathodea Campanulatum|4106|4481|2381|2836
    # '''
    bands = display_tif_bands(dec_rgb, "Spathodea Campanulatum_0", bounds=[4106, 4481, 2381, 2836])
    # display_tif_rgb(bands, "Spathodea Campanulatum_0")

    band_re = display_tif_bands(dec_RE, "Spathodea Campanulatum_0_re", bounds=[4106, 4481, 2381, 2836])
    # display_tif_bands(band_re, "Spathodea Campanulatum_0")

    band_nir = display_tif_bands(dec_NIR, "Spathodea Campanulatum_0_nir", bounds=[4106, 4481, 2381, 2836])
    # display_tif_bands(band_nir, "Spathodea Campanulatum_0")
    # 
    # print("done")
    # bands = display_tif_bands(dec_rgb, "dec_rgb")
    # display_tif_rgb(bands, "dec_rgb")
    #
    # bands = display_tif_bands(may_rgb, "may_rgb")
    # display_tif_rgb(bands, "may_rgb")