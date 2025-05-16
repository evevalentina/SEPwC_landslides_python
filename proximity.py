"""This contins a single function to caluate the Euclidian
   distance from a value in a numpy ratser, using a rasterio
   object as a template. It matches GDAL's 'proximity' function."""

import math
import rasterio
from scipy import spatial
import numpy as np

def proximity(raster, rasterised, value):
    """Calculate distance to source pixel value in every
    cell of the raster.

    Calculates Euclidian distance. Matches GDAL's "proximity" function.

    Input: raster = rastioio raster object used as template
           rasterised = a numpy array
           value = the values in raster that are used as target 
                    (i.e. where distance will == 0)

    Output: A numpy array with dimensions of the input raster with shortest
            distance to any pixels in the input raster of value.

    Example usage:
        raster = rasterio.open("Rasterized_polygon.tif")
        distance = proximity(raster, 1)
    """

    # pylint: disable=too-many-locals
    gt = raster.transform
    pixel_size_x = gt[0]
    pixel_size_y =-gt[4]
    dx = math.sqrt(pixel_size_x * pixel_size_y)

    height, width = rasterised.shape # Find the height and width of the array
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(raster.transform, rows, cols)
    # They are actually lists, convert them to arrays
    xcoords = np.array(xs).reshape(height, width)
    ycoords = np.array(ys).reshape(height, width)


    # find coords of points that have the target value in the rasterised raster
    xindex, yindex = np.where(rasterised==value)
    source_coords = []
    for x, y in zip(xindex, yindex):
        source_coords.append([xcoords[x,y],ycoords[x,y]])

    # now create all coords in the raster where we want distance
    target_coords = []
    for xs, ys in zip(xcoords, ycoords):
        for x, y in zip(xs,ys):
            target_coords.append([x, y])

    source_coords = np.array(source_coords)
    target_coords = np.array(target_coords)

    distance = np.ones((height,width))*float('inf')
    for coords in source_coords:
        dist = spatial.distance.cdist([coords], target_coords)
        dist = dist.reshape(height,width)
        distance = np.minimum(distance,dist)

    distance = distance / dx

    return distance
