import argparse
import rasterio
import scipy
import rasterio.mask
import numpy as np
import geopandas as gpd
import pandas as pd

import matplotlib.pyplot as plt  # For visualization
import os  # For file system interaction
import fiona  # For lower-level vector data handling
import shapely  # For geometric operations
import pyproj # For coordinate transformations

def convert_to_rasterio(raster_data, template_raster):
  
    return


def extract_values_from_raster(raster, shape_object):

    return


def make_classifier(x, y, verbose=False):

    return

def make_prob_raster_data(topo, geo, lc, dist_fault, slope, classifier):

    return

def create_dataframe(topo, geo, lc, dist_fault, slope, shape, landslides):

    return


def main():


    parser = argparse.ArgumentParser(
                     prog="Landslide hazard using ML",
                     description="Calculate landslide hazards using simple ML",
                     epilog="Copyright 2024, Jon Hill"
                     )
    parser.add_argument('--topography',
                    required=True,
                    help="topographic raster file")
    parser.add_argument('--geology',
                    required=True,
                    help="geology raster file")
    parser.add_argument('--landcover',
                    required=True,
                    help="landcover raster file")
    parser.add_argument('--faults',
                    required=True,
                    help="fault location shapefile")
    parser.add_argument("landslides",
                    help="the landslide location shapefile")
    parser.add_argument("output",
                    help="the output raster file")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")

    args = parser.parse_args()


if __name__ == '__main__':
    main()
