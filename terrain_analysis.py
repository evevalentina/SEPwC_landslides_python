"""
IMPORTANT- please can this exam be ran in a MAC OR LINEUX opperating system as
the processing is at times inconsistant in windows
This module implements landslide risk analysis using rasterio.
It takes various geographic data inputs and produces a probability map
of landslide occurrence.
"""

from dataclasses import dataclass
from typing import List, Tuple
import argparse

import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shapely.geometry

@dataclass
class RasterData:
    """Container for raster data."""
    topo: rasterio.DatasetReader
    geo: rasterio.DatasetReader
    lc: rasterio.DatasetReader
    slope: rasterio.DatasetReader
    fault_dist: rasterio.DatasetReader

def convert_to_rasterio(raster_data, template_raster):
    """Convert numpy array to rasterio dataset."""
    profile = template_raster.profile.copy()
    profile.update(
        dtype=raster_data.dtype,
        height=raster_data.shape[0],
        width=raster_data.shape[1],
        count=1,
        compress='lzw'
    )
    with rasterio.open("temp_raster.tif", 'w', **profile) as dst:
        dst.write(raster_data, 1)
    return rasterio.open("temp_raster.tif")

def extract_values_from_raster(raster, shape_object):
    """
    Extract values from a raster at the locations specified by shape objects.
    Args:
        raster: A rasterio dataset
        shape_object: List of shapely geometry objects (Points or Polygons)
    Returns:
        List of values from the raster at the specified locations
    """
    values = []
    for geom in shape_object:
        if isinstance(geom, shapely.geometry.Point):
            x, y = geom.x, geom.y
        else:  # Polygon or other geometry
            x, y = geom.centroid.x, geom.centroid.y
        # Check if the point is within the raster bounds
        if (x < raster.bounds.left or x > raster.bounds.right or
            y < raster.bounds.bottom or y > raster.bounds.top):
            # If point is outside bounds, use the nearest valid pixel
            x = max(raster.bounds.left, min(x, raster.bounds.right))
            y = max(raster.bounds.bottom, min(y, raster.bounds.top))
        # Get the row and column indices for the geometry
        row, col = raster.index(x, y)
        # Ensure indices are within bounds
        row = max(0, min(row, raster.height - 1))
        col = max(0, min(col, raster.width - 1))
        # Read the value at that location
        value = raster.read(1)[row, col]
        values.append(value)
    return values

def make_classifier(features, target, verbose=False):
    """
    Create and train a Random Forest classifier.
    Args:
        features: Features DataFrame
        target: Target values
        verbose: Whether to print additional information
    Returns:
        Trained Random Forest classifier
    """
    # Split the data into training and testing sets
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    # Create and train the classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(train_features, train_target)
    if verbose:
        # Calculate and print accuracy
        train_accuracy = classifier.score(train_features, train_target)
        test_accuracy = classifier.score(test_features, test_target)
        print(f"Training accuracy: {train_accuracy:.3f}")
        print(f"Testing accuracy: {test_accuracy:.3f}")
    return classifier

def make_prob_raster_data(raster_data: RasterData, classifier):
    """
    Generate probability predictions for each pixel in the raster.
    Args:
        raster_data: Container with all raster data
        classifier: Trained Random Forest classifier
    Returns:
        Numpy array of probability predictions
    """
    # Get the shape of the input rasters
    height, width = raster_data.topo.shape
    # Create arrays for each feature
    elev = raster_data.topo.read(1)
    fault = raster_data.fault_dist.read(1)
    slope_data = raster_data.slope.read(1)
    lc_data = raster_data.lc.read(1)
    geo_data = raster_data.geo.read(1)
    # Reshape arrays for prediction
    feature_matrix = np.column_stack((
        elev.flatten(),
        fault.flatten(),
        slope_data.flatten(),
        lc_data.flatten(),
        geo_data.flatten()
    ))
    # Make predictions
    probabilities = classifier.predict_proba(feature_matrix)[:, 1]
    # Reshape back to original dimensions
    return probabilities.reshape(height, width)

# pylint: disable=too-many-arguments, too-many-positional-arguments

def create_dataframe(topo, geo=None, lc=None, dist_fault=None,
                     slope=None, shape=None, landslides=None):
    """
    Create a GeoDataFrame with features for the classifier.
    Args:
        topo: Either a RasterData object or a topography raster
        geo: Geology raster (optional if topo is RasterData)
        lc: Landcover raster (optional if topo is RasterData)
        dist_fault: Distance from faults raster (optional if topo is RasterData)
        slope: Slope raster (optional if topo is RasterData)
        shape: List of geometry objects (required)
        landslides: Whether these are landslide locations (1) or not (0) (required)
    Returns:
        GeoDataFrame with features and target variable
    """
    if isinstance(topo, RasterData):
        raster_data = topo
        shape = geo  # In this case, geo is actually the shape parameter
        landslides = lc  # In this case, lc is actually the landslides parameter
    else:
        raster_data = RasterData(
            topo=topo,
            geo=geo,
            lc=lc,
            slope=slope,
            fault_dist=dist_fault
        )

    # Extract values for each feature
    elev_values = extract_values_from_raster(raster_data.topo, shape)
    fault_values = extract_values_from_raster(raster_data.fault_dist, shape)
    slope_values = extract_values_from_raster(raster_data.slope, shape)
    lc_values = extract_values_from_raster(raster_data.lc, shape)
    geo_values = extract_values_from_raster(raster_data.geo, shape)

    df = pd.DataFrame({
        'elev': elev_values,
        'fault': fault_values,
        'slope': slope_values,
        'LC': lc_values,
        'Geol': geo_values,
        'ls': [landslides] * len(shape)
    })
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df)
    return gdf

def calculate_slope(topo: rasterio.DatasetReader) -> Tuple[np.ndarray, rasterio.DatasetReader]:
    """Calculate slope from topography."""
    elevation = topo.read(1)
    slope = np.zeros_like(elevation)
    for i in range(1, elevation.shape[0]-1):
        for j in range(1, elevation.shape[1]-1):
            dz_dx = (elevation[i, j+1] - elevation[i, j-1]) / (2 * topo.res[0])
            dz_dy = (elevation[i+1, j] - elevation[i-1, j]) / (2 * topo.res[1])
            slope[i, j] = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) * 180 / np.pi
    return slope, convert_to_rasterio(slope, topo)

'''def calculate_slope_vectorized(topo: rasterio.DatasetReader) -> Tuple[np.ndarray, rasterio.DatasetReader]:
    """Calculate slope from topography using vectorized operations."""
    elevation = topo.read(1)
    resolution_x, resolution_y = topo.res

    # Use array slicing for finite differences
    dz_dx = (elevation[:, 2:] - elevation[:, :-2]) / (2 * resolution_x)
    dz_dy = (elevation[2:, :] - elevation[:-2, :]) / (2 * resolution_y)

    # Handle boundary conditions (e.g., by padding or using a different approach)
    # This is a simplified example and might need adjustments based on desired boundary behavior

    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) * 180 / np.pi

    # The shape of 'slope' will be smaller than 'elevation'. You'll need to
    # decide how to handle the boundaries (e.g., pad with zeros or crop).
    # For simplicity here, we'll just convert the calculated part to a raster.
    # A more robust implementation would handle the full output size.
    profile = topo.profile.copy()
    profile.update(dtype=slope.dtype, count=1, compress='lzw', height=slope.shape[0], width=slope.shape[1])
    with rasterio.open("temp_slope.tif", 'w', **profile) as dst:
        dst.write(slope, 1)
    return slope, rasterio.open("temp_slope.tif")'''

def calculate_fault_distance(topo: rasterio.DatasetReader,
                           faults: gpd.GeoDataFrame) -> Tuple[np.ndarray, rasterio.DatasetReader]:
    """Calculate distance from faults."""
    elevation = topo.read(1)
    fault_dist = np.zeros_like(elevation)
    for i in range(elevation.shape[0]):
        for j in range(elevation.shape[1]):
            x, y = topo.xy(i, j)
            point = shapely.geometry.Point(x, y)
            distances = [point.distance(fault) for fault in faults.geometry]
            fault_dist[i, j] = min(distances)
    return fault_dist, convert_to_rasterio(fault_dist, topo)

def generate_non_landslide_points(topo: rasterio.DatasetReader,
                            num_points: int) -> List[shapely.geometry.Point]:
    """Generate random non-landslide points."""
    np.random.seed(42)
    points = []
    for _ in range(num_points):
        i = np.random.randint(0, topo.height)
        j = np.random.randint(0, topo.width)
        x, y = topo.xy(i, j)
        points.append(shapely.geometry.Point(x, y))
    return points

def save_probability_map(prob_map: np.ndarray,
                        template: rasterio.DatasetReader,
                        output_path: str):
    """Save probability map to file."""
    profile = template.profile.copy()
    profile.update(
        dtype='float32',
        count=1,
        compress='lzw'
    )
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prob_map.astype('float32'), 1)

def parse_arguments():
    """Parse command line arguments."""
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
    return parser.parse_args()

def prepare_training_data(raster_data: RasterData, landslides: gpd.GeoDataFrame):
    """Prepare training data for the classifier."""
    landslide_points = list(landslides.geometry)
    non_landslide_points = generate_non_landslide_points(
        raster_data.topo,
        len(landslide_points)
    )
    # Create dataframes for both classes
    landslide_df = create_dataframe(raster_data, landslide_points, 1)
    non_landslide_df = create_dataframe(raster_data, non_landslide_points, 0)
    # Combine dataframes
    return pd.concat([landslide_df, non_landslide_df])

def main():
    """Main function to run the landslide risk analysis."""
    args = parse_arguments()
    if args.verbose:
        print("Loading input files...")
    # Load input files
    topo = rasterio.open(args.topography)
    geo = rasterio.open(args.geology)
    lc = rasterio.open(args.landcover)
    faults = gpd.read_file(args.faults)
    landslides = gpd.read_file(args.landslides)
    if args.verbose:
        print("Calculating slope...")
        '''temporerily changing calculate_slope to calculate_slope_vectorized
        to see if it has a positive change/not'''
    _, slope_raster = calculate_slope(topo)
    if args.verbose:
        print("Calculating distance from faults...")
    _, fault_dist_raster = calculate_fault_distance(topo, faults)
    # Create raster data container
    raster_data = RasterData(
        topo=topo,
        geo=geo,
        lc=lc,
        slope=slope_raster,
        fault_dist=fault_dist_raster
    )
    if args.verbose:
        print("Preparing training data...")
    training_data = prepare_training_data(raster_data, landslides)
    if args.verbose:
        print("Training classifier...")
    # Train classifier
    features = ['elev', 'fault', 'slope', 'LC', 'Geol']
    classifier = make_classifier(training_data[features], training_data['ls'], args.verbose)
    if args.verbose:
        print("Generating probability map...")
    # Generate probability map
    prob_map = make_prob_raster_data(raster_data, classifier)
    if args.verbose:
        print("Saving output...")
    # Save output
    save_probability_map(prob_map, topo, args.output)
    if args.verbose:
        print("Done!")

if __name__ == '__main__':
    main()
