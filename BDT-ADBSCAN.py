"""
ATL03 Photon Point Cloud Classification
======================================

This module implements an Adaptive DBSCAN algorithm based on Bayesian Decision Theory
for classifying ICESat-2 ATL03 photon point clouds into ground, canopy, and noise points.


"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
import matplotlib.colors as mcolors
import os
from scipy.signal import find_peaks
import time
import logging
import argparse
from typing import List, Dict, Tuple, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("atl03_classifier")


class BDT_ADBSCAN:
    """
    Adaptive DBSCAN algorithm based on Bayesian Decision Theory for
    ATL03 photon point cloud noise reduction and classification.

    Parameters
    ----------
    min_pts : int
        Minimum number of points in a neighborhood for a point to be considered a core point
    eps_init : float
        Initial epsilon value for the neighborhood search
    adapt_factor : float
        Factor to adjust the adaptive epsilon calculation
    """

    def __init__(self, min_pts: int = 5, eps_init: float = 5.0, adapt_factor: float = 0.8):
        self.min_pts = min_pts
        self.eps_init = eps_init
        self.adapt_factor = adapt_factor

    def estimate_local_density(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate the local density around each point.

        Parameters
        ----------
        X : np.ndarray
            Input data points of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Array of average distances to k-nearest neighbors for each point
        """
        # Use KD-tree to calculate the average distance to k nearest neighbors
        kdtree = KDTree(X)
        k = min(self.min_pts, X.shape[0] - 1)
        distances, _ = kdtree.query(X, k=k + 1)  # +1 because it includes the point itself
        # Return average distance to k nearest neighbors (excluding self)
        return np.mean(distances[:, 1:], axis=1)

    def adaptive_eps(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate adaptive epsilon values for each point.

        Parameters
        ----------
        X : np.ndarray
            Input data points of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Array of adaptive epsilon values for each point
        """
        local_densities = self.estimate_local_density(X)
        # Calculate scaling factor as inverse proportion of local density
        density_factors = np.clip(
            np.median(local_densities) / local_densities,
            0.5, 2.0  # Limit the factor range to prevent extreme values
        )
        # Apply adaptive factor
        eps_values = self.eps_init * density_factors * self.adapt_factor
        return eps_values

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform clustering and return labels.

        Parameters
        ----------
        X : np.ndarray
            Input data points of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Cluster labels for each point (-1 represents noise)
        """
        if X.shape[0] < self.min_pts:
            return np.zeros(X.shape[0]) - 1  # Mark all points as noise

        # Calculate adaptive epsilon values
        eps_values = self.adaptive_eps(X)

        # Initialize labels as noise
        labels = np.zeros(X.shape[0]) - 1

        # Current cluster ID
        current_cluster = 0

        # Traverse all unclassified points
        for i in range(X.shape[0]):
            if labels[i] != -1:
                continue  # Skip already classified points

            # Use KD-tree to find points within eps_i range
            kdtree = KDTree(X)
            idx = kdtree.query_ball_point(X[i], eps_values[i])

            # If not enough points in neighborhood, mark as noise
            if len(idx) < self.min_pts:
                continue

            # Otherwise, start a new cluster
            labels[i] = current_cluster

            # Expand cluster
            seed_idx = idx.copy()
            seed_idx.remove(i)  # Remove current point

            for j in seed_idx:
                if labels[j] == -1:  # Unclassified point
                    labels[j] = current_cluster

                    # Find j's neighborhood
                    j_neighbors = kdtree.query_ball_point(X[j], eps_values[j])

                    # If j's neighborhood is large enough, add unprocessed points to seed points
                    if len(j_neighbors) >= self.min_pts:
                        for neighbor in j_neighbors:
                            if neighbor not in seed_idx and labels[neighbor] == -1:
                                seed_idx.append(neighbor)

            current_cluster += 1

        return labels


class ATL03Processor:
    """
    Processor for ATL03 photon point cloud data.

    This class handles reading, segmenting, classifying, and visualizing
    ATL03 photon point cloud data.
    """

    # Class constants for point classification
    NOISE = 0
    GROUND = 1
    CANOPY = 2
    CANOPY_TOP = 3

    # Class labels for visualization
    CLASS_LABELS = {
        NOISE: 'Noise',
        GROUND: 'Ground',
        CANOPY: 'Canopy',
        CANOPY_TOP: 'Canopy Top'
    }

    # Class colors for visualization
    CLASS_COLORS = {
        NOISE: 'gray',
        GROUND: 'green',
        CANOPY: 'blue',
        CANOPY_TOP: 'red'
    }

    def __init__(self, file_path: str, output_folder: str):
        """
        Initialize the ATL03 processor.

        Parameters
        ----------
        file_path : str
            Path to the ATL03 h5 file
        output_folder : str
            Path to the output folder
        """
        self.file_path = file_path
        self.output_folder = output_folder

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

    def read_atl03_data(self, beam: str = 'gt1r') -> Optional[pd.DataFrame]:
        """
        Read ATL03 data from the h5 file.

        Parameters
        ----------
        beam : str
            Beam to process (e.g., 'gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l')

        Returns
        -------
        pd.DataFrame or None
            DataFrame containing photon data or None if reading fails
        """
        try:
            with h5py.File(self.file_path, 'r') as f:
                # Check all available beams in the file
                available_beams = [key for key in f.keys() if key.startswith('gt')]
                logger.info(f"Available beams: {available_beams}")

                if beam not in available_beams:
                    logger.warning(f"Beam {beam} not found in file. Using {available_beams[0]} instead.")
                    beam = available_beams[0]

                # Read latitude, longitude, height, and confidence level
                lats = f[f'{beam}/heights/lat_ph'][:]
                lons = f[f'{beam}/heights/lon_ph'][:]
                heights = f[f'{beam}/heights/h_ph'][:]

                # Read photon confidence
                # In ATL03, confidence is 0-4, with 4 being highest confidence
                conf = f[f'{beam}/heights/signal_conf_ph'][:, 0]  # Take first column for surface classification

                # Read delta_time and convert to seconds from orbit start
                delta_time = f[f'{beam}/heights/delta_time'][:]

                # Try to read more useful fields if possible
                try:
                    dist_along_track = f[f'{beam}/heights/dist_ph_along'][:]
                except KeyError:
                    # If not present, calculate approximate along-track distance
                    dist_along_track = np.zeros_like(delta_time)
                    if len(delta_time) > 1:
                        # Use simple linear interpolation, not considering Earth curvature
                        dist_along_track = (delta_time - delta_time[0]) * 7000  # Assuming satellite speed of ~7 km/s

                # Create DataFrame
                data = pd.DataFrame({
                    'longitude': lons,
                    'latitude': lats,
                    'height': heights,
                    'confidence': conf,
                    'delta_time': delta_time,
                    'dist_along_track': dist_along_track
                })

                logger.info(f"Successfully read {len(data)} photon points from beam {beam}")
                return data

        except Exception as e:
            logger.error(f"Error reading ATL03 data: {str(e)}")
            return None

    def segment_track(self, df: pd.DataFrame, segment_length: float = 100.0) -> List[pd.DataFrame]:
        """
        Divide the track into fixed-length segments.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing point cloud data
        segment_length : float
            Length of each segment (meters)

        Returns
        -------
        list
            List of DataFrames, each representing a segment
        """
        # Ensure dist_along_track is monotonically increasing
        if not np.all(np.diff(df['dist_along_track']) >= 0):
            df = df.sort_values(by='dist_along_track').reset_index(drop=True)

        min_dist = df['dist_along_track'].min()
        max_dist = df['dist_along_track'].max()

        segments = []
        for start in np.arange(min_dist, max_dist, segment_length):
            end = start + segment_length
            segment = df[(df['dist_along_track'] >= start) & (df['dist_along_track'] < end)]

            if len(segment) > 10:  # Only process segments with enough points
                segments.append(segment)

        logger.info(f"Track divided into {len(segments)} segments")
        return segments

    def classify_segment(self, segment: pd.DataFrame, confidence_threshold: int = 2) -> pd.DataFrame:
        """
        Classify points in a segment.

        Parameters
        ----------
        segment : pd.DataFrame
            Segment of point cloud data
        confidence_threshold : int
            ATL03 confidence threshold, points above this are considered potential signal

        Returns
        -------
        pd.DataFrame
            DataFrame with classification labels
        """
        # Initialize classification column
        segment = segment.copy()
        segment['class'] = self.NOISE  # 0: Unclassified/Noise

        # Step 1: Use ATL03 confidence as first-level filter
        potential_signal = segment[segment['confidence'] >= confidence_threshold].copy()

        if len(potential_signal) < 10:
            # If not enough high confidence points, try with lower threshold
            potential_signal = segment.copy()

        # Step 2: Prepare features for clustering
        # Use [along-track distance, height] as features
        X = potential_signal[['dist_along_track', 'height']].values

        # Normalize features
        scale_factors = [100.0, 5.0]  # Scaling factors for distance and height
        X_scaled = np.column_stack([
            X[:, 0] / scale_factors[0],
            X[:, 1] / scale_factors[1]
        ])

        # Step 3: Apply BDT-ADBSCAN clustering
        clusterer = BDT_ADBSCAN(min_pts=5, eps_init=0.5, adapt_factor=1.0)
        potential_signal['cluster'] = clusterer.fit_predict(X_scaled)

        # Step 4: Classify clusters
        # Add classification labels back to original DataFrame
        segment.loc[potential_signal.index, 'cluster'] = potential_signal['cluster']

        # Initialize all points as noise (class 0)
        segment['class'] = self.NOISE

        # Process each cluster
        for cluster_id in potential_signal['cluster'].unique():
            if cluster_id == -1:
                continue  # Skip noise points

            # Get points in current cluster
            cluster_points = potential_signal[potential_signal['cluster'] == cluster_id]

            # If cluster has too few points, treat as noise
            if len(cluster_points) < 10:
                continue

            # Get vertical range of cluster
            min_height = cluster_points['height'].min()
            max_height = cluster_points['height'].max()
            height_range = max_height - min_height

            # Clusters with small vertical range might be ground points
            if height_range < 1.0:
                segment.loc[cluster_points.index, 'class'] = self.GROUND  # 1: Ground
            else:
                # Find canopy top points
                # Group by along-track distance, highest point in each group is canopy top
                dist_bins = np.arange(
                    cluster_points['dist_along_track'].min(),
                    cluster_points['dist_along_track'].max() + 1,
                    1.0  # 1 meter interval
                )

                if len(dist_bins) < 2:
                    continue

                grouped = cluster_points.groupby(pd.cut(cluster_points['dist_along_track'], dist_bins))

                for _, group in grouped:
                    if len(group) > 0:
                        # Find highest point in each group
                        top_idx = group['height'].idxmax()
                        # Mark as canopy top
                        segment.loc[top_idx, 'class'] = self.CANOPY_TOP  # 3: Canopy top

                        # Mark remaining points as canopy
                        other_idx = group.index[group.index != top_idx]
                        segment.loc[other_idx, 'class'] = self.CANOPY  # 2: Canopy

        # Count points in each class
        class_counts = segment['class'].value_counts()
        logger.info("Point cloud classification results:")
        logger.info(f"Noise points: {class_counts.get(self.NOISE, 0)}")
        logger.info(f"Ground points: {class_counts.get(self.GROUND, 0)}")
        logger.info(f"Canopy points: {class_counts.get(self.CANOPY, 0)}")
        logger.info(f"Canopy top points: {class_counts.get(self.CANOPY_TOP, 0)}")

        return segment

    def visualize_classification(self, df: pd.DataFrame, beam: str) -> None:
        """
        Visualize classification results.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with classification labels
        beam : str
            Beam being processed
        """
        # Set large figure size
        plt.figure(figsize=(16, 10))

        # Plot points by class
        for class_id, color in self.CLASS_COLORS.items():
            subset = df[df['class'] == class_id]
            if len(subset) > 0:
                plt.scatter(
                    subset['dist_along_track'],
                    subset['height'],
                    c=color,
                    s=1,  # Small point size for plotting efficiency
                    alpha=0.7,
                    label=f"{self.CLASS_LABELS[class_id]} ({len(subset)})"
                )

        plt.xlabel('Along-track Distance (m)')
        plt.ylabel('Elevation (m)')
        plt.title(f'ATL03 Point Cloud Classification Results (Beam: {beam})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save image
        fig_path = os.path.join(self.output_folder, f"classification_viz_{beam}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Visualization saved to: {fig_path}")

        # Generate separate views for each class
        for class_id, label in self.CLASS_LABELS.items():
            subset = df[df['class'] == class_id]
            if len(subset) > 100:  # Only generate separate views for classes with enough points
                plt.figure(figsize=(16, 8))
                plt.scatter(
                    subset['dist_along_track'],
                    subset['height'],
                    c=self.CLASS_COLORS[class_id],
                    s=2,
                    alpha=0.8
                )
                plt.xlabel('Along-track Distance (m)')
                plt.ylabel('Elevation (m)')
                plt.title(f'{label} Points (Beam: {beam})')
                plt.grid(True, alpha=0.3)

                class_fig_path = os.path.join(self.output_folder, f"class_{class_id}_{beam}.png")
                plt.savefig(class_fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"{label} visualization saved to: {class_fig_path}")

    def process(self, beam: str = 'gt1r') -> None:
        """
        Process the ATL03 file for a specific beam.

        Parameters
        ----------
        beam : str
            Beam to process
        """
        logger.info(f"Processing file: {self.file_path}")
        start_time = time.time()

        # Read ATL03 data
        df = self.read_atl03_data(beam)

        if df is None or len(df) == 0:
            logger.error("Failed to read valid data")
            return

        logger.info(f"Read {len(df)} photon points")

        # Segment track
        segments = self.segment_track(df)

        # Process each segment and merge results
        all_results = []
        for i, segment in enumerate(segments):
            logger.info(f"Processing segment {i + 1}/{len(segments)}")
            classified_segment = self.classify_segment(segment)
            all_results.append(classified_segment)

        # Merge all results
        if all_results:
            result_df = pd.concat(all_results)

            # Save classification results
            output_file = os.path.join(self.output_folder, f"classified_atl03_{beam}.csv")
            result_df.to_csv(output_file, index=False)
            logger.info(f"Classification results saved to: {output_file}")

            # Generate visualization
            self.visualize_classification(result_df, beam)
        else:
            logger.warning("Failed to process any data segments")

        end_time = time.time()
        logger.info(f"Processing complete, time taken: {end_time - start_time:.2f} seconds")


def get_available_beams(file_path: str) -> List[str]:
    """
    Get list of available beams in an ATL03 file.

    Parameters
    ----------
    file_path : str
        Path to the ATL03 h5 file

    Returns
    -------
    list
        List of available beam names
    """
    try:
        with h5py.File(file_path, 'r') as f:
            return [key for key in f.keys() if key.startswith('gt')]
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        return []


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ATL03 Photon Point Cloud Classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to ATL03 h5 file')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory (defaults to "results" in the input file directory)')
    parser.add_argument('--beam', '-b', type=str, default=None,
                        help='Specific beam to process (if not specified, will use first available beam)')
    parser.add_argument('--all-beams', '-a', action='store_true',
                        help='Process all available beams')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    return parser.parse_args()


def main():
    """Main function to run the ATL03 processor."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate input file
    if not os.path.isfile(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    # Set output directory
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "results")

    # Get available beams
    available_beams = get_available_beams(args.input)
    if not available_beams:
        logger.error("No valid beams found in the input file")
        return

    # Process beams
    if args.all_beams:
        for beam in available_beams:
            processor = ATL03Processor(args.input, args.output)
            processor.process(beam)
    else:
        beam = args.beam if args.beam in available_beams else available_beams[0]
        processor = ATL03Processor(args.input, args.output)
        processor.process(beam)


if __name__ == "__main__":
    main()