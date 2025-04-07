# ATL03 Photon Point Cloud Classification

A Python implementation of an Adaptive DBSCAN algorithm based on Bayesian Decision Theory (BDT-ADBSCAN) for classifying ICESat-2 ATL03 photon point clouds. This tool processes ATL03 photon data and classifies points into ground, canopy, canopy top, and noise classes.

## Features

- **Adaptive DBSCAN**: Implements a density-based clustering algorithm with adaptive parameters for each point
- **Point Cloud Classification**: Classifies photon points into ground, canopy, canopy top, and noise
- **Segment Processing**: Processes large datasets in manageable segments
- **Visualization**: Generates visualizations of classification results
- **Batch Processing**: Can process multiple beams from the same ATL03 file

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/atl03-classification.git
   cd atl03-classification
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

The script can be run from the command line:

```bash
python atl03_classifier.py --input /path/to/ATL03_file.h5 --output /path/to/output_folder --beam gt1r
```

#### Arguments:

- `--input`, `-i`: Path to the ATL03 h5 file (required)
- `--output`, `-o`: Output directory for results (default: "results" in the input file directory)
- `--beam`, `-b`: Specific beam to process (default: first available beam)
- `--all-beams`, `-a`: Process all available beams
- `--verbose`, `-v`: Enable verbose logging

### Python API

You can also use the tool as a Python module:

```python
from atl03_classifier import ATL03Processor

# Initialize processor
processor = ATL03Processor(
    file_path="/path/to/ATL03_file.h5",
    output_folder="/path/to/output_folder"
)

# Process a specific beam
processor.process(beam="gt1r")
```

## Output

The script produces the following outputs:

1. CSV files with classified point data
2. Visualization of all classified points
3. Separate visualizations for each point class (ground, canopy, canopy top, noise)

## Algorithm Details

The BDT-ADBSCAN algorithm adapts DBSCAN by:

1. Estimating local density around each point
2. Calculating adaptive epsilon values based on local density
3. Using these adaptive parameters for clustering
4. Applying classification rules based on cluster characteristics

## License

This project is licensed under the MIT License - see the LICENSE file for details.
