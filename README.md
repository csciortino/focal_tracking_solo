# Macaque Individual Tracking System

A comprehensive video analysis pipeline for identifying and tracking individual macaques in camera trap footage.

## Overview

This system uses computer vision and machine learning to:

1. **Detect** primates in camera trap videos using YOLO object detection
2. **Track** individuals across frames using motion-based tracking
3. **Extract features** from detected animals for identification
4. **Cluster** tracklets to propose individual identities using HDBSCAN
5. **Generate video clips** for each proposed individual for validation

## Project Structure

```
focal_tracking/
├── data/raw/                    # Camera trap videos
├── src/macaque_tracker/         # Core tracking modules
│   ├── detector.py             # YOLO-based detection
│   ├── tracker.py              # Simple tracking algorithm
│   ├── clustering.py           # Individual identification
│   └── video_utils.py          # Video processing utilities
├── notebooks/                   # Analysis workflows
│   ├── 01_video_preprocessing_detection.ipynb
│   ├── 02_individual_clustering.ipynb
│   └── 03_video_snippet_extraction.ipynb
├── output/                      # Results and extracted clips
├── requirements.txt             # Python dependencies
└── setup.py                     # Package installation
```

## Quick Start

### 1. Installation

```bash
# Make a virtual env (currently using 3.11 for requirements compatibility)
python3.11 -m venv ~/.venv/focal3.11

# Start the environment
source ~/.venv/focal3.11/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Video Data

Place your camera trap videos in the `data/raw/` directory. The system supports common video formats (MP4, AVI, MOV).

### 3. Run Analysis Pipeline

Start the notebook server this way:
```
# activate the virtual environment
source ~/.venv/focal3.11/bin/activate

# add this virtual environment to the available kernels in jupyter
python -m ipykernel install --user --name=focal3.11 --display-name "Python 3.11 (focal tracking)"

# start the notebook server
jupyter notebook
```
In the notebook interface:
• When opening or creating a notebook, select `Kernel` > `Change Kernel` > `Python 3.11 (focal tracking)` to use the correct environment.


Execute the Jupyter notebooks in sequence:

1. **Detection & Tracking**: `notebooks/01_video_preprocessing_detection.ipynb`
   - Processes videos to detect and track primates
   - Extracts visual features for each tracklet
   - Saves results for clustering analysis

2. **Individual Identification**: `notebooks/02_individual_clustering.ipynb`
   - Clusters tracklets to propose individual identities
   - Uses HDBSCAN for robust clustering
   - Provides visualization and validation tools

3. **Video Clip Extraction**: `notebooks/03_video_snippet_extraction.ipynb`
   - Extracts video clips for each proposed individual
   - Creates organized libraries for manual validation
   - Generates usage statistics and previews

## Key Features

### Detection & Tracking
- **YOLOv8** for robust animal detection
- **Simple centroid tracking** for motion continuity
- **Feature extraction** using color histograms and texture analysis
- **Configurable confidence thresholds** and tracking parameters

### Individual Identification
- **HDBSCAN clustering** for automatic individual proposals
- **Dimensionality reduction** with UMAP/PCA for visualization
- **Multiple clustering algorithms** (HDBSCAN, DBSCAN) with parameter optimization
- **Silhouette analysis** for cluster quality assessment

### Video Processing
- **Temporal segmentation** to identify continuous appearance periods
- **Buffer-based clip extraction** with configurable padding
- **Organized output** with individual-specific directories
- **Comprehensive indexing** for research applications

## Output Files

The system generates several types of output:

### Detection Results
- `detections_*.csv`: Bounding boxes, confidence scores, track IDs
- `tracklet_features_*.npz`: Visual features for each tracklet
- `track_mapping_*.csv`: Track ID to feature mapping

### Clustering Results  
- `individual_mapping_*.csv`: Track ID to individual ID assignments
- `detections_with_individuals_*.csv`: Enhanced detection data
- `individual_statistics_*.csv`: Per-individual summary statistics
- `clustering_summary_*.csv`: Method parameters and performance

### Video Clips
- `clips/individual_XX/`: Video clips organized by proposed individual
- `clip_index_*.csv`: Comprehensive clip inventory
- `clip_statistics_*.csv`: Storage and duration statistics

## Configuration

Key parameters can be adjusted in the notebooks:

### Detection
- `confidence`: Detection confidence threshold (default: 0.3)
- `frame_skip`: Process every N frames (default: 5)

### Tracking  
- `max_disappeared`: Frames before removing lost tracks (default: 10)
- `max_distance`: Maximum distance for track association (default: 150)

### Clustering
- `min_cluster_size`: Minimum tracklets per individual (default: 3)
- `method`: Clustering algorithm ('hdbscan' or 'dbscan')

### Clip Extraction
- `buffer_seconds`: Temporal padding around detections (default: 3.0)
- `max_clips_per_individual`: Limit clips to manage storage (default: 10)

## Research Applications

### Individual Behavior Analysis
- Study behavior patterns unique to each macaque
- Analyze temporal activity patterns
- Investigate site usage preferences

### Social Network Analysis
- Identify co-occurrence patterns between individuals
- Study group dynamics and social interactions
- Map spatial associations

### Population Monitoring
- Track individual presence/absence over time
- Monitor population size and composition
- Study habitat usage patterns

### Machine Learning
- Train individual recognition models using extracted clips
- Develop automated behavior classification systems
- Create datasets for computer vision research

## Technical Notes

### Dependencies
- **Computer Vision**: OpenCV, Ultralytics YOLO
- **Machine Learning**: scikit-learn, HDBSCAN, UMAP
- **Data Science**: NumPy, pandas, matplotlib
- **Video Processing**: MoviePy, OpenCV

### Performance Considerations
- Processing time scales with video duration and frame rate
- Memory usage depends on number of simultaneous tracklets
- Storage requirements scale with number of individuals and clip duration

### Limitations
- Detection performance depends on video quality and lighting
- Tracking may fail with rapid movements or occlusions
- Individual identification relies on visual features only
- Manual validation recommended for research applications

## Contributing

To extend the system:

1. **Add new detectors**: Implement detection interface in `detector.py`
2. **Improve tracking**: Enhance algorithms in `tracker.py` 
3. **Add clustering methods**: Extend options in `clustering.py`
4. **Custom features**: Modify feature extraction in `detector.py`

## Citation

If you use this system in your research, please cite:

```
Macaque Individual Tracking System
Camera trap video analysis for primate behavioral ecology
[Your Institution/Research Group]
```

## Support

For questions or issues:
1. Check the notebook outputs for debugging information
2. Verify video file formats and accessibility  
3. Ensure sufficient disk space for output files
4. Review parameter settings for your specific use case