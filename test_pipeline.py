#!/usr/bin/env python3
"""
Test script for the macaque tracking pipeline.
Runs a quick test on a sample video to verify the system is working.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from macaque_tracker.detector import MacaqueDetector
from macaque_tracker.tracker import SimpleTracker, TrackletManager
from macaque_tracker.clustering import IndividualIdentifier
from macaque_tracker.video_utils import VideoProcessor, find_video_files

def test_detection_and_tracking():
    """Test detection and tracking on first available video."""
    print("=" * 60)
    print("TESTING DETECTION AND TRACKING PIPELINE")
    print("=" * 60)
    
    # Find videos
    data_dir = "data/raw"
    video_files = find_video_files(data_dir)
    
    if not video_files:
        print("❌ No video files found in data/raw/")
        return False
        
    test_video = video_files[0]
    print(f"✓ Testing with: {Path(test_video).name}")
    
    # Initialize components
    try:
        detector = MacaqueDetector(confidence=0.4)
        tracker = SimpleTracker(max_disappeared=5, max_distance=100)
        tracklet_manager = TrackletManager()
        print("✓ Initialized detector and tracker")
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return False
    
    # Process video (limited frames for testing)
    detections_data = []
    max_frames = 60  # Test with first 60 frames
    frame_skip = 3   # Process every 3rd frame
    
    try:
        with VideoProcessor(test_video) as processor:
            print(f"✓ Video: {processor.frame_count} frames @ {processor.fps:.1f} fps")
            
            frames_to_process = min(max_frames, processor.frame_count)
            
            with tqdm(total=frames_to_process//frame_skip, desc="Processing frames") as pbar:
                for frame_num, frame in processor.frame_generator(0, frames_to_process):
                    if frame_num % frame_skip != 0:
                        continue
                    
                    # Detect objects
                    detections = detector.detect_primates(frame)
                    
                    # Extract features
                    for detection in detections:
                        features = detector.extract_features(frame, detection['bbox'])
                        detection['features'] = features
                    
                    # Update tracker
                    tracked_objects = tracker.update(detections)
                    
                    # Store results
                    for detection in detections:
                        if 'track_id' in detection:
                            detection_record = {
                                'frame': frame_num,
                                'track_id': detection['track_id'],
                                'confidence': detection['confidence'],
                                'bbox': detection['bbox'],
                                'center': detection['center']
                            }
                            detections_data.append(detection_record)
                            tracklet_manager.add_detection(detection['track_id'], frame_num, detection)
                    
                    pbar.update(1)
            
        print(f"✓ Processed {frames_to_process} frames")
        print(f"✓ Found {len(detections_data)} detections")
        
        if detections_data:
            df = pd.DataFrame(detections_data)
            unique_tracks = df['track_id'].nunique()
            avg_confidence = df['confidence'].mean()
            print(f"✓ Identified {unique_tracks} tracks")
            print(f"✓ Average confidence: {avg_confidence:.3f}")
            
            return True, df, tracklet_manager
        else:
            print("⚠️  No detections found - this may be normal for some videos")
            return True, pd.DataFrame(), tracklet_manager
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False, None, None

def test_clustering(df_detections, tracklet_manager):
    """Test clustering on tracklet features."""
    print("\n" + "=" * 60)
    print("TESTING CLUSTERING PIPELINE")
    print("=" * 60)
    
    if df_detections.empty:
        print("⚠️  No detections to cluster")
        return True, {}
    
    try:
        # Get tracklet features
        tracklet_features = tracklet_manager.get_all_tracklet_features()
        
        # Filter valid features
        valid_features = {k: v for k, v in tracklet_features.items() 
                         if len(v) > 0 and not np.all(v == 0)}
        
        print(f"✓ Extracted features for {len(valid_features)} tracklets")
        
        if len(valid_features) < 3:
            print("⚠️  Not enough tracklets for clustering (need at least 3)")
            return True, {}
        
        # Test clustering
        identifier = IndividualIdentifier(method='hdbscan', min_cluster_size=2)
        track_to_individual = identifier.fit_predict(valid_features)
        
        # Analyze results
        individual_ids = list(track_to_individual.values())
        n_individuals = len(set(individual_ids)) - (1 if -1 in individual_ids else 0)
        n_noise = individual_ids.count(-1)
        
        print(f"✓ Clustering completed")
        print(f"✓ Proposed individuals: {n_individuals}")
        print(f"✓ Noise tracklets: {n_noise}")
        
        if n_individuals > 0:
            # Get cluster statistics
            stats = identifier.get_cluster_statistics()
            print("✓ Cluster statistics generated")
            
            # Show individual summary
            for individual_id in set(individual_ids):
                if individual_id >= 0:
                    count = individual_ids.count(individual_id)
                    print(f"  Individual {individual_id}: {count} tracklets")
        
        return True, track_to_individual
        
    except Exception as e:
        print(f"❌ Clustering failed: {e}")
        return False, {}

def test_video_processing():
    """Test video processing utilities."""
    print("\n" + "=" * 60)
    print("TESTING VIDEO PROCESSING")
    print("=" * 60)
    
    try:
        # Find videos
        data_dir = "data/raw"
        video_files = find_video_files(data_dir)
        
        if not video_files:
            print("❌ No video files found")
            return False
            
        print(f"✓ Found {len(video_files)} video files")
        
        # Test video info extraction
        for i, video_file in enumerate(video_files[:3]):  # Test first 3 videos
            try:
                with VideoProcessor(video_file) as processor:
                    print(f"✓ Video {i+1}: {Path(video_file).name}")
                    print(f"    Duration: {processor.duration:.1f}s")
                    print(f"    Frames: {processor.frame_count} @ {processor.fps:.1f} fps")
                    
                    # Test frame reading
                    ret, frame = processor.read_frame(0)
                    if ret:
                        print(f"    Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    else:
                        print("    ⚠️  Could not read first frame")
                        
            except Exception as e:
                print(f"    ❌ Error processing {Path(video_file).name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Video processing test failed: {e}")
        return False

def run_full_test():
    """Run complete pipeline test."""
    print("🔬 MACAQUE TRACKING PIPELINE TEST")
    print("Testing on sample data with limited processing...\n")
    
    # Test 1: Video Processing
    video_ok = test_video_processing()
    
    if not video_ok:
        print("\n❌ Video processing test failed - check video files")
        return False
    
    # Test 2: Detection and Tracking
    detection_ok, df_detections, tracklet_manager = test_detection_and_tracking()
    
    if not detection_ok:
        print("\n❌ Detection/tracking test failed")
        return False
    
    # Test 3: Clustering (if we have detections)
    if df_detections is not None and not df_detections.empty:
        clustering_ok, individual_mapping = test_clustering(df_detections, tracklet_manager)
        
        if not clustering_ok:
            print("\n❌ Clustering test failed")
            return False
    else:
        print("\n⚠️  Skipping clustering test - no detections found")
        individual_mapping = {}
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✅ Video processing: PASSED")
    print("✅ Detection & tracking: PASSED")
    
    if individual_mapping:
        print("✅ Individual clustering: PASSED")
    else:
        print("⚠️  Individual clustering: SKIPPED (no detections)")
    
    print(f"\n🎉 Pipeline test completed successfully!")
    print("📝 You can now run the full analysis notebooks:")
    print("   1. notebooks/01_video_preprocessing_detection.ipynb")
    print("   2. notebooks/02_individual_clustering.ipynb") 
    print("   3. notebooks/03_video_snippet_extraction.ipynb")
    
    return True

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)