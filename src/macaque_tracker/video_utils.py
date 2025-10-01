import cv2
import numpy as np
import os
from collections.abc import Generator
import moviepy
from moviepy import VideoFileClip, concatenate_videoclips
import pandas as pd
from pathlib import Path


class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
        
    def read_frame(self, frame_number: int = None) -> tuple[bool, np.ndarray]:
        if frame_number is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return self.cap.read()
    
    def frame_generator(self, start_frame: int = 0, end_frame: int = None) -> Generator[tuple[int, np.ndarray], None, None]:
        if end_frame is None:
            end_frame = self.frame_count
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_num in range(start_frame, min(end_frame, self.frame_count)):
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_num, frame
            
    def extract_frames_around_detections(self, detections_df: pd.DataFrame, 
                                       buffer_seconds: float = 2.0) -> dict[int, list[tuple[int, int]]]:
        buffer_frames = int(buffer_seconds * self.fps)
        
        # Group by track_id and get frame ranges
        track_segments = {}
        
        for track_id in detections_df['track_id'].unique():
            track_detections = detections_df[detections_df['track_id'] == track_id]
            
            segments = []
            frames = sorted(track_detections['frame'].tolist())
            
            if not frames:
                continue
                
            # Create segments from consecutive frame sequences
            current_start = frames[0]
            current_end = frames[0]
            
            for frame in frames[1:]:
                if frame <= current_end + buffer_frames * 2:  # Merge nearby segments
                    current_end = frame
                else:
                    # Save current segment and start new one
                    segments.append((
                        max(0, current_start - buffer_frames),
                        min(self.frame_count - 1, current_end + buffer_frames)
                    ))
                    current_start = frame
                    current_end = frame
                    
            # Add final segment
            segments.append((
                max(0, current_start - buffer_frames),
                min(self.frame_count - 1, current_end + buffer_frames)
            ))
            
            track_segments[track_id] = segments
            
        return track_segments


class VideoClipExtractor:
    def __init__(self, output_dir: str = "output/clips"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_clips_by_individual(self, video_path: str, 
                                   detections_df: pd.DataFrame,
                                   individual_mapping: dict[int, int],
                                   buffer_seconds: float = 2.0) -> dict[int, list[str]]:
        
        # Add individual IDs to detections
        detections_df = detections_df.copy()
        detections_df['individual_id'] = detections_df['track_id'].map(individual_mapping)
        
        # Remove unmapped tracklets
        detections_df = detections_df.dropna(subset=['individual_id'])
        detections_df['individual_id'] = detections_df['individual_id'].astype(int)
        
        video_name = Path(video_path).stem
        clips_by_individual = {}
        
        with VideoProcessor(video_path) as processor:
            for individual_id in detections_df['individual_id'].unique():
                if individual_id < 0:  # Skip noise
                    continue
                    
                individual_detections = detections_df[detections_df['individual_id'] == individual_id]
                track_segments = processor.extract_frames_around_detections(individual_detections, buffer_seconds)
                
                individual_clips = []
                clip_counter = 0
                
                for track_id, segments in track_segments.items():
                    for start_frame, end_frame in segments:
                        clip_path = self.output_dir / f"{video_name}_individual_{individual_id}_clip_{clip_counter}.mp4"
                        
                        # Extract clip using moviepy
                        start_time = start_frame / processor.fps
                        end_time = end_frame / processor.fps
                        
                        try:
                            with VideoFileClip(video_path) as clip:
                                subclip = clip.subclip(start_time, end_time)
                                subclip.write_videofile(str(clip_path), verbose=False, logger=None)
                                
                            individual_clips.append(str(clip_path))
                            clip_counter += 1
                            
                        except Exception as e:
                            print(f"Error extracting clip for individual {individual_id}: {e}")
                            continue
                            
                clips_by_individual[individual_id] = individual_clips
                
        return clips_by_individual
    
    def create_summary_video(self, clips_by_individual: dict[int, list[str]], 
                            output_path: str = None) -> str:
        if output_path is None:
            output_path = str(self.output_dir / "individual_summary.mp4")
            
        all_clips = []
        
        for individual_id, clip_paths in clips_by_individual.items():
            # Take first few clips per individual to keep summary manageable
            selected_clips = clip_paths[:3]  
            
            for clip_path in selected_clips:
                try:
                    clip = VideoFileClip(clip_path)
                    # Add text overlay
                    txt_clip = (clip.set_duration(clip.duration)
                              .margin(left=8, right=8, top=8, bottom=8, color=(0,0,0))
                              .set_fps(24))
                    all_clips.append(txt_clip)
                except Exception as e:
                    print(f"Error processing clip {clip_path}: {e}")
                    continue
                    
        if all_clips:
            try:
                final_video = concatenate_videoclips(all_clips, method="compose")
                final_video.write_videofile(output_path, verbose=False, logger=None)
                return output_path
            except Exception as e:
                print(f"Error creating summary video: {e}")
                return None
        else:
            print("No clips available for summary video")
            return None


def find_video_files(directory: str) -> list[str]:
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
                
    return sorted(video_files)


def process_videos_in_directory(directory: str, output_dir: str = "output") -> list[str]:
    video_files = find_video_files(directory)
    processed_files = []
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for video_file in video_files:
        try:
            # Basic processing check
            with VideoProcessor(video_file) as processor:
                if processor.frame_count > 0:
                    processed_files.append(video_file)
                    print(f"✓ {video_file}: {processor.frame_count} frames, {processor.duration:.1f}s")
                else:
                    print(f"✗ {video_file}: Invalid or empty video")
        except Exception as e:
            print(f"✗ {video_file}: Error - {e}")
            
    return processed_files