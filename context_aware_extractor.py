import cv2
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
import argparse
import subprocess
import whisper

class SegmentAwareExtractor:
    def __init__(self, output_dir: str = "processed_videos", log_level: str = "INFO"):
        """
        Initialize Segment-Aware Frame Extractor
        
        Args:
            output_dir: Base directory for processed videos
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'segment_extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Whisper model settings
        self.whisper_model = "base"
        self.whisper_model_instance = None
        
        # Frame extraction settings
        self.quality = 95

    def _load_whisper_model(self):
        """Load Whisper model if not already loaded"""
        if self.whisper_model_instance is None:
            self.logger.info(f"Loading Whisper model: {self.whisper_model}")
            self.whisper_model_instance = whisper.load_model(self.whisper_model)
            self.logger.info("Whisper model loaded successfully")

    def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video file using FFmpeg
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Path to extracted audio file
        """
        video_path = Path(video_path)
        video_id = video_path.stem
        audio_output_dir = self.output_dir / video_id
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_output_dir / "audio.wav"
        
        self.logger.info(f"Extracting audio from: {video_path.name}")
        
        # FFmpeg command for audio extraction
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-ac', '1',  # Mono audio
            '-ar', '16000',  # 16kHz sample rate
            '-acodec', 'pcm_s16le',
            '-y',
            str(audio_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"Audio extracted successfully: {audio_path}")
            return str(audio_path)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to extract audio: {e.stderr}")

    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional)
            
        Returns:
            Dictionary with transcription results
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.logger.info(f"Transcribing audio: {audio_path.name}")
        
        # Load Whisper model
        self._load_whisper_model()
        
        try:
            transcribe_options = {
                'fp16': False,
                'verbose': False
            }
            
            if language:
                transcribe_options['language'] = language
            
            result = self.whisper_model_instance.transcribe(str(audio_path), **transcribe_options)
            
            # Process transcription results
            transcription_data = {
                'text': result['text'],
                'language': result.get('language', 'unknown'),
                'segments': [],
                'words': [],
                'duration': result.get('duration', 0),
                'audio_path': str(audio_path),
                'timestamp': datetime.now().isoformat()
            }
            
            # Process segments with timestamps
            if 'segments' in result:
                for segment in result['segments']:
                    segment_data = {
                        'id': segment.get('id', 0),
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'text': segment.get('text', '').strip(),
                        'tokens': segment.get('tokens', []),
                        'temperature': segment.get('temperature', 0),
                        'avg_logprob': segment.get('avg_logprob', 0),
                        'compression_ratio': segment.get('compression_ratio', 0),
                        'no_speech_prob': segment.get('no_speech_prob', 0)
                    }
                    transcription_data['segments'].append(segment_data)
            
            # Process word-level timestamps if available
            if 'words' in result:
                for word in result['words']:
                    word_data = {
                        'start': word.get('start', 0),
                        'end': word.get('end', 0),
                        'word': word.get('word', '').strip(),
                        'probability': word.get('probability', 0)
                    }
                    transcription_data['words'].append(word_data)
            
            self.logger.info(f"Transcription completed. Duration: {transcription_data['duration']:.2f}s")
            self.logger.info(f"Segments: {len(transcription_data['segments'])}")
            
            return transcription_data
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription failed: {str(e)}")

    def extract_frame_at_timestamp(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """
        Extract a frame from video at specific timestamp
        
        Args:
            video_path: Path to video file
            timestamp: Timestamp in seconds
            
        Returns:
            Frame as numpy array or None if failed
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return None
        
        # Set position to timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        else:
            self.logger.warning(f"Failed to extract frame at {timestamp:.2f}s")
            return None

    def extract_segment_aware_frames(self, video_path: str, transcription_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract frames based on transcription segments
        
        Args:
            video_path: Path to video file
            transcription_data: Transcription results from Whisper
            
        Returns:
            Dictionary with extraction results
        """
        video_path = Path(video_path)
        video_id = video_path.stem
        frames_output_dir = self.output_dir / video_id / "segmented_frames"
        frames_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Extracting segment-aware frames for: {video_id}")
        
        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        self.logger.info(f"Video properties: {width}x{height}, {fps:.2f} FPS, {duration:.2f}s")
        
        # Process segments
        segments = transcription_data.get('segments', [])
        extracted_frames = []
        
        start_time = datetime.now()
        
        for i, segment in enumerate(segments):
            segment_start = segment['start']
            segment_end = segment['end']
            segment_text = segment['text']
            
            # Skip very short segments
            if segment_end - segment_start < 0.5:
                continue
            
            # Calculate middle timestamp of the segment
            middle_timestamp = (segment_start + segment_end) / 2
            
            # Extract frame at middle timestamp
            frame = self.extract_frame_at_timestamp(video_path, middle_timestamp)
            if frame is None:
                continue
            
            # Save frame
            frame_filename = f"segment_{i:04d}_{middle_timestamp:.2f}s.jpg"
            frame_path = frames_output_dir / frame_filename
            
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            
            # Store frame metadata
            frame_info = {
                'segment_id': i,
                'segment_start': segment_start,
                'segment_end': segment_end,
                'segment_text': segment_text,
                'frame_timestamp': middle_timestamp,
                'frame_filename': frame_filename,
                'segment_duration': segment_end - segment_start
            }
            
            extracted_frames.append(frame_info)
            
            # Progress logging
            if (i + 1) % 50 == 0:
                progress = ((i + 1) / len(segments)) * 100
                self.logger.info(f"Progress: {progress:.1f}% - Processed {i + 1}/{len(segments)} segments")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
                # Create summary
        summary = {
            'video_id': video_id,
            'video_path': str(video_path),
            'video_properties': {
                'width': width,
                'height': height,
                'fps': fps,
                'duration': duration,
                'total_frames': total_frames
            },
            'extraction_settings': {
                'quality': self.quality,
                'extraction_method': 'segment_middle_frame'
            },
            'results': {
                'segments_processed': len(extracted_frames),
                'total_segments': len(segments),
                'processing_time': processing_time,
                'frames_per_second': len(extracted_frames) / processing_time if processing_time > 0 else 0
            },
            'frame_metadata': extracted_frames,
            'transcription_summary': {
                'duration': transcription_data['duration'],
                'language': transcription_data['language'],
                'segments_count': len(transcription_data['segments']),
                'words_count': len(transcription_data.get('words', []))
            },
            'timestamp': start_time.isoformat()
        }
        
        # Save metadata
        metadata_path = frames_output_dir.parent / "segment_frame_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Segment-aware extraction completed: {len(extracted_frames)} frames")
        self.logger.info(f"Processing time: {processing_time:.2f}s")
        
        return summary

    def process_video_segments(self, video_path: str, language: str = None) -> Dict[str, Any]:
        """
        Complete segment-aware processing pipeline for a video
        
        Args:
            video_path: Path to input video file
            language: Language code for transcription
            
        Returns:
            Dictionary with processing results
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        video_id = video_path.stem
        self.logger.info(f"Processing video with segment-aware extraction: {video_id}")
        
        start_time = datetime.now()
        
        try:
            # Extract audio
            audio_path = self.extract_audio(video_path)
            
            # Transcribe audio
            transcription_data = self.transcribe_audio(audio_path, language)
            
            # Extract segment-aware frames
            frame_extraction_result = self.extract_segment_aware_frames(video_path, transcription_data)
            
            # Save transcription data
            transcription_output_path = self.output_dir / video_id / "transcript.json"
            with open(transcription_output_path, 'w') as f:
                json.dump(transcription_data, f, indent=2)
            
            # Calculate total processing time
            end_time = datetime.now()
            total_processing_time = (end_time - start_time).total_seconds()
            
            # Create final summary
            final_summary = {
                'video_id': video_id,
                'video_path': str(video_path),
                'audio_path': audio_path,
                'transcription_path': str(transcription_output_path),
                'frame_extraction_result': frame_extraction_result,
                'total_processing_time': total_processing_time,
                'status': 'completed',
                'timestamp': start_time.isoformat()
            }
            
            self.logger.info(f"Segment-aware processing completed for {video_id}")
            self.logger.info(f"Total processing time: {total_processing_time:.2f}s")
            
            return final_summary
            
        except Exception as e:
            self.logger.error(f"Segment-aware processing failed for {video_id}: {str(e)}")
            return {
                'video_id': video_id,
                'video_path': str(video_path),
                'status': 'failed',
                'error': str(e),
                'timestamp': start_time.isoformat()
            }

    def batch_process_segments(self, video_dir: str, language: str = None) -> List[Dict[str, Any]]:
        """
        Process multiple videos with segment-aware extraction
        
        Args:
            video_dir: Directory containing video files
            language: Language code for transcription
            
        Returns:
            List of processing results
        """
        video_dir = Path(video_dir)
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        
        # Find video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        # video_files = [f for f in video_dir.iterdir() 
        #               if f.suffix.lower() in video_extensions]

        video_files = []
        for video_file in video_dir.rglob('*'):
            if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                video_files.append(video_file)
        
        if not video_files:
            self.logger.warning(f"No video files found in {video_dir}")
            return []
        
        self.logger.info(f"Found {len(video_files)} video files for segment-aware processing")
        
        results = []
        for video_file in video_files:
            try:
                self.logger.info(f"Processing: {video_file.name}")
                result = self.process_video_segments(str(video_file), language)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {video_file.name}: {str(e)}")
                continue
        
        return results


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Segment-Aware Frame Extractor')
    parser.add_argument('input', help='Video file or directory path')
    parser.add_argument('-o', '--output', default='processed_videos', 
                       help='Output directory')
    parser.add_argument('-l', '--language', help='Language code for transcription')
    parser.add_argument('--model', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size')
    parser.add_argument('-q', '--quality', type=int, default=95, 
                       help='JPEG quality (1-100)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = SegmentAwareExtractor(args.output, args.log_level)
    extractor.whisper_model = args.model
    extractor.quality = args.quality
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single video processing
        try:
            result = extractor.process_video_segments(str(input_path), args.language)
            print(f"Segment-aware processing completed for {input_path.name}")
            if result['status'] == 'completed':
                frame_result = result['frame_extraction_result']
                print(f"Segments processed: {frame_result['results']['segments_processed']}")
                print(f"Processing time: {result['total_processing_time']:.2f}s")
            else:
                print(f"Status: {result['status']}")
                if 'error' in result:
                    print(f"Error: {result['error']}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    elif input_path.is_dir():
        # Batch processing
        try:
            results = extractor.batch_process_segments(str(input_path), args.language)
            print(f"Batch processing completed: {len(results)} videos processed")
            successful = sum(1 for r in results if r['status'] == 'completed')
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()