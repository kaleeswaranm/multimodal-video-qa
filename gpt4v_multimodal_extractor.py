import openai
import cv2
import json
import base64
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
import argparse
from PIL import Image
import io
import re
import time

class GPT4VMultimodalExtractor:
    def __init__(self, output_dir: str = "processed_videos", log_level: str = "INFO", 
             api_key: str = None, model: str = "gpt-4o-mini"):
        """
        Initialize GPT-4V Multimodal Feature Extractor
        
        Args:
            output_dir: Base directory for processed videos
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            api_key: OpenAI API key (if None, will try to get from environment)
            model: GPT-4V model name
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'gpt4v_extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # API settings
        self.model = model
        self.api_key = api_key or self._get_api_key()
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Feature extraction settings
        self.max_description_length = 500
        self.batch_size = 5  # Process frames in batches
        self.request_delay = 1.0  # Delay between API requests (seconds)
        
        # Statistics
        self.total_requests = 0
        self.total_cost = 0.0

    def _get_api_key(self) -> str:
        """Get OpenAI API key from environment or user input"""
        import os
        
        # Try to get from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        # Prompt user for API key
        print("OpenAI API key not found in environment variables.")
        api_key = input("Please enter your OpenAI API key: ").strip()
        
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        return api_key

    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image to base64 for API transmission
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to encode image {image_path}: {str(e)}")
            return ""

    def generate_comprehensive_description(self, image_path: str, transcript_text: str = "") -> Dict[str, str]:
        """
        Generate comprehensive image description using GPT-4V (includes OCR)
        
        Args:
            image_path: Path to image file
            transcript_text: Transcript text for context
            
        Returns:
            Dictionary with description and OCR text
        """
        try:
            # Encode image
            base64_image = self.encode_image_to_base64(image_path)
            if not base64_image:
                return {"description": "", "ocr_text": ""}
            
            # Prepare comprehensive prompt
            if transcript_text:
                prompt = f"""TASK: Multimodal retrieval for educational video frames.

You are given a video frame and related spoken content. Produce a concise, highly-informative, retrieval-optimized description with strong cross-modal alignment.

Provide the output in EXACTLY these sections:
VISUAL DESCRIPTION: Key objects, diagrams, axes/labels, colors, spatial layout, notable regions; use precise nouns.
TEXT EXTRACTION: All visible text EXACTLY as shown (equations, symbols, labels, captions, titles, axis ticks).
CONTEXT INTEGRATION: How the visuals relate to the spoken content below (align concepts, claims, variables).

SPOKEN CONTENT:
{transcript_text}
"""
            else:
                prompt = """TASK: Multimodal retrieval for educational video frames.

You are given a video frame. Produce a concise, highly-informative, retrieval-optimized description with strong cross-modal alignment.

Provide the output in EXACTLY these sections:
VISUAL DESCRIPTION: Key objects, diagrams, axes/labels, colors, spatial layout, notable regions; use precise nouns.
TEXT EXTRACTION: All visible text EXACTLY as shown (equations, symbols, labels, captions, titles, axis ticks).
CONTEXT INTEGRATION: Relation between the visuals and likely educational concepts (if inferable).
"""
            
            # Make API request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )

            # Extract response
            response_text = response.choices[0].message.content.strip()

            # Parse response
            description = ""
            ocr_text = ""
            context_integration = ""
            
            # Split response into sections
            sections = response_text.split('\n')
            current_section = ""
            
            for line in sections:
                line = line.strip()
                
                # Handle both formats: "VISUAL DESCRIPTION:" and "**VISUAL DESCRIPTION:**"
                if line.startswith("VISUAL DESCRIPTION:") or line.startswith("**VISUAL DESCRIPTION:**"):
                    current_section = "description"
                    description = line.replace("VISUAL DESCRIPTION:", "").replace("**VISUAL DESCRIPTION:**", "").strip()
                elif line.startswith("TEXT EXTRACTION:") or line.startswith("**TEXT EXTRACTION:**"):
                    current_section = "ocr"
                    ocr_text = line.replace("TEXT EXTRACTION:", "").replace("**TEXT EXTRACTION:**", "").strip()
                elif line.startswith("CONTEXT INTEGRATION:") or line.startswith("**CONTEXT INTEGRATION:**"):
                    current_section = "context"
                    context_integration = line.replace("CONTEXT INTEGRATION:", "").replace("**CONTEXT INTEGRATION:**", "").strip()
                elif line and current_section:
                    # Continue current section
                    if current_section == "description":
                        description += " " + line
                    elif current_section == "ocr":
                        ocr_text += " " + line
                    elif current_section == "context":
                        context_integration += " " + line
            
            # Clean up text
            description = re.sub(r'\s+', ' ', description).strip()
            ocr_text = re.sub(r'\s+', ' ', ocr_text).strip()
            context_integration = re.sub(r'\s+', ' ', context_integration).strip()
            
            # Update statistics
            self.total_requests += 1
            # Estimate cost (rough calculation)
            input_tokens = len(prompt.split()) + 85  # Base tokens for image
            output_tokens = len(response_text.split())
            cost = (input_tokens * 0.00001) + (output_tokens * 0.00003)  # Rough estimate
            self.total_cost += cost
            
            return {
                "description": description,
                "ocr_text": ocr_text,
                "context_integration": context_integration
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate description for {image_path}: {str(e)}")
            return {"description": "", "ocr_text": "", "context_integration": ""}

    def extract_multimodal_features(self, image_path: str, transcript_text: str) -> Dict[str, Any]:
        """
        Extract multimodal features from image and text using GPT-4V
        
        Args:
            image_path: Path to image file
            transcript_text: Transcript text for the segment
            
        Returns:
            Dictionary with extracted features
        """
        try:
            # Generate comprehensive description using GPT-4V
            gpt4v_response = self.generate_comprehensive_description(image_path, transcript_text)

            # Extract components
            image_description = gpt4v_response["description"]
            ocr_text = gpt4v_response["ocr_text"]
            context_integration = gpt4v_response["context_integration"]
            
            # Combine all text
            combined_text = f"{image_description} {ocr_text} {context_integration} {transcript_text}".strip()
            
            # Clean and format text
            combined_text = re.sub(r'\s+', ' ', combined_text)  # Remove extra whitespace
            
            features = {
                'image_path': str(image_path),
                'image_description': image_description,
                'ocr_text': ocr_text,
                'context_integration': context_integration,
                'transcript_text': transcript_text,
                'combined_text': combined_text,
                'text_length': len(combined_text),
                'has_ocr_text': len(ocr_text) > 0,
                'description_length': len(image_description),
                'ocr_length': len(ocr_text),
                'transcript_length': len(transcript_text),
                'api_requests': self.total_requests,
                'estimated_cost': self.total_cost,
                'timestamp': datetime.now().isoformat()
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract features for {image_path}: {str(e)}")
            return {
                'image_path': str(image_path),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def process_video_segments(self, video_id: str, segments_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process all segments for a video
        
        Args:
            video_id: Video identifier
            segments_data: List of segment data with frame paths and transcripts
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing GPT-4V multimodal features for video: {video_id}")
        
        # Create output directory
        features_output_dir = self.output_dir / video_id / "gpt4v_features"
        features_output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = datetime.now()
        processed_segments = []
        failed_segments = []
        
        for i, segment in enumerate(segments_data):
            try:
                self.logger.info(f"Processing segment {i+1}/{len(segments_data)}")
                
                # Extract features
                features = self.extract_multimodal_features(
                    segment['frame_path'],
                    segment['transcript_text']
                )
                
                # Add segment metadata
                features.update({
                    'segment_id': segment.get('segment_id', i),
                    'timestamp_start': segment.get('timestamp_start', 0),
                    'timestamp_end': segment.get('timestamp_end', 0),
                    'frame_filename': Path(segment['frame_path']).name
                })
                
                processed_segments.append(features)
                
                # Add delay between requests to avoid rate limiting
                if i < len(segments_data) - 1:  # Don't delay after last request
                    time.sleep(self.request_delay)
                
                # Progress logging
                if (i + 1) % 5 == 0:
                    progress = ((i + 1) / len(segments_data)) * 100
                    self.logger.info(f"Progress: {progress:.1f}% - Processed {i + 1}/{len(segments_data)} segments")
                    self.logger.info(f"Total API requests: {self.total_requests}, Estimated cost: ${self.total_cost:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to process segment {i}: {str(e)}")
                failed_segments.append({
                    'segment_id': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Create summary
        summary = {
            'video_id': video_id,
            'processing_time': processing_time,
            'total_segments': len(segments_data),
            'processed_segments': len(processed_segments),
            'failed_segments': len(failed_segments),
            'segments_per_second': len(processed_segments) / processing_time if processing_time > 0 else 0,
            'total_api_requests': self.total_requests,
            'total_estimated_cost': self.total_cost,
            'features': processed_segments,
            'errors': failed_segments,
            'timestamp': start_time.isoformat()
        }
        
        # Save features
        features_file = features_output_dir / "gpt4v_features.json"
        with open(features_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save processed text for embedding extraction
        text_file = features_output_dir / "processed_text.json"
        processed_text = []
        for segment in processed_segments:
            processed_text.append({
                'segment_id': segment['segment_id'],
                'combined_text': segment['combined_text'],
                'image_description': segment['image_description'],
                'ocr_text': segment['ocr_text'],
                'context_integration': segment['context_integration'],
                'transcript_text': segment['transcript_text'],
                'frame_filename': segment['frame_filename'],
                'timestamp_start': segment['timestamp_start'],
                'timestamp_end': segment['timestamp_end']
            })
        
        with open(text_file, 'w') as f:
            json.dump(processed_text, f, indent=2)
        
        self.logger.info(f"GPT-4V multimodal feature extraction completed for {video_id}")
        self.logger.info(f"Processed: {len(processed_segments)} segments")
        self.logger.info(f"Failed: {len(failed_segments)} segments")
        self.logger.info(f"Processing time: {processing_time:.2f}s")
        self.logger.info(f"Total API requests: {self.total_requests}")
        self.logger.info(f"Estimated total cost: ${self.total_cost:.4f}")
        
        return summary

    def batch_process_videos(self, video_dir: str) -> List[Dict[str, Any]]:
        """
        Process multiple videos with GPT-4V multimodal feature extraction
        
        Args:
            video_dir: Directory containing processed videos
            
        Returns:
            List of processing results
        """
        video_dir = Path(video_dir)
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        
        # Find processed videos
        video_ids = []
        for video_path in video_dir.iterdir():
            if video_path.is_dir() and (video_path / "segment_frame_metadata.json").exists():
                if (video_path / "gpt4v_features" / "processed_text.json").exists():
                    self.logger.info(f"Skipping video {video_path.name} because it already has GPT-4V features")
                    continue
                video_ids.append(video_path.name)
        
        if not video_ids:
            self.logger.warning(f"No processed videos found in {video_dir}")
            return []
        
        self.logger.info(f"Found {len(video_ids)} videos for GPT-4V multimodal processing")
        
        results = []
        for video_id in video_ids:
            try:
                # Load segment data
                segment_file = video_dir / video_id / "segment_frame_metadata.json"
                with open(segment_file, 'r') as f:
                    segment_data = json.load(f)
                
                # Extract segment information
                segments = []
                for frame_info in segment_data.get('frame_metadata', []):
                    segments.append({
                        'segment_id': frame_info['segment_id'],
                        'frame_path': str(video_dir / video_id / "segmented_frames" / frame_info['frame_filename']),
                        'transcript_text': frame_info['segment_text'],
                        'timestamp_start': frame_info['segment_start'],
                        'timestamp_end': frame_info['segment_end']
                    })
                
                # Process segments
                result = self.process_video_segments(video_id, segments)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process video {video_id}: {str(e)}")
                continue
        
        return results

    def get_processing_summary(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get processing summary for a specific video
        
        Args:
            video_id: Video identifier
            
        Returns:
            Processing summary or None if not found
        """
        features_file = self.output_dir / video_id / "gpt4v_features" / "gpt4v_features.json"
        if features_file.exists():
            with open(features_file, 'r') as f:
                return json.load(f)
        return None


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='GPT-4V Multimodal Feature Extractor')
    parser.add_argument('input', help='Video directory path')
    parser.add_argument('-o', '--output', default='processed_videos', 
                       help='Output directory')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model', default='gpt-4o-mini', help='GPT-4V model name')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between API requests (seconds)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = GPT4VMultimodalExtractor(
        args.output, 
        args.log_level, 
        args.api_key,
        args.model
    )
    
    # Set request delay
    extractor.request_delay = args.delay
    
    # Process videos
    try:
        results = extractor.batch_process_videos(args.input)
        print(f"GPT-4V multimodal processing completed: {len(results)} videos processed")
        
        total_segments = sum(r['total_segments'] for r in results)
        processed_segments = sum(r['processed_segments'] for r in results)
        failed_segments = sum(r['failed_segments'] for r in results)
        total_cost = sum(r['total_estimated_cost'] for r in results)
        
        print(f"Total segments: {total_segments}")
        print(f"Processed: {processed_segments}")
        print(f"Failed: {failed_segments}")
        print(f"Total estimated cost: ${total_cost:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()