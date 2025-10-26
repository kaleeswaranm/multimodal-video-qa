import openai
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
import argparse
import time
import chromadb
from chromadb.config import Settings
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

class SlidingWindowEmbeddingExtractor:
    def __init__(self, output_dir: str = "processed_videos", log_level: str = "INFO", 
                 api_key: str = None, model_name: str = "intfloat/mmE5-mllama-11b-instruct",
                 window_size: int = 5, hop_size: int = 3, vector_db_path: str = "vector_db"):
        """
        Initialize Sliding Window Embedding Extractor
        
        Args:
            output_dir: Base directory for processed videos
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            api_key: OpenAI API key for text summarization
            model_name: Multimodal embedding model name
            window_size: Number of segments per window
            hop_size: Number of segments to advance between windows
            vector_db_path: Path to ChromaDB database
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'sliding_window_extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Window parameters
        self.window_size = window_size
        self.hop_size = hop_size
        self.middle_index = window_size // 2  # Index of middle segment
        
        # Initialize OpenAI client for summarization
        self.api_key = api_key or self._get_api_key()
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Initialize multimodal embedding model using Transformers
        self.logger.info(f"Loading multimodal embedding model: {model_name}")
        self._initialize_embedding_model(model_name)
        
        # Initialize ChromaDB
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Initializing ChromaDB at: {self.vector_db_path}")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vector_db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection_name = "video_segments"
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            self.logger.info(f"Found existing collection: {self.collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Video segment embeddings with sliding window approach",
                          "hnsw:space": "cosine"}
            )
            self.logger.info(f"Created new collection: {self.collection_name}")
        
        # Statistics
        self.total_windows = 0
        self.total_summaries = 0
        self.total_embeddings = 0
        self.total_stored = 0

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

    def _initialize_embedding_model(self, model_name: str):
        """Initialize embedding model using Transformers with MPS dtype handling"""
        try:
            import torch

            # Detect device
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                dtype = torch.float16  # Supported on MPS
                self.logger.info("Using MPS backend (Apple Silicon)")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
                dtype = torch.bfloat16  # Supported on CUDA
                self.logger.info("Using CUDA backend")
            else:
                device = torch.device("cpu")
                dtype = torch.float32
                self.logger.info("Using CPU backend")

            # Load processor and model safely
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=None  # Avoid auto-mapping incompatible layers
            ).to(device)
            self.model.eval()

            self.device = device
            self.logger.info(f"Successfully loaded model: {model_name} on {device} ({dtype})")

        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise

    def last_pooling(self, last_hidden_state, attention_mask, normalize=True):
        """Pooling function from mmE5 example"""
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        if normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def encode_multimodal(self, image_path: str, text: str) -> np.ndarray:
        """
        Encode image and text using mmE5 model
        
        Args:
            image_path: Path to image file
            text: Text to encode with image
            
        Returns:
            Embedding vector
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Prepare input text with image token
            # input_text = f'<|image|><|begin_of_text|>{text}\n'
            input_text = f"<|image|><|begin_of_text|>Represent the given educational video frame with the following text for retrieval.\nTEXT:\n{text}\n"

            # Process inputs
            inputs = self.processor(
                text=input_text, 
                images=[image], 
                return_tensors="pt"
            )
            
            # Move to same device as model
            # device = next(self.model.parameters()).device
            # device = "cpu"
            device = self.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)
                embedding = self.last_pooling(
                    outputs.hidden_states[-1], 
                    inputs['attention_mask']
                )
            
            return embedding.cpu().numpy()[0]
            
        except Exception as e:
            self.logger.error(f"Failed to encode multimodal: {str(e)}")
            raise

    def summarize_window_text(self, segments_text: List[str]) -> str:
        """
        Summarize text from multiple segments using GPT
        
        Args:
            segments_text: List of text from segments in the window
            
        Returns:
            Summarized text
        """
        try:
            # Combine all segment texts
            combined_text = " ".join(segments_text)
            
            # Truncate if too long (to avoid token limits)
            # if len(combined_text) > 8000:
                # combined_text = combined_text[:8000] + "..."
            
            # Create summarization prompt
            prompt = f"""Summarize the following educational video content into a retrieval-optimized, structured document.

CONTENT:
{combined_text}

OUTPUT FORMAT (use these exact headings, one block each):
TOPIC: [Primary subject]
KEY_CONCEPTS: [3-7 comma-separated terms/phrases; include variables/symbols if present]
DEFINITIONS: [Short, precise definitions of key terms if present]
EQUATIONS: [Canonical forms; include variable names, units if mentioned]
PROCESS: [If procedural, list core steps succinctly]
TAKEAWAYS: [2-3 concise sentences with the most important points]

Guidelines:
- Prefer precise nouns, technical terms, symbols, and quantities over prose.
- Keep it concise and information-dense; avoid filler.
- Preserve math and symbols verbatim where possible.
"""
            
            # Make API request
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for summarization
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            self.total_summaries += 1
            # print(summary)
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to summarize text: {str(e)}")
            # Fallback: return first segment's text
            return segments_text[0] if segments_text else ""

    def extract_window_embedding(self, window_segments: List[Dict[str, Any]], video_id: str) -> Dict[str, Any]:
        """
        Extract embedding for a window of segments
        
        Args:
            window_segments: List of segment data for the window
            video_id: Video identifier for constructing frame paths
            
        Returns:
            Dictionary with embedding and metadata
        """
        try:
            # Get middle segment
            middle_segment = window_segments[self.middle_index]
            
            # Extract text from all segments in window
            segments_text = []
            for segment in window_segments:
                combined_text = segment.get('combined_text', '')
                if combined_text:
                    segments_text.append(combined_text)
            
            # Store combined text before summarization
            combined_text_before_summarization = " ".join(segments_text)

            # Summarize the window text
            summarized_text = self.summarize_window_text(segments_text)
            
            # Construct frame path from frame_filename
            frame_filename = middle_segment.get('frame_filename', '')
            if not frame_filename:
                raise ValueError("No frame_filename found in segment data")
            
            # Construct the full path to the frame
            frame_path = f"processed_videos/{video_id}/segmented_frames/{frame_filename}"

            # Extract multimodal embedding using mmE5
            embedding = self.encode_multimodal(
                frame_path, 
                summarized_text
            )
            
            # Create window metadata
            window_metadata = {
                'window_id': self.total_windows,
                'middle_segment_id': middle_segment['segment_id'],
                'middle_frame_path': frame_path,
                'window_segments': [s['segment_id'] for s in window_segments],
                'summarized_text': summarized_text,
                'combined_text_before_summarization': combined_text_before_summarization,  # NEW
                'embedding': embedding.tolist(),  # Convert to list for JSON serialization
                'embedding_dimension': len(embedding),
                'timestamp_start': window_segments[0]['timestamp_start'],
                'timestamp_end': window_segments[-1]['timestamp_end'],
                'window_size': self.window_size,
                'hop_size': self.hop_size
            }
            
            self.total_windows += 1
            self.total_embeddings += 1
            
            return window_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to extract window embedding: {str(e)}")
            return {
                'window_id': self.total_windows,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def store_embedding_in_vector_db(self, window_embedding: Dict[str, Any], video_id: str) -> bool:
        """
        Store embedding in ChromaDB vector database
        
        Args:
            window_embedding: Window embedding data
            video_id: Video identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if 'embedding' not in window_embedding or 'error' in window_embedding:
                return False
            
            # Create unique ID
            embedding_id = f"{video_id}_window_{window_embedding['window_id']}"
            
            # Prepare metadata for ChromaDB
            metadata = {
                'video_id': video_id,
                'window_id': window_embedding['window_id'],
                'middle_segment_id': window_embedding['middle_segment_id'],
                'middle_frame_path': window_embedding['middle_frame_path'],
                'summarized_text': window_embedding['summarized_text'],
                'combined_text_before_summarization': window_embedding.get('combined_text_before_summarization', ''),
                'timestamp_start': window_embedding['timestamp_start'],
                'timestamp_end': window_embedding['timestamp_end'],
                'window_segments': json.dumps(window_embedding['window_segments']),
                'window_size': window_embedding['window_size'],
                'hop_size': window_embedding['hop_size'],
                'embedding_dimension': window_embedding['embedding_dimension']
            }
            
            # Store in ChromaDB
            self.collection.add(
                ids=[embedding_id],
                embeddings=[window_embedding['embedding']],
                metadatas=[metadata],
                documents=[window_embedding['summarized_text']]  # Store summarized text as document
            )
            
            self.total_stored += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store embedding in vector DB: {str(e)}")
            return False

    def process_video_windows(self, video_id: str, segments_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process all windows for a video using sliding window approach
        
        Args:
            video_id: Video identifier
            segments_data: List of all segment data
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing sliding window embeddings for video: {video_id}")
        self.logger.info(f"Total segments: {len(segments_data)}")
        self.logger.info(f"Window size: {self.window_size}, Hop size: {self.hop_size}")
        
        # Create output directory
        embeddings_output_dir = self.output_dir / video_id / "sliding_window_embeddings"
        embeddings_output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = datetime.now()
        processed_windows = []
        failed_windows = []
        stored_embeddings = 0
        
        # Create sliding windows
        num_windows = (len(segments_data) - self.window_size) // self.hop_size + 1
        self.logger.info(f"Expected number of windows: {num_windows}")
        
        for window_idx in range(num_windows):
            try:
                # Calculate window boundaries
                start_idx = window_idx * self.hop_size
                end_idx = start_idx + self.window_size
                
                # Get window segments
                window_segments = segments_data[start_idx:end_idx]
                
                if len(window_segments) < self.window_size:
                    self.logger.warning(f"Window {window_idx} has only {len(window_segments)} segments, skipping")
                    continue
                
                self.logger.info(f"Processing window {window_idx + 1}/{num_windows} (segments {start_idx}-{end_idx-1})")
                
                # Extract embedding for this window
                window_embedding = self.extract_window_embedding(window_segments, video_id)
                processed_windows.append(window_embedding)
                
                # Store in vector database
                if self.store_embedding_in_vector_db(window_embedding, video_id):
                    stored_embeddings += 1
                
                # Progress logging
                if (window_idx + 1) % 5 == 0:
                    progress = ((window_idx + 1) / num_windows) * 100
                    self.logger.info(f"Progress: {progress:.1f}% - Processed {window_idx + 1}/{num_windows} windows")
                    self.logger.info(f"Total summaries: {self.total_summaries}, Total embeddings: {self.total_embeddings}")
                    self.logger.info(f"Stored in vector DB: {stored_embeddings}")
                
            except Exception as e:
                self.logger.error(f"Failed to process window {window_idx}: {str(e)}")
                failed_windows.append({
                    'window_idx': window_idx,
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
            'window_size': self.window_size,
            'hop_size': self.hop_size,
            'expected_windows': num_windows,
            'processed_windows': len(processed_windows),
            'failed_windows': len(failed_windows),
            'stored_embeddings': stored_embeddings,
            'windows_per_second': len(processed_windows) / processing_time if processing_time > 0 else 0,
            'total_summaries': self.total_summaries,
            'total_embeddings': self.total_embeddings,
            'embeddings': processed_windows,
            'errors': failed_windows,
            'timestamp': start_time.isoformat()
        }
        
        # Save embeddings
        embeddings_file = embeddings_output_dir / "sliding_window_embeddings.json"
        with open(embeddings_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save embeddings for vector database (backup)
        vector_db_file = embeddings_output_dir / "vector_db_embeddings.json"
        vector_db_data = []
        for window in processed_windows:
            if 'embedding' in window:
                vector_db_data.append({
                    'id': f"{video_id}_window_{window['window_id']}",
                    'embedding': window['embedding'],
                    'metadata': {
                        'video_id': video_id,
                        'window_id': window['window_id'],
                        'middle_segment_id': window['middle_segment_id'],
                        'summarized_text': window['summarized_text'],
                        'combined_text_before_summarization': window.get('combined_text_before_summarization', ''),  # NEW
                        'timestamp_start': window['timestamp_start'],
                        'timestamp_end': window['timestamp_end'],
                        'window_segments': window['window_segments']
                    }
                })
        
        with open(vector_db_file, 'w') as f:
            json.dump(vector_db_data, f, indent=2)
        
        self.logger.info(f"Sliding window embedding extraction completed for {video_id}")
        self.logger.info(f"Processed: {len(processed_windows)} windows")
        self.logger.info(f"Failed: {len(failed_windows)} windows")
        self.logger.info(f"Stored in vector DB: {stored_embeddings}")
        self.logger.info(f"Processing time: {processing_time:.2f}s")
        self.logger.info(f"Total summaries: {self.total_summaries}")
        self.logger.info(f"Total embeddings: {self.total_embeddings}")
        
        return summary

    def batch_process_videos(self, video_dir: str) -> List[Dict[str, Any]]:
        """
        Process multiple videos with sliding window embedding extraction
        
        Args:
            video_dir: Directory containing processed videos
            
        Returns:
            List of processing results
        """
        video_dir = Path(video_dir)
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        
        # Find processed videos with GPT-4V features
        video_ids = []
        for video_path in video_dir.iterdir():
            if video_path.is_dir() and (video_path / "gpt4v_features" / "processed_text.json").exists():
                video_ids.append(video_path.name)
        
        if not video_ids:
            self.logger.warning(f"No videos with GPT-4V features found in {video_dir}")
            return []
        
        self.logger.info(f"Found {len(video_ids)} videos for sliding window embedding processing")
        
        results = []
        for video_id in video_ids:
            try:
                # Load processed text data
                text_file = video_dir / video_id / "gpt4v_features" / "processed_text.json"
                with open(text_file, 'r') as f:
                    segments_data = json.load(f)
                
                # Process windows
                result = self.process_video_windows(video_id, segments_data)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process video {video_id}: {str(e)}")
                continue
        
        return results

    def search_similar_segments(self, query_text: str, query_image_path: str = None, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar segments using text and/or image query
        
        Args:
            query_text: Text query
            query_image_path: Optional image path for multimodal search
            n_results: Number of results to return
            
        Returns:
            List of similar segments
        """
        try:
            # Prepare query embedding
            if query_image_path and Path(query_image_path).exists():
                # Multimodal query
                query_embedding = self.encode_multimodal(query_image_path, query_text)
            else:
                # Text-only query
                query_embedding = self.encode_multimodal("", query_text)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i, (metadata, document, distance) in enumerate(zip(
                results['metadatas'][0], 
                results['documents'][0], 
                results['distances'][0]
            )):
                formatted_results.append({
                    'rank': i + 1,
                    'similarity_score': 1 - distance,  # Convert distance to similarity
                    'video_id': metadata['video_id'],
                    'window_id': metadata['window_id'],
                    'middle_segment_id': metadata['middle_segment_id'],
                    'summarized_text': document,
                    'timestamp_start': metadata['timestamp_start'],
                    'timestamp_end': metadata['timestamp_end'],
                    'window_segments': json.loads(metadata['window_segments']),
                    'middle_frame_path': metadata['middle_frame_path']
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Failed to search similar segments: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'total_embeddings': count,
                'vector_db_path': str(self.vector_db_path)
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")
            return {}


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Sliding Window Embedding Extractor with Vector DB')
    parser.add_argument('input', help='Video directory path')
    parser.add_argument('-o', '--output', default='processed_videos', 
                       help='Output directory')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model', default='intfloat/mmE5-mllama-11b-instruct',
                       help='Multimodal embedding model name')
    parser.add_argument('--window-size', type=int, default=5,
                       help='Number of segments per window')
    parser.add_argument('--hop-size', type=int, default=3,
                       help='Number of segments to advance between windows')
    parser.add_argument('--vector-db-path', default='vector_db',
                       help='Path to ChromaDB database')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = SlidingWindowEmbeddingExtractor(
        args.output, 
        args.log_level, 
        args.api_key,
        args.model,
        args.window_size,
        args.hop_size,
        args.vector_db_path
    )
    
    # Process videos
    try:
        results = extractor.batch_process_videos(args.input)
        print(f"Sliding window embedding processing completed: {len(results)} videos processed")
        
        total_windows = sum(r['processed_windows'] for r in results)
        failed_windows = sum(r['failed_windows'] for r in results)
        total_summaries = sum(r['total_summaries'] for r in results)
        total_embeddings = sum(r['total_embeddings'] for r in results)
        total_stored = sum(r['stored_embeddings'] for r in results)
        
        print(f"Total windows: {total_windows}")
        print(f"Failed windows: {failed_windows}")
        print(f"Total summaries: {total_summaries}")
        print(f"Total embeddings: {total_embeddings}")
        print(f"Stored in vector DB: {total_stored}")
        
        # Show collection stats
        stats = extractor.get_collection_stats()
        print(f"Vector DB collection: {stats.get('collection_name', 'N/A')}")
        print(f"Total embeddings in DB: {stats.get('total_embeddings', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()