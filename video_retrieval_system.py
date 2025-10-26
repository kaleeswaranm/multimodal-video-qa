import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import argparse
import chromadb
from chromadb.config import Settings
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import time

class VideoRetrievalSystem:
    def __init__(self, vector_db_path: str = "vector_db", 
                 model_name: str = "intfloat/mmE5-mllama-11b-instruct",
                 log_level: str = "INFO"):
        """
        Initialize Video Retrieval System
        
        Args:
            vector_db_path: Path to ChromaDB database
            model_name: Multimodal embedding model name
            log_level: Logging level
        """
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('video_retrieval.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB
        self.vector_db_path = Path(vector_db_path)
        if not self.vector_db_path.exists():
            raise FileNotFoundError(f"Vector database not found: {vector_db_path}")
        
        self.logger.info(f"Loading ChromaDB from: {self.vector_db_path}")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vector_db_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        # Load collection
        self.collection_name = "video_segments"
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            self.logger.info(f"Loaded collection: {self.collection_name}")
        except Exception as e:
            raise FileNotFoundError(f"Collection '{self.collection_name}' not found: {str(e)}")
        
        # Initialize embedding model
        self.logger.info(f"Loading embedding model: {model_name}")
        self._initialize_embedding_model(model_name)
        
        # Load video metadata
        self.video_metadata = self._load_video_metadata()
        
        # Statistics
        self.total_queries = 0
        self.total_results = 0

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

    def encode_query(self, query_text: str, query_image_path: str = None) -> np.ndarray:
        """
        Encode query (text + optional image) using mmE5 model
        
        Args:
            query_text: Text query
            query_image_path: Optional image path for multimodal search
            
        Returns:
            Query embedding vector
        """
        try:
            if query_image_path and Path(query_image_path).exists():
                # Multimodal query
                image = Image.open(query_image_path)
                # input_text = f'<|image|><|begin_of_text|>{query_text}\n'
                # input_text = f"<|image|><|begin_of_text|>Represent the given educational video frame with the following text for retrieval.\nTEXT:\n{query_text}\n"
                input_text = f"<|image|><|begin_of_text|>Find me educational video frames that best match the image and the following query:\n{query_text}\n"

                # Process inputs
                inputs = self.processor(
                    text=input_text, 
                    images=[image], 
                    return_tensors="pt"
                )
            else:
                # Text-only query
                # input_text = f'<|begin_of_text|>{query_text}\n'
                input_text = f"Find me educational video frames that best match the following query:\n{query_text}\n"
                
                # Process inputs
                inputs = self.processor(
                    text=input_text, 
                    return_tensors="pt"
                )
            
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
            self.logger.error(f"Failed to encode query: {str(e)}")
            raise

    def _load_video_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load video metadata from processed videos directory"""
        video_metadata = {}
        
        # Look for processed videos
        processed_videos_dir = Path("processed_videos")
        if not processed_videos_dir.exists():
            self.logger.warning("Processed videos directory not found")
            return video_metadata
        
        for video_dir in processed_videos_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            video_id = video_dir.name
            
            # Load segment metadata
            segment_file = video_dir / "segment_frame_metadata.json"
            if segment_file.exists():
                try:
                    with open(segment_file, 'r') as f:
                        segment_data = json.load(f)
                    video_metadata[video_id] = {
                        'segment_metadata': segment_data,
                        'video_path': str(video_dir),
                        'total_segments': len(segment_data.get('frame_metadata', []))
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to load metadata for {video_id}: {str(e)}")
        
        self.logger.info(f"Loaded metadata for {len(video_metadata)} videos")
        return video_metadata

    def search_videos(self, query_text: str, query_image_path: str = None, 
                     n_results: int = 10, min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for relevant video segments
        
        Args:
            query_text: Text query
            query_image_path: Optional image path for multimodal search
            n_results: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of ranked search results
        """
        try:
            start_time = time.time()
            
            # Encode query
            query_embedding = self.encode_query(query_text, query_image_path)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Format results
            search_results = []
            for i, (metadata, document, distance) in enumerate(zip(
                results['metadatas'][0], 
                results['documents'][0], 
                results['distances'][0]
            )):
                similarity_score = 1 - distance  # Convert distance to similarity
                
                if similarity_score < min_similarity:
                    continue
                
                # Get additional metadata
                video_id = metadata['video_id']
                window_id = metadata['window_id']
                
                # Get video info
                video_info = self.video_metadata.get(video_id, {})
                
                result = {
                    'rank': i + 1,
                    'similarity_score': similarity_score,
                    'video_id': video_id,
                    'window_id': window_id,
                    'middle_segment_id': metadata['middle_segment_id'],
                    'summarized_text': document,
                    'combined_text_before_summarization': metadata['combined_text_before_summarization'],
                    'timestamp_start': metadata['timestamp_start'],
                    'timestamp_end': metadata['timestamp_end'],
                    'window_segments': json.loads(metadata['window_segments']),
                    'middle_frame_path': metadata['middle_frame_path'],
                    'video_info': video_info,
                    'query_time': time.time() - start_time
                }
                
                search_results.append(result)
            
            # Update statistics
            self.total_queries += 1
            self.total_results += len(search_results)
            
            self.logger.info(f"Search completed: {len(search_results)} results in {time.time() - start_time:.2f}s")
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Failed to search videos: {str(e)}")
            return []

    def get_video_segment_details(self, video_id: str, segment_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific video segment
        
        Args:
            video_id: Video identifier
            segment_id: Segment identifier
            
        Returns:
            Detailed segment information
        """
        try:
            video_info = self.video_metadata.get(video_id, {})
            segment_metadata = video_info.get('segment_metadata', {})
            frame_metadata = segment_metadata.get('frame_metadata', [])
            
            # Find the segment
            segment_info = None
            for frame_info in frame_metadata:
                if frame_info.get('segment_id') == segment_id:
                    segment_info = frame_info
                    break
            
            if not segment_info:
                raise ValueError(f"Segment {segment_id} not found in video {video_id}")
            
            # Get frame path
            frame_filename = segment_info.get('frame_filename', '')
            frame_path = f"processed_videos/{video_id}/segmented_frames/{frame_filename}"
            
            # Get transcript
            transcript_file = Path(f"processed_videos/{video_id}/transcript.json")
            transcript_text = ""
            if transcript_file.exists():
                try:
                    with open(transcript_file, 'r') as f:
                        transcript_data = json.load(f)
                        # Find transcript segment
                        for segment in transcript_data.get('segments', []):
                            if abs(segment.get('start', 0) - segment_info.get('segment_start', 0)) < 1.0:
                                transcript_text = segment.get('text', '')
                                break
                except Exception as e:
                    self.logger.warning(f"Failed to load transcript: {str(e)}")
            
            return {
                'video_id': video_id,
                'segment_id': segment_id,
                'frame_path': frame_path,
                'frame_filename': frame_filename,
                'segment_text': segment_info.get('segment_text', ''),
                'transcript_text': transcript_text,
                'timestamp_start': segment_info.get('segment_start', 0),
                'timestamp_end': segment_info.get('segment_end', 0),
                'segment_duration': segment_info.get('segment_end', 0) - segment_info.get('segment_start', 0),
                'video_info': video_info
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get segment details: {str(e)}")
            return {}

    def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """
        Get information about a specific video
        
        Args:
            video_id: Video identifier
            
        Returns:
            Video information
        """
        return self.video_metadata.get(video_id, {})

    def list_videos(self) -> List[Dict[str, Any]]:
        """
        List all available videos
        
        Returns:
            List of video information
        """
        videos = []
        for video_id, video_info in self.video_metadata.items():
            videos.append({
                'video_id': video_id,
                'total_segments': video_info.get('total_segments', 0),
                'video_path': video_info.get('video_path', '')
            })
        
        return videos

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
                'vector_db_path': str(self.vector_db_path),
                'total_videos': len(self.video_metadata),
                'total_queries': self.total_queries,
                'total_results': self.total_results
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")
            return {}

    def search_by_video(self, video_id: str, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search within a specific video
        
        Args:
            video_id: Video identifier
            query_text: Text query
            n_results: Number of results to return
            
        Returns:
            List of search results from the specified video
        """
        try:
            # Encode query
            query_embedding = self.encode_query(query_text)
            
            # Search in ChromaDB with metadata filter
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results * 2,  # Get more results to filter
                where={"video_id": video_id},
                include=['metadatas', 'documents', 'distances']
            )
            
            # Format results
            search_results = []
            for i, (metadata, document, distance) in enumerate(zip(
                results['metadatas'][0], 
                results['documents'][0], 
                results['distances'][0]
            )):
                similarity_score = 1 - distance
                
                result = {
                    'rank': i + 1,
                    'similarity_score': similarity_score,
                    'video_id': video_id,
                    'window_id': metadata['window_id'],
                    'middle_segment_id': metadata['middle_segment_id'],
                    'summarized_text': document,
                    'combined_text_before_summarization': metadata['combined_text_before_summarization'],
                    'timestamp_start': metadata['timestamp_start'],
                    'timestamp_end': metadata['timestamp_end'],
                    'window_segments': json.loads(metadata['window_segments']),
                    'middle_frame_path': metadata['middle_frame_path']
                }
                
                search_results.append(result)
                
                if len(search_results) >= n_results:
                    break
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Failed to search by video: {str(e)}")
            return []


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Video Retrieval System')
    parser.add_argument('--vector-db-path', default='vector_db',
                       help='Path to ChromaDB database')
    parser.add_argument('--model', default='intfloat/mmE5-mllama-11b-instruct',
                       help='Multimodal embedding model name')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search videos')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--image', help='Optional image path for multimodal search')
    search_parser.add_argument('--results', type=int, default=10, help='Number of results')
    search_parser.add_argument('--min-similarity', type=float, default=0.0, help='Minimum similarity threshold')
    
    # Video search command
    video_search_parser = subparsers.add_parser('search-video', help='Search within a specific video')
    video_search_parser.add_argument('video_id', help='Video identifier')
    video_search_parser.add_argument('query', help='Search query')
    video_search_parser.add_argument('--results', type=int, default=5, help='Number of results')
    
    # Segment command
    segment_parser = subparsers.add_parser('segment', help='Get segment details')
    segment_parser.add_argument('video_id', help='Video identifier')
    segment_parser.add_argument('segment_id', type=int, help='Segment identifier')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all videos')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show collection statistics')
    
    args = parser.parse_args()
    
    # Initialize retrieval system
    try:
        retrieval_system = VideoRetrievalSystem(
            args.vector_db_path,
            args.model,
            args.log_level
        )
    except Exception as e:
        print(f"Failed to initialize retrieval system: {str(e)}")
        return
    
    # Execute command
    if args.command == 'search':
        results = retrieval_system.search_videos(
            args.query, 
            args.image, 
            args.results, 
            args.min_similarity
        )
        
        print(f"\nFound {len(results)} results for query: '{args.query}'")
        print("=" * 80)
        
        for result in results:
            print(f"Rank {result['rank']}: {result['similarity_score']:.3f}")
            print(f"Video: {result['video_id']}")
            print(f"Timestamp: {result['timestamp_start']:.1f}s - {result['timestamp_end']:.1f}s")
            print(f"Summary: {result['summarized_text'][:200]}...")
            print(f"Frame: {result['middle_frame_path']}")
            print("-" * 80)
    
    elif args.command == 'search-video':
        results = retrieval_system.search_by_video(
            args.video_id,
            args.query,
            args.results
        )
        
        print(f"\nFound {len(results)} results in video '{args.video_id}' for query: '{args.query}'")
        print("=" * 80)
        
        for result in results:
            print(f"Rank {result['rank']}: {result['similarity_score']:.3f}")
            print(f"Timestamp: {result['timestamp_start']:.1f}s - {result['timestamp_end']:.1f}s")
            print(f"Summary: {result['summarized_text'][:200]}...")
            print(f"Frame: {result['middle_frame_path']}")
            print("-" * 80)
    
    elif args.command == 'segment':
        segment_info = retrieval_system.get_video_segment_details(
            args.video_id,
            args.segment_id
        )
        
        if segment_info:
            print(f"\nSegment {args.segment_id} from video '{args.video_id}':")
            print("=" * 80)
            print(f"Timestamp: {segment_info['timestamp_start']:.1f}s - {segment_info['timestamp_end']:.1f}s")
            print(f"Duration: {segment_info['segment_duration']:.1f}s")
            print(f"Frame: {segment_info['frame_path']}")
            print(f"Segment text: {segment_info['segment_text']}")
            print(f"Transcript: {segment_info['transcript_text']}")
        else:
            print(f"Segment {args.segment_id} not found in video '{args.video_id}'")
    
    elif args.command == 'list':
        videos = retrieval_system.list_videos()
        
        print(f"\nAvailable videos ({len(videos)}):")
        print("=" * 80)
        
        for video in videos:
            print(f"Video ID: {video['video_id']}")
            print(f"Segments: {video['total_segments']}")
            print(f"Path: {video['video_path']}")
            print("-" * 80)
    
    elif args.command == 'stats':
        stats = retrieval_system.get_collection_stats()
        
        print("\nCollection Statistics:")
        print("=" * 80)
        print(f"Collection: {stats.get('collection_name', 'N/A')}")
        print(f"Total embeddings: {stats.get('total_embeddings', 'N/A')}")
        print(f"Total videos: {stats.get('total_videos', 'N/A')}")
        print(f"Total queries: {stats.get('total_queries', 'N/A')}")
        print(f"Total results: {stats.get('total_results', 'N/A')}")
        print(f"Vector DB path: {stats.get('vector_db_path', 'N/A')}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()