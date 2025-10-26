import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add workspace to path
WS = Path(__file__).parent.resolve()
sys.path.append(str(WS))

# Import pipeline modules
# from audio_processor import AudioProcessor
from context_aware_extractor import SegmentAwareExtractor
from gpt4v_multimodal_extractor import GPT4VMultimodalExtractor
from sliding_window_embedding_extractor import SlidingWindowEmbeddingExtractor

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

class VideoQAPipeline:
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(Path(config_path))
        self.start_time = time.time()
        
        # Set up environment variables
        self._setup_environment()
        
        # Initialize logging
        self._setup_logging()
        
    def _load_config(self, path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        text = path.read_text(encoding="utf-8")
        
        if path.suffix.lower() in [".yml", ".yaml"] and HAS_YAML:
            return yaml.safe_load(text)
        else:
            return json.loads(text)
    
    def _setup_environment(self):
        """Set up environment variables from config"""
        if self.config.get("openai", {}).get("api_key"):
            os.environ["OPENAI_API_KEY"] = self.config["openai"]["api_key"]
    
    def _setup_logging(self):
        """Set up pipeline logging"""
        import logging
        
        log_level = self.config.get("runtime", {}).get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _log_stage(self, stage_name: str, message: str = ""):
        """Log stage information"""
        elapsed = time.time() - self.start_time
        self.logger.info(f"[{elapsed:.1f}s] {stage_name}: {message}")
    
    def run_audio_stage(self) -> bool:
        """Run audio processing stage"""
        try:
            self._log_stage("AUDIO", "Starting audio processing")
            
            input_dir = Path(self.config["paths"]["input_videos_dir"]).resolve()
            output_dir = Path(self.config["paths"]["processed_videos_dir"]).resolve()
            
            # Initialize AudioProcessor
            audio_processor = AudioProcessor(
                output_dir=str(output_dir),
                log_level=self.config.get("runtime", {}).get("log_level", "INFO")
            )
            
            # Set Whisper model if specified
            if "whisper_model" in self.config.get("audio", {}):
                audio_processor.whisper_model = self.config["audio"]["whisper_model"]
            
            # Process videos
            language = self.config.get("audio", {}).get("language")
            results = audio_processor.batch_process_audio(str(input_dir), language=language)
            
            self._log_stage("AUDIO", f"Completed processing {len(results)} videos")
            return True
            
        except Exception as e:
            self.logger.error(f"Audio stage failed: {str(e)}")
            return False
    
    def run_segmentation_stage(self) -> bool:
        """Run segment-aware frame extraction stage"""
        try:
            self._log_stage("SEGMENTATION", "Starting segment-aware frame extraction")
                        
            input_dir = Path(self.config["paths"]["input_videos_dir"]).resolve()
            processed_dir = Path(self.config["paths"]["processed_videos_dir"]).resolve()
            segment_extractor = SegmentAwareExtractor(
                output_dir=str(processed_dir),
                log_level=self.config.get("runtime", {}).get("log_level", "INFO")
            )
            
            # Process videos
            language = self.config.get("segmentation", {}).get("language")
            results = segment_extractor.batch_process_segments(str(input_dir), language=language)
            
            self._log_stage("SEGMENTATION", f"Completed processing {len(results)} videos")
            return True
            
        except Exception as e:
            self.logger.error(f"Segmentation stage failed: {str(e)}")
            return False
    
    def run_gpt4v_stage(self) -> bool:
        """Run GPT-4V multimodal feature extraction stage"""
        try:
            self._log_stage("GPT4V", "Starting GPT-4V multimodal feature extraction")
            
            processed_dir = Path(self.config["paths"]["processed_videos_dir"]).resolve()
            
            # Initialize GPT4VMultimodalExtractor
            gpt4v_extractor = GPT4VMultimodalExtractor(
                output_dir=str(processed_dir),
                log_level=self.config.get("runtime", {}).get("log_level", "INFO"),
                api_key=self.config.get("gpt4v", {}).get("api_key"),
                model=self.config.get("gpt4v", {}).get("model", "gpt-4o-mini")
            )
            
            # Set request delay
            gpt4v_extractor.request_delay = float(self.config.get("gpt4v", {}).get("request_delay", 1.0))
            
            # Process videos
            results = gpt4v_extractor.batch_process_videos(str(processed_dir))
            
            self._log_stage("GPT4V", f"Completed processing {len(results)} videos")
            return True
            
        except Exception as e:
            self.logger.error(f"GPT-4V stage failed: {str(e)}")
            return False
    
    def run_embeddings_stage(self) -> bool:
        """Run sliding window embedding extraction stage"""
        try:
            self._log_stage("EMBEDDINGS", "Starting sliding window embedding extraction")
            
            processed_dir = Path(self.config["paths"]["processed_videos_dir"]).resolve()
            
            # Initialize SlidingWindowEmbeddingExtractor
            embedding_extractor = SlidingWindowEmbeddingExtractor(
                output_dir=str(processed_dir),
                log_level=self.config.get("runtime", {}).get("log_level", "INFO"),
                api_key=self.config.get("openai", {}).get("api_key"),
                model_name=self.config.get("sliding_window", {}).get("model_name", "intfloat/mmE5-mllama-11b-instruct"),
                window_size=int(self.config.get("sliding_window", {}).get("window_size", 5)),
                hop_size=int(self.config.get("sliding_window", {}).get("hop_size", 3)),
                vector_db_path=self.config.get("chroma", {}).get("vector_db_path", "vector_db")
            )
            
            # Process videos
            results = embedding_extractor.batch_process_videos(str(processed_dir))
            
            # Get collection stats
            stats = embedding_extractor.get_collection_stats()
            self._log_stage("EMBEDDINGS", f"Completed processing {len(results)} videos")
            self._log_stage("EMBEDDINGS", f"ChromaDB stats: {stats}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Embeddings stage failed: {str(e)}")
            return False
    
    def run_pipeline(self, stop_after: Optional[str] = None):
        """Run the complete pipeline"""
        self._log_stage("PIPELINE", "Starting end-to-end video Q&A pipeline")
        
        stages = [
            ("audio", self.run_audio_stage),
            ("segmentation", self.run_segmentation_stage),
            ("gpt4v", self.run_gpt4v_stage),
            ("embeddings", self.run_embeddings_stage)
        ]
        
        for stage_name, stage_func in stages:
            # Check if stage should be skipped
            if not self.config.get("stages", {}).get(stage_name, True):
                self._log_stage(stage_name.upper(), "Skipped (disabled in config)")
                continue
            
            # Run stage
            success = stage_func()
            if not success:
                self.logger.error(f"Pipeline failed at {stage_name} stage")
                return False
            
            # Check if we should stop after this stage
            if stop_after == stage_name:
                self._log_stage("PIPELINE", f"Stopped after {stage_name} stage")
                return True
        
        total_time = time.time() - self.start_time
        self._log_stage("PIPELINE", f"Completed successfully in {total_time:.1f}s")
        return True


def main():
    parser = argparse.ArgumentParser(description="End-to-end Video Q&A Pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    parser.add_argument("--stop-after", 
                       choices=["audio", "segmentation", "gpt4v", "embeddings"],
                       help="Stop pipeline after specified stage")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be processed without running")
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = VideoQAPipeline(args.config)
        
        if args.dry_run:
            print("DRY RUN - Configuration loaded successfully:")
            print(json.dumps(pipeline.config, indent=2))
            return
        
        # Run pipeline
        success = pipeline.run_pipeline(stop_after=args.stop_after)
        
        if success:
            print("Pipeline completed successfully!")
        else:
            print("Pipeline failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Pipeline initialization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()