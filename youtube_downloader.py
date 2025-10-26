import yt_dlp
import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import argparse

class YouTubeDownloader:
    def __init__(self, output_dir: str = "downloads", log_level: str = "INFO"):
        """
        Initialize YouTube Downloader
        
        Args:
            output_dir: Directory to save downloaded videos
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'download.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Download statistics
        self.stats = {
            'total_downloaded': 0,
            'total_failed': 0,
            'total_skipped': 0,
            'start_time': None,
            'end_time': None
        }

    def setup_ydl_opts(self, quality: str = "best", output_path: str = None, 
                      extract_audio: bool = False, write_subtitles: bool = True) -> Dict[str, Any]:
        """
        Configure yt-dlp options
        
        Args:
            quality: Video quality preference
            output_path: Custom output path
            extract_audio: Whether to extract audio only
            write_subtitles: Whether to download subtitles
            
        Returns:
            Dictionary of yt-dlp options
        """
        if output_path is None:
            output_path = str(self.output_dir)
        
        # Base output template
        if extract_audio:
            output_template = os.path.join(output_path, "%(playlist_title)s", "%(title)s.%(ext)s")
        else:
            output_template = os.path.join(output_path, "%(playlist_title)s", "%(title)s.%(ext)s")
        
        opts = {
            'outtmpl': output_template,
            'format': self._get_format_selector(quality, extract_audio),
            'writesubtitles': write_subtitles,
            'writeautomaticsub': write_subtitles,
            'subtitleslangs': ['en', 'en-US', 'en-GB'],
            'subtitlesformat': 'vtt',
            'writedescription': True,
            'writeinfojson': True,
            'writethumbnail': True,
            'ignoreerrors': True,  # Continue on download errors
            'no_warnings': False,
            'extractaudio': extract_audio,
            'audioformat': 'mp3' if extract_audio else None,
            'embed_subs': True,
            'embed_thumbnail': True,
        }
        
        return opts

    def _get_format_selector(self, quality: str, extract_audio: bool) -> str:
        """Get format selector based on quality preference"""
        if extract_audio:
            return "bestaudio/best"
        
        quality_map = {
            "best": "best[height<=1080]",
            "720p": "best[height<=720]",
            "480p": "best[height<=480]",
            "360p": "best[height<=360]",
            "worst": "worst"
        }
        
        return quality_map.get(quality, "best[height<=1080]")

    def download_video(self, url: str, quality: str = "best", 
                      extract_audio: bool = False, write_subtitles: bool = True) -> bool:
        """
        Download a single video
        
        Args:
            url: YouTube video URL
            quality: Video quality preference
            extract_audio: Whether to extract audio only
            write_subtitles: Whether to download subtitles
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Starting download: {url}")
            
            opts = self.setup_ydl_opts(quality, extract_audio=extract_audio, 
                                     write_subtitles=write_subtitles)
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'Unknown')
                self.logger.info(f"Video: {video_title}")
                
                # Download the video
                ydl.download([url])
                
            self.stats['total_downloaded'] += 1
            self.logger.info(f"Successfully downloaded: {video_title}")
            return True
            
        except Exception as e:
            self.stats['total_failed'] += 1
            self.logger.error(f"Failed to download {url}: {str(e)}")
            return False

    def download_playlist(self, url: str, quality: str = "best", 
                         extract_audio: bool = False, write_subtitles: bool = True,
                         max_downloads: Optional[int] = None) -> Dict[str, Any]:
        """
        Download an entire playlist
        
        Args:
            url: YouTube playlist URL
            quality: Video quality preference
            extract_audio: Whether to extract audio only
            write_subtitles: Whether to download subtitles
            max_downloads: Maximum number of videos to download (None for all)
            
        Returns:
            Dictionary with download statistics
        """
        self.stats['start_time'] = datetime.now()
        self.logger.info(f"Starting playlist download: {url}")
        
        try:
            opts = self.setup_ydl_opts(quality, extract_audio=extract_audio, 
                                     write_subtitles=write_subtitles)
            
            # Add progress hook
            def progress_hook(d):
                if d['status'] == 'downloading':
                    percent = d.get('_percent_str', 'N/A')
                    speed = d.get('_speed_str', 'N/A')
                    self.logger.info(f"Downloading: {percent} at {speed}")
                elif d['status'] == 'finished':
                    self.logger.info(f"Finished downloading: {d['filename']}")
            
            opts['progress_hooks'] = [progress_hook]
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                # Get playlist info
                playlist_info = ydl.extract_info(url, download=False)
                playlist_title = playlist_info.get('title', 'Unknown Playlist')
                total_videos = len(playlist_info.get('entries', []))
                
                self.logger.info(f"Playlist: {playlist_title}")
                self.logger.info(f"Total videos: {total_videos}")
                
                if max_downloads and max_downloads < total_videos:
                    self.logger.info(f"Limiting to {max_downloads} videos")
                
                # Download the playlist
                ydl.download([url])
                
        except Exception as e:
            self.logger.error(f"Failed to download playlist {url}: {str(e)}")
        
        self.stats['end_time'] = datetime.now()
        return self._get_download_stats()

    def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Get video information without downloading
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary with video information
        """
        try:
            opts = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    'title': info.get('title'),
                    'description': info.get('description'),
                    'duration': info.get('duration'),
                    'upload_date': info.get('upload_date'),
                    'view_count': info.get('view_count'),
                    'like_count': info.get('like_count'),
                    'channel': info.get('uploader'),
                    'channel_id': info.get('channel_id'),
                    'video_id': info.get('id'),
                    'thumbnail': info.get('thumbnail'),
                    'formats': len(info.get('formats', [])),
                    'subtitles': list(info.get('subtitles', {}).keys()),
                    'automatic_captions': list(info.get('automatic_captions', {}).keys())
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get video info for {url}: {str(e)}")
            return {}

    def get_playlist_info(self, url: str) -> Dict[str, Any]:
        """
        Get playlist information without downloading
        
        Args:
            url: YouTube playlist URL
            
        Returns:
            Dictionary with playlist information
        """
        try:
            opts = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    'title': info.get('title'),
                    'description': info.get('description'),
                    'uploader': info.get('uploader'),
                    'video_count': len(info.get('entries', [])),
                    'videos': [
                        {
                            'title': entry.get('title'),
                            'url': entry.get('webpage_url'),
                            'duration': entry.get('duration'),
                            'upload_date': entry.get('upload_date')
                        }
                        for entry in info.get('entries', [])[:10]  # First 10 videos
                    ]
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get playlist info for {url}: {str(e)}")
            return {}

    def _get_download_stats(self) -> Dict[str, Any]:
        """Get download statistics"""
        duration = None
        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        return {
            'total_downloaded': self.stats['total_downloaded'],
            'total_failed': self.stats['total_failed'],
            'total_skipped': self.stats['total_skipped'],
            'duration_seconds': duration,
            'start_time': self.stats['start_time'].isoformat() if self.stats['start_time'] else None,
            'end_time': self.stats['end_time'].isoformat() if self.stats['end_time'] else None
        }

    def reset_stats(self):
        """Reset download statistics"""
        self.stats = {
            'total_downloaded': 0,
            'total_failed': 0,
            'total_skipped': 0,
            'start_time': None,
            'end_time': None
        }


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='YouTube Video Downloader')
    parser.add_argument('url', help='YouTube video or playlist URL')
    parser.add_argument('-o', '--output', default='downloads', help='Output directory')
    parser.add_argument('-q', '--quality', default='best', 
                       choices=['best', '720p', '480p', '360p', 'worst'],
                       help='Video quality')
    parser.add_argument('--audio-only', action='store_true', help='Extract audio only')
    parser.add_argument('--no-subtitles', action='store_true', help='Skip subtitle download')
    parser.add_argument('--max-downloads', type=int, help='Maximum number of videos to download')
    parser.add_argument('--info-only', action='store_true', help='Show video/playlist info only')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = YouTubeDownloader(args.output, args.log_level)
    
    # Check if URL is a playlist
    is_playlist = 'playlist' in args.url.lower()
    
    if args.info_only:
        if is_playlist:
            info = downloader.get_playlist_info(args.url)
            print(json.dumps(info, indent=2))
        else:
            info = downloader.get_video_info(args.url)
            print(json.dumps(info, indent=2))
        return
    
    # Download
    if is_playlist:
        stats = downloader.download_playlist(
            args.url, 
            args.quality, 
            args.audio_only, 
            not args.no_subtitles,
            args.max_downloads
        )
        print(f"\nDownload completed!")
        print(f"Statistics: {json.dumps(stats, indent=2)}")
    else:
        success = downloader.download_video(
            args.url, 
            args.quality, 
            args.audio_only, 
            not args.no_subtitles
        )
        if success:
            print("Video downloaded successfully!")
        else:
            print("Video download failed!")


if __name__ == "__main__":
    print("Starting YouTube Downloader")
    main()