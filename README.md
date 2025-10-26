# multimodal-video-qa
🎥 Ask questions about video content and get intelligent, cited answers. Built with multimodal AI, conversational memory, and vector databases.

## ✨ Features

- **📥 YouTube Video Download**: Download videos or playlists with automatic subtitle extraction
- **🎙️ Audio Processing**: Automatic transcription using OpenAI Whisper
- **🖼️ Context-Aware Segmentation**: Intelligent frame extraction based on video content and transcripts
- **🧠 GPT-4V Feature Extraction**: Visual understanding and multimodal feature extraction
- **🔍 Sliding Window Embeddings**: Semantic search using state-of-the-art multimodal embeddings (mmE5)
- **💬 Conversational Memory**: Context-aware responses with automatic conversation summarization
- **🌐 Streamlit Web Interface**: Intuitive UI for interactive video Q&A
- **📚 Citation Support**: Answers include video timestamps and frame references

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd multimodal-video-qa

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
# Required environment variable
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Download Videos

```bash
# Download a single video
python youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Download a playlist
python youtube_downloader.py "https://www.youtube.com/playlist?list=PLAYLIST_ID"

# Download with specific quality
python youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID" -q 720p
```

### 4. Process Videos

Edit `pipeline_config.yaml` to configure processing options, then run:

```bash
# Run the complete pipeline
python pipeline_processor.py --config pipeline_config.yaml
```

### 5. Ask Questions

**Web Interface**

```bash
streamlit run streamlit_app_conversational.py
```

Then open your browser to `http://localhost:8501`

