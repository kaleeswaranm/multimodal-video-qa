# multimodal-video-qa
🎥 Ask questions about video content and get intelligent, cited answers. Built with multimodal AI, conversational memory, and vector databases.

[![Watch the video](https://raw.githubusercontent.com/kaleeswaranm/multimodal-video-qa/main/images/thumbnail.png)](https://raw.githubusercontent.com/kaleeswaranm/multimodal-video-qa/main/images/streamlit-streamlit_app_conversational-2025-10-26-18-10-28.mp4)

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

## The streamlit UI

### Interface showing the system and conversation info with the list of available videos for searching
![Interface showing the system and conversation info with the list of available videos for searching](images/UI1.png)

### Interface to ask question and configure the retrieval parameters
![Interface to ask question and configure the retrieval parameters](images/UI2.png)

### Interface showing the answer and the citations with explainable evidence
![Interface showing the answer and the citations with explainable evidence](images/UI3.png)

### Interface showing the expanded view of the citations and the current conversational context
![Interface showing the expanded view of the citations and the current conversational context](images/UI4.png)

### Interface showing the all the previous conversations for the current session
![Interface showing the all the previous conversations for the current session](images/UI5.png)

## 🤝 Contributing
Contributions welcome! Please open an issue or submit a pull request.
Note! This application is tested on MPS backend, if you face any issue with CUDA or CPU backend please raise and issue

## 📄 License
See LICENSE file for details.

## Social
Connect with me on LinkedIn - https://www.linkedin.com/in/kaleeswaran-m/