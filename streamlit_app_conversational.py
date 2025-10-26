import streamlit as st
import tempfile
import os
import uuid
from pathlib import Path
from video_qa_system_conversational import ConversationalVideoQASystem
import time

# Page configuration
st.set_page_config(
    page_title="Conversational Video Q&A System",
    page_icon="🎥",
    layout="wide"
)

# Initialize Q&A system (cached for performance)
@st.cache_resource
def load_qa_system():
    """Load the conversational Q&A system with error handling"""
    try:
        return ConversationalVideoQASystem()
    except Exception as e:
        st.error(f"Failed to load Q&A system: {str(e)}")
        st.info("Make sure you have processed videos and have embeddings in the vector database.")
        return None

def save_uploaded_image(uploaded_file):
    """Save uploaded image to temporary file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

def display_conversation_history(qa_system, session_id):
    """Display conversation history"""
    try:
        session_history = qa_system.conversation_memory.get_session_history(session_id)
        
        if session_history['messages']:
            st.subheader("💬 Conversation History")
            
            # Show summary if available
            if session_history['summary']:
                with st.expander("📋 Conversation Summary", expanded=False):
                    st.write(session_history['summary'])
            
            # Show recent messages
            messages = session_history['messages']
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    question_msg = messages[i]
                    answer_msg = messages[i + 1]
                    
                    with st.expander(f"Q{i//2 + 1}: {question_msg['content'][:50]}...", expanded=False):
                        st.write(f"**Question:** {question_msg['content']}")
                        st.write(f"**Answer:** {answer_msg['content']}")
                        st.write(f"**Confidence:** {answer_msg.get('confidence', 'N/A')}")
                        
                        # Show citations if available
                        if 'citations' in answer_msg:
                            st.write("**Citations:**")
                            for j, citation in enumerate(answer_msg['citations'][:3], 1):
                                st.write(f"{j}. {citation['video_id']} ({citation['timestamp_start']:.1f}s - {citation['timestamp_end']:.1f}s)")
            
            st.divider()
    except Exception as e:
        st.warning(f"Could not load conversation history: {str(e)}")

def display_qa_results_with_context(response):
    """Display Q&A results with conversation context"""
    # Answer section
    st.subheader("📝 Answer")
    st.markdown(response['answer'])
    
    # Show conversation context if available
    if response.get('conversation_context'):
        with st.expander("🔗 Conversation Context", expanded=False):
            st.write(response['conversation_context'])
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", f"{response['confidence']:.3f}")
    with col2:
        st.metric("Processing Time", f"{response['processing_time']:.2f}s")
    with col3:
        st.metric("Segments Found", len(response['retrieved_segments']))
    
    # Citations
    st.subheader("📚 Citations")
    for i, citation in enumerate(response['citations'], 1):
        with st.expander(f"Citation {i}: {citation['video_id']} ({citation['timestamp_start']:.1f}s - {citation['timestamp_end']:.1f}s)"):
            st.write(f"**Relevance Score:** {citation['similarity_score']:.3f}")
            st.write(f"**Content:** {citation['summarized_text']}")
            
            # Show frame if available
            if citation['middle_frame_path'] and Path(citation['middle_frame_path']).exists():
                st.image(citation['middle_frame_path'], caption="Frame Preview", width=300)

def main():
    """Main Streamlit app"""
    st.title("🎥 Conversational Video Q&A System")
    st.markdown("Ask questions about your educational videos and get comprehensive answers with citations. Features conversation memory and context-aware responses.")
    
    # Load Q&A system
    qa_system = load_qa_system()
    if qa_system is None:
        st.stop()
    
    # Initialize session
    if 'session_id' not in st.session_state:
        st.session_state.session_id = qa_system.conversation_memory.create_session()
    
    # Sidebar with system info
    with st.sidebar:
        st.header("System Info")
        
        # Get system stats
        try:
            stats = qa_system.get_system_stats()
            st.write(f"**Total Videos:** {stats['retrieval_stats'].get('total_videos', 'N/A')}")
            st.write(f"**Total Embeddings:** {stats['retrieval_stats'].get('total_embeddings', 'N/A')}")
            st.write(f"**Total Queries:** {stats['total_qa_queries']}")
        except Exception as e:
            st.warning(f"Could not load stats: {str(e)}")
        
        st.divider()
        
        # Conversation session info
        st.subheader("Conversation")
        st.write(f"**Session:** {st.session_state.session_id[:8]}...")
        
        # Get conversation stats
        try:
            session_history = qa_system.conversation_memory.get_session_history(st.session_state.session_id)
            st.write(f"**Follow-ups:** {session_history['follow_up_count']}")
            
            if session_history['summary']:
                st.write("**Summary:**")
                st.write(session_history['summary'][:200] + "...")
        except Exception as e:
            st.warning(f"Could not load conversation stats: {str(e)}")
        
        # New conversation button
        if st.button("🔄 New Conversation"):
            st.session_state.session_id = qa_system.conversation_memory.create_session()
            st.rerun()
        
        st.divider()
        
        # Available videos
        try:
            videos = qa_system.retrieval_system.list_videos()
            if videos:
                st.subheader("Available Videos")
                for video in videos:
                    st.write(f"• {video['video_id']}")
                    st.write(f"  ({video['total_segments']} segments)")
            else:
                st.warning("No videos found")
        except Exception as e:
            st.warning(f"Could not load videos: {str(e)}")
    
    # Display conversation history
    display_conversation_history(qa_system, st.session_state.session_id)
    
    # Main interface
    st.header("Ask a Question")
    
    # Define uploaded_file before columns
    uploaded_file = st.file_uploader(
        "Upload image (optional):",
        type=['png', 'jpg', 'jpeg'],
        help="For multimodal queries - ask questions about images"
    )
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What is attention mechanism in transformers? How does gradient descent work?",
            help="Ask any question about the content in your processed videos. This system remembers previous questions and can provide context-aware answers."
        )
        
        # Show uploaded image in main content area if available
        if uploaded_file:
            st.subheader("📷 Uploaded Image")
            st.image(uploaded_file, caption="Query Image", width=500)
    
    with col2:
        st.subheader("Options")
        
        # Search parameters
        n_results = st.slider("Number of results:", 1, 10, 5)
        min_similarity = st.slider("Minimum similarity:", 0.0, 1.0, 0.0, 0.1)
        
        # Video-specific search
        try:
            videos = qa_system.retrieval_system.list_videos()
            if videos:
                video_options = ["All Videos"] + [v['video_id'] for v in videos]
                selected_video = st.selectbox("Search in:", video_options)
            else:
                selected_video = "All Videos"
        except:
            selected_video = "All Videos"
    
    # Search button
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not query.strip():
            st.error("Please enter a question!")
        else:
            with st.spinner("Searching and generating answer..."):
                try:
                    start_time = time.time()
                    
                    # Process query with conversation context
                    if uploaded_file:
                        # Save uploaded image temporarily
                        image_path = save_uploaded_image(uploaded_file)
                        try:
                            if selected_video == "All Videos":
                                response = qa_system.ask_question_with_context(
                                    st.session_state.session_id,
                                    query,
                                    image_path,
                                    n_results,
                                    min_similarity
                                )
                            else:
                                response = qa_system.ask_question_by_video_with_context(
                                    st.session_state.session_id,
                                    selected_video,
                                    query,
                                    n_results
                                )
                        finally:
                            # Clean up temporary file
                            if os.path.exists(image_path):
                                os.unlink(image_path)
                    else:
                        if selected_video == "All Videos":
                            response = qa_system.ask_question_with_context(
                                st.session_state.session_id,
                                query,
                                None,
                                n_results,
                                min_similarity
                            )
                        else:
                            response = qa_system.ask_question_by_video_with_context(
                                st.session_state.session_id,
                                selected_video,
                                query,
                                n_results
                            )
                    
                    # Display results
                    st.divider()
                    display_qa_results_with_context(response)
                    
                    # Show processing time
                    total_time = time.time() - start_time
                    st.caption(f"Total processing time: {total_time:.2f}s")
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.info("Make sure your vector database is properly set up and contains embeddings.")

if __name__ == "__main__":
    main()