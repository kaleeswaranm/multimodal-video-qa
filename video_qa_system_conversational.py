import openai
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
import argparse
import time
import uuid
from video_retrieval_system import VideoRetrievalSystem
from conversation_memory import ConversationMemory

class VideoAnswerGenerator:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """
        Initialize Video Answer Generator
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use for answer generation
        """
        self.api_key = api_key or self._get_api_key()
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('video_qa_conversational.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Answer generation settings
        self.max_context_tokens = 8000  # Leave room for response
        self.max_answer_tokens = 1000
        self.temperature = 0.3
        
        # Statistics
        self.total_queries = 0
        self.total_answers = 0
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

    def _build_context(self, retrieved_segments: List[Dict[str, Any]]) -> str:
        """
        Build context from retrieved segments
        
        Args:
            retrieved_segments: List of retrieved video segments
            
        Returns:
            Formatted context string
        """
        if not retrieved_segments:
            return "No relevant video segments found."
        
        context_parts = []
        
        for i, segment in enumerate(retrieved_segments, 1):
            video_id = segment.get('video_id', 'Unknown')
            timestamp_start = segment.get('timestamp_start', 0)
            timestamp_end = segment.get('timestamp_end', 0)
            summarized_text = segment.get('summarized_text', '')
            combined_text_before_summarization = segment.get('combined_text_before_summarization', '')
            similarity_score = segment.get('similarity_score', 0)
            
            # Format timestamp
            timestamp_str = f"{timestamp_start:.1f}s - {timestamp_end:.1f}s"
            
            # Build context with both summarized and original text
            context_parts.append(
                f"Segment {i} (Video: {video_id}, Time: {timestamp_str}, Relevance: {similarity_score:.3f}):\n"
                f"SUMMARY: {summarized_text}\n"
                f"ORIGINAL CONTENT: {combined_text_before_summarization}\n"
            )
        
        return "\n".join(context_parts)

    def _generate_contextual_answer(self, query: str, context: str, conversation_context: str) -> str:
        """
        Generate answer using GPT-4 with conversation context
        
        Args:
            query: User's question
            context: Context from retrieved segments
            conversation_context: Previous conversation context
            
        Returns:
            Generated answer
        """
        try:
            # Build context-aware prompt
            if conversation_context:
                prompt = f"""You are an expert AI assistant that answers questions based on educational video content. 
                You have access to both current video segments and previous conversation context.

                Current Question: {query}

                Previous Conversation Context:
                {conversation_context}

                Current Video Context:
                {context}

                Instructions:
                1. Answer the question considering both video content and previous conversation
                2. Reference previous topics when relevant (e.g., "As we discussed earlier...")
                3. Build upon previous knowledge when appropriate
                4. Use the EXACT structure below for your answer
                5. Be concise but comprehensive
                6. Maintain accuracy and avoid hallucination
                7. **CRITICAL**: You MUST include ALL retrieved segments in the "Evidence from Video" section

                REQUIRED ANSWER STRUCTURE:
                **Answer:** [Direct, concise answer to the question]

                **Key Points:** [2-4 bullet points with the most important information]

                **Evidence from Video:** [MUST include ALL retrieved segments]
                - **Segment 1** (Video: [video_id], Time: [timestamp]): [How this segment specifically answers the question, or "Not directly relevant but provides context about [topic]"]
                - **Segment 2** (Video: [video_id], Time: [timestamp]): [How this segment specifically answers the question, or "Not directly relevant but provides context about [topic]"]
                - **Segment 3** (Video: [video_id], Time: [timestamp]): [How this segment specifically answers the question, or "Not directly relevant but provides context about [topic]"]
                [... continue for ALL segments]

                **Additional Context:** [Any relevant connections to previous conversation or broader concepts]

                Answer:"""
            else:
                prompt = f"""You are an expert AI assistant that answers questions based on educational video content. 
                Provide accurate, comprehensive answers with proper citations to video timestamps.

                Question: {query}

                Context from video segments:
                {context}

                Instructions:
                1. Answer the question based on the provided video context
                2. Use the EXACT structure below for your answer
                3. Be concise but comprehensive
                4. Clearly show how each segment answers the question
                5. Maintain accuracy and avoid hallucination
                6. **CRITICAL**: You MUST include ALL retrieved segments in the "Evidence from Video" section

                REQUIRED ANSWER STRUCTURE:
                **Answer:** [Direct, concise answer to the question]

                **Key Points:** [2-4 bullet points with the most important information]

                **Evidence from Video:** [MUST include ALL retrieved segments]
                - **Segment 1** (Video: [video_id], Time: [timestamp]): [How this segment specifically answers the question, or "Not directly relevant but provides context about [topic]"]
                - **Segment 2** (Video: [video_id], Time: [timestamp]): [How this segment specifically answers the question, or "Not directly relevant but provides context about [topic]"]
                - **Segment 3** (Video: [video_id], Time: [timestamp]): [How this segment specifically answers the question, or "Not directly relevant but provides context about [topic]"]
                [... continue for ALL segments]

                **Additional Context:** [Any relevant broader concepts or implications]

                Answer:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_answer_tokens,
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Update statistics
            self.total_queries += 1
            self.total_answers += 1
            
            # Estimate cost
            input_tokens = len(prompt.split()) + 100  # Rough estimate
            output_tokens = len(answer.split())
            cost = (input_tokens * 0.00003) + (output_tokens * 0.00006)  # GPT-4o pricing
            self.total_cost += cost
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Failed to generate contextual answer: {str(e)}")
            return f"Error generating answer: {str(e)}"

    def _extract_citations(self, retrieved_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract citations from retrieved segments
        
        Args:
            retrieved_segments: List of retrieved video segments
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        
        for segment in retrieved_segments:
            citation = {
                'video_id': segment.get('video_id', 'Unknown'),
                'timestamp_start': segment.get('timestamp_start', 0),
                'timestamp_end': segment.get('timestamp_end', 0),
                'similarity_score': segment.get('similarity_score', 0),
                'middle_frame_path': segment.get('middle_frame_path', ''),
                'summarized_text': segment.get('summarized_text', '')[:200] + '...' if len(segment.get('summarized_text', '')) > 200 else segment.get('summarized_text', '')
            }
            citations.append(citation)
        
        return citations

    def _calculate_confidence(self, retrieved_segments: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on retrieved segments
        
        Args:
            retrieved_segments: List of retrieved video segments
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not retrieved_segments:
            return 0.0
        
        # Calculate average similarity score
        similarity_scores = [seg.get('similarity_score', 0) for seg in retrieved_segments]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Calculate confidence based on number of segments and similarity
        num_segments = len(retrieved_segments)
        segment_factor = min(num_segments / 5.0, 1.0)  # Cap at 5 segments
        
        confidence = (avg_similarity * 0.7) + (segment_factor * 0.3)
        return min(confidence, 1.0)

    def generate_answer_with_context(self, query: str, retrieved_segments: List[Dict[str, Any]], 
                                   conversation_context: str = "") -> Dict[str, Any]:
        """
        Generate comprehensive answer considering conversation context
        
        Args:
            query: User's question
            retrieved_segments: List of relevant video segments
            conversation_context: Previous conversation context
            
        Returns:
            Structured answer with citations
        """
        start_time = time.time()
        
        # Build context from segments
        context = self._build_context(retrieved_segments)
        
        # Generate answer using GPT-4 with conversation context
        answer = self._generate_contextual_answer(query, context, conversation_context)
        
        # Extract citations and timestamps
        citations = self._extract_citations(retrieved_segments)
        
        # Calculate confidence
        confidence = self._calculate_confidence(retrieved_segments)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        result = {
            'query': query,
            'answer': answer,
            'citations': citations,
            'confidence': confidence,
            'related_segments': retrieved_segments,
            'context_length': len(context),
            'conversation_context': conversation_context,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Generated contextual answer for query: '{query[:50]}...'")
        self.logger.info(f"Confidence: {confidence:.3f}, Processing time: {processing_time:.2f}s")
        
        return result


class ConversationalVideoQASystem:
    def __init__(self, vector_db_path: str = "vector_db", 
                 model_name: str = "intfloat/mmE5-mllama-11b-instruct",
                 api_key: str = None, log_level: str = "INFO"):
        """
        Initialize Conversational Video Q&A System
        
        Args:
            vector_db_path: Path to ChromaDB database
            model_name: Multimodal embedding model name
            api_key: OpenAI API key for answer generation
            log_level: Logging level
        """
        # Initialize retrieval system
        self.retrieval_system = VideoRetrievalSystem(
            vector_db_path=vector_db_path,
            model_name=model_name,
            log_level=log_level
        )
        
        # Initialize answer generator
        self.answer_generator = VideoAnswerGenerator(api_key=api_key)
        
        # Initialize conversation memory
        self.conversation_memory = ConversationMemory()
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('conversational_video_qa_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.total_qa_queries = 0
        self.total_qa_answers = 0

    def _build_contextualized_query(self, query: str, conversation_context: str) -> str:
        """
        Build a contextualized query for better retrieval
        
        Args:
            query: Current user query
            conversation_context: Previous conversation context
            
        Returns:
            Contextualized query string
        """
        if not conversation_context:
            return query  # No context available, use original query
        
        # Check if query contains pronouns or needs contextualization
        pronouns = ["it", "this", "that", "they", "them", "these", "those", "its", "their"]
        has_pronouns = any(pronoun in query.lower() for pronoun in pronouns)
        
        # Check if query is very short or vague
        is_vague = len(query.split()) <= 3 and not any(word in query.lower() for word in ["what", "how", "why", "when", "where", "who"])
        
        if not has_pronouns and not is_vague:
            return query  # No contextualization needed
        
        # Use GPT to contextualize the query
        prompt = f"""Given the conversation context and current question, create a concise query for video search.

        Conversation Context:
        {conversation_context}

        Current Question: {query}

        Instructions:
        1. Replace pronouns with specific terms from conversation
        2. Keep it short (max 8 words)
        3. Focus on core concept only
        4. Return only the query

        Query:"""

        try:
            response = self.answer_generator.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            contextualized_query = response.choices[0].message.content.strip()
            
            # Log the contextualization
            if contextualized_query != query:
                self.logger.info(f"Query contextualized: '{query}' → '{contextualized_query}'")
            else:
                self.logger.info(f"Query unchanged: '{query}'")
            
            return contextualized_query
            
        except Exception as e:
            self.logger.error(f"Failed to contextualize query: {str(e)}")
            return query  # Fallback to original query

    def ask_question_with_context(self, session_id: str, query: str, 
                                 query_image_path: str = None, 
                                 n_results: int = 5, min_similarity: float = 0.0) -> Dict[str, Any]:
        """
        Ask a question with conversation context and contextualized retrieval
        
        Args:
            session_id: Conversation session ID
            query: User's question
            query_image_path: Optional image path for multimodal search
            n_results: Number of segments to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            Complete Q&A response with conversation context
        """
        start_time = time.time()
        
        self.logger.info(f"Processing contextual Q&A query: '{query[:50]}...'")
        
        # 1. Get conversation context
        conversation_context = self.conversation_memory.get_context_for_query(session_id, query)
        
        # 2. Build contextualized query for retrieval
        contextualized_query = self._build_contextualized_query(query, conversation_context)
        
        # 3. Retrieve relevant segments using contextualized query
        if query_image_path and Path(query_image_path).exists():
            retrieved_segments = self.retrieval_system.search_videos(
                query_text=contextualized_query,  # Use contextualized query
                query_image_path=query_image_path,
                n_results=n_results,
                min_similarity=min_similarity
            )
        else:
            retrieved_segments = self.retrieval_system.search_videos(
                query_text=contextualized_query,  # Use contextualized query
                n_results=n_results,
                min_similarity=min_similarity
            )
        
        # 4. Generate answer with conversation context
        answer_result = self.answer_generator.generate_answer_with_context(
            query=query,  # Use original query for answer generation
            retrieved_segments=retrieved_segments,
            conversation_context=conversation_context
        )
        
        # 5. Store in conversation memory
        self.conversation_memory.add_message(
            session_id, 'user_question', query, 
            query_image=query_image_path,
            contextualized_query=contextualized_query
        )
        self.conversation_memory.add_message(
            session_id, 'system_answer', answer_result['answer'],
            citations=answer_result['citations'],
            confidence=answer_result['confidence'],
            retrieved_segments=retrieved_segments
        )
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Create final response
        response = {
            'query': query,
            'contextualized_query': contextualized_query,
            'answer': answer_result['answer'],
            'citations': answer_result['citations'],
            'confidence': answer_result['confidence'],
            'retrieved_segments': retrieved_segments,
            'conversation_context': conversation_context,
            'processing_time': total_time,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update statistics
        self.total_qa_queries += 1
        self.total_qa_answers += 1
        
        self.logger.info(f"Contextual Q&A completed: {len(retrieved_segments)} segments, confidence: {answer_result['confidence']:.3f}")
        
        return response

    def ask_question_by_video_with_context(self, session_id: str, video_id: str, query: str, 
                                         n_results: int = 3) -> Dict[str, Any]:
        """
        Ask a question within a specific video with conversation context
        
        Args:
            session_id: Conversation session ID
            video_id: Video identifier
            query: Question to ask
            n_results: Number of segments to retrieve
            
        Returns:
            Complete Q&A response
        """
        self.logger.info(f"Processing contextual Q&A query for video {video_id}: '{query[:50]}...'")
        
        # Get conversation context
        conversation_context = self.conversation_memory.get_context_for_query(session_id, query)
        
        # Build contextualized query for retrieval
        contextualized_query = self._build_contextualized_query(query, conversation_context)
        
        # Retrieve relevant segments from specific video using contextualized query
        retrieved_segments = self.retrieval_system.search_by_video(
            video_id=video_id,
            query_text=contextualized_query,  # Use contextualized query
            n_results=n_results
        )
        
        # Generate answer with conversation context
        answer_result = self.answer_generator.generate_answer_with_context(
            query=query,
            retrieved_segments=retrieved_segments,
            conversation_context=conversation_context
        )
        
        # Store in conversation memory
        self.conversation_memory.add_message(
            session_id, 'user_question', query, 
            video_id=video_id,
            contextualized_query=contextualized_query
        )
        self.conversation_memory.add_message(
            session_id, 'system_answer', answer_result['answer'],
            citations=answer_result['citations'],
            confidence=answer_result['confidence'],
            retrieved_segments=retrieved_segments
        )
        
        # Create final response
        response = {
            'query': query,
            'contextualized_query': contextualized_query,
            'video_id': video_id,
            'answer': answer_result['answer'],
            'citations': answer_result['citations'],
            'confidence': answer_result['confidence'],
            'retrieved_segments': retrieved_segments,
            'conversation_context': conversation_context,
            'processing_time': answer_result['processing_time'],
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }
        
        return response

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        retrieval_stats = self.retrieval_system.get_collection_stats()
        conversation_stats = self.conversation_memory.get_session_stats()
        
        return {
            'total_qa_queries': self.total_qa_queries,
            'total_qa_answers': self.total_qa_answers,
            'retrieval_stats': retrieval_stats,
            'conversation_stats': conversation_stats,
            'answer_generator_stats': {
                'total_queries': self.answer_generator.total_queries,
                'total_answers': self.answer_generator.total_answers,
                'total_cost': self.answer_generator.total_cost
            }
        }


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Conversational Video Q&A System')
    parser.add_argument('--vector-db-path', default='vector_db',
                       help='Path to ChromaDB database')
    parser.add_argument('--model', default='intfloat/mmE5-mllama-11b-instruct',
                       help='Multimodal embedding model name')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a question')
    ask_parser.add_argument('query', help='Question to ask')
    ask_parser.add_argument('--image', help='Optional image path for multimodal search')
    ask_parser.add_argument('--results', type=int, default=5, help='Number of segments to retrieve')
    ask_parser.add_argument('--min-similarity', type=float, default=0.0, help='Minimum similarity threshold')
    
    # Ask video command
    ask_video_parser = subparsers.add_parser('ask-video', help='Ask a question within a specific video')
    ask_video_parser.add_argument('video_id', help='Video identifier')
    ask_video_parser.add_argument('query', help='Question to ask')
    ask_video_parser.add_argument('--results', type=int, default=3, help='Number of segments to retrieve')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show system statistics')
    
    args = parser.parse_args()
    
    # Initialize Q&A system
    try:
        qa_system = ConversationalVideoQASystem(
            args.vector_db_path,
            args.model,
            args.api_key,
            args.log_level
        )
    except Exception as e:
        print(f"Failed to initialize Q&A system: {str(e)}")
        return
    
    # Execute command
    if args.command == 'ask':
        # Create a new session for command line usage
        session_id = qa_system.conversation_memory.create_session()
        
        response = qa_system.ask_question_with_context(
            session_id,
            args.query,
            args.image,
            args.results,
            args.min_similarity
        )
        
        print(f"\nQuestion: {response['query']}")
        if response['contextualized_query'] != response['query']:
            print(f"Contextualized Query: {response['contextualized_query']}")
        print("=" * 80)
        print(f"Answer: {response['answer']}")
        print("\nCitations:")
        for i, citation in enumerate(response['citations'], 1):
            print(f"{i}. Video: {citation['video_id']}")
            print(f"   Time: {citation['timestamp_start']:.1f}s - {citation['timestamp_end']:.1f}s")
            print(f"   Relevance: {citation['similarity_score']:.3f}")
            print(f"   Content: {citation['summarized_text']}")
            print()
        
        print(f"Confidence: {response['confidence']:.3f}")
        print(f"Processing time: {response['processing_time']:.2f}s")
    
    elif args.command == 'ask-video':
        # Create a new session for command line usage
        session_id = qa_system.conversation_memory.create_session()
        
        response = qa_system.ask_question_by_video_with_context(
            session_id,
            args.video_id,
            args.query,
            args.results
        )
        
        print(f"\nQuestion: {response['query']}")
        if response['contextualized_query'] != response['query']:
            print(f"Contextualized Query: {response['contextualized_query']}")
        print(f"Video: {response['video_id']}")
        print("=" * 80)
        print(f"Answer: {response['answer']}")
        print("\nCitations:")
        for i, citation in enumerate(response['citations'], 1):
            print(f"{i}. Time: {citation['timestamp_start']:.1f}s - {citation['timestamp_end']:.1f}s")
            print(f"   Relevance: {citation['similarity_score']:.3f}")
            print(f"   Content: {citation['summarized_text']}")
            print()
        
        print(f"Confidence: {response['confidence']:.3f}")
        print(f"Processing time: {response['processing_time']:.2f}s")
    
    elif args.command == 'stats':
        stats = qa_system.get_system_stats()
        
        print("\nConversational Video Q&A System Statistics:")
        print("=" * 80)
        print(f"Total Q&A queries: {stats['total_qa_queries']}")
        print(f"Total Q&A answers: {stats['total_qa_answers']}")
        print(f"Total embeddings in DB: {stats['retrieval_stats'].get('total_embeddings', 'N/A')}")
        print(f"Total videos: {stats['retrieval_stats'].get('total_videos', 'N/A')}")
        print(f"Total conversation sessions: {stats['conversation_stats']['total_sessions']}")
        print(f"Active conversation sessions: {stats['conversation_stats']['active_sessions']}")
        print(f"Answer generator queries: {stats['answer_generator_stats']['total_queries']}")
        print(f"Answer generator cost: ${stats['answer_generator_stats']['total_cost']:.4f}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()