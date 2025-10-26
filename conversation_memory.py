import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import openai
import logging

class ConversationMemory:
    def __init__(self, max_context_tokens: int = 4000, summarization_threshold: int = 5):
        """
        Initialize Conversation Memory
        
        Args:
            max_context_tokens: Maximum tokens for context window
            summarization_threshold: Number of follow-ups before summarization
        """
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.max_context_tokens = max_context_tokens
        self.summarization_threshold = summarization_threshold
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client for summarization
        self.api_key = self._get_api_key()
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def _get_api_key(self) -> str:
        """Get OpenAI API key from environment"""
        import os
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return api_key
    
    def create_session(self) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'messages': [],
            'conversation_summary': '',
            'follow_up_count': 0,
            'total_tokens': 0
        }
        self.logger.info(f"Created new conversation session: {session_id}")
        return session_id
    
    def add_message(self, session_id: str, message_type: str, content: str, **kwargs) -> bool:
        """
        Add a message to the conversation
        
        Args:
            session_id: Session identifier
            message_type: 'user_question' or 'system_answer'
            content: Message content
            **kwargs: Additional message data (citations, confidence, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        if session_id not in self.sessions:
            self.logger.error(f"Session {session_id} not found")
            return False
        
        message = {
            'type': message_type,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        self.sessions[session_id]['messages'].append(message)
        self.sessions[session_id]['last_updated'] = datetime.now().isoformat()
        
        # Increment follow-up count for user questions
        if message_type == 'user_question':
            self.sessions[session_id]['follow_up_count'] += 1
            
            # Check if we need to summarize
            if self.sessions[session_id]['follow_up_count'] % self.summarization_threshold == 0:
                self._summarize_conversation(session_id)
        
        self.logger.info(f"Added {message_type} to session {session_id}")
        return True
    
    def _summarize_conversation(self, session_id: str) -> str:
        """
        Summarize conversation when threshold is reached
        
        Args:
            session_id: Session identifier
            
        Returns:
            Generated summary
        """
        try:
            session = self.sessions[session_id]
            messages = session['messages']
            
            # Build conversation text for summarization
            conversation_text = ""
            for msg in messages:
                if msg['type'] == 'user_question':
                    conversation_text += f"Q: {msg['content']}\n"
                else:
                    conversation_text += f"A: {msg['content']}\n"
            
            # Create summarization prompt
            prompt = f"""Summarize the following conversation about educational video content. 
            Focus on key topics, concepts, and important information discussed.

            Conversation:
            {conversation_text}

            Provide a concise summary that captures:
            1. Main topics discussed
            2. Key concepts and definitions
            3. Important relationships between concepts
            4. Any specific video segments or timestamps mentioned

            Summary:"""
            
            # Generate summary using GPT
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            session['conversation_summary'] = summary
            
            self.logger.info(f"Summarized conversation for session {session_id}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to summarize conversation: {str(e)}")
            return ""
    
    def get_context_for_query(self, session_id: str, new_query: str) -> str:
        """
        Get relevant context for a new query
        
        Args:
            session_id: Session identifier
            new_query: New user query
            
        Returns:
            Context string for the query
        """
        if session_id not in self.sessions:
            return ""
        
        session = self.sessions[session_id]
        
        # If we have a summary, use it as base context
        context_parts = []
        if session['conversation_summary']:
            context_parts.append(f"Previous conversation summary:\n{session['conversation_summary']}\n")
        
        # Add recent messages (last 3 Q&A pairs)
        recent_messages = session['messages'][-6:]  # Last 3 Q&A pairs
        if recent_messages:
            context_parts.append("Recent conversation:")
            for msg in recent_messages:
                if msg['type'] == 'user_question':
                    context_parts.append(f"Q: {msg['content']}")
                else:
                    context_parts.append(f"A: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """
        Get conversation history for display
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session history data
        """
        if session_id not in self.sessions:
            return {'messages': [], 'summary': ''}
        
        session = self.sessions[session_id]
        return {
            'messages': session['messages'],
            'summary': session['conversation_summary'],
            'follow_up_count': session['follow_up_count'],
            'created_at': session['created_at']
        }
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a conversation session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info(f"Cleared session {session_id}")
            return True
        return False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about all sessions"""
        return {
            'total_sessions': len(self.sessions),
            'active_sessions': len([s for s in self.sessions.values() 
                                  if s['follow_up_count'] > 0])
        }