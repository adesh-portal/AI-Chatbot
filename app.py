"""
Enhanced Flask Chatbot Application - Fixed Version
Advanced chatbot with dynamic entity extraction, adaptive confidence, and knowledge integration.
"""

import os
import json
import pickle
import random
import secrets
import numpy as np
from flask import Flask, request, jsonify, session, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import threading
import time
import re
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from contextlib import contextmanager
# import spacy  # Optional for advanced NER (kept disabled)
import requests
from typing import Dict, List, Optional, Tuple
# Utilities for network info
import socket
# import wikipedia  # Commented out - not needed for core model
# import dateutil.parser as date_parser  # Commented out - not needed for core model
# from dotenv import load_dotenv  # Commented out - not needed for core model

# Load environment variables
# load_dotenv()  # Commented out for simplicity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def get_lan_ip() -> str:
    """Best-effort retrieval of local LAN IP for display."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Does not need to be reachable; used to pick outbound interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

# Configuration
class Config:
    def __init__(self):
        # Security
        self.SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
        
        # File paths
        self.MODEL_DIR = os.environ.get('MODEL_DIR', 'models')
        # Prefer new Keras format if available; fallback to legacy .h5
        preferred_keras = os.path.join(self.MODEL_DIR, 'chatbot_v2_model.keras')
        legacy_h5 = os.path.join(self.MODEL_DIR, 'chatbot_v2_model.h5')
        self.MODEL_FILE = preferred_keras if os.path.exists(preferred_keras) else legacy_h5
        self.TOKENIZER_FILE = os.path.join(self.MODEL_DIR, 'tokenizer.pkl')
        self.LABEL_ENCODER_FILE = os.path.join(self.MODEL_DIR, 'label_encoder.pkl')
        self.INTENTS_FILE = os.environ.get('INTENTS_FILE', 'intents.json')
        
        # Model parameters
        self.MAX_SEQUENCE_LENGTH = int(os.environ.get('MAX_SEQUENCE_LENGTH', '50'))
        self.BASE_CONFIDENCE_THRESHOLD = float(os.environ.get('BASE_CONFIDENCE_THRESHOLD', '0.55'))
        # Sampling parameters (ChatGPT-like)
        self.TEMPERATURE = float(os.environ.get('TEMPERATURE', '0.9'))  # <1.0 more deterministic
        self.TOP_P = float(os.environ.get('TOP_P', '0.9'))              # nucleus sampling
        self.TOP_K = int(os.environ.get('TOP_K', '5'))                  # limit candidates
        self.RESPONSE_TEMPERATURE = float(os.environ.get('RESPONSE_TEMPERATURE', '0.7'))
        # Response length/expansion
        self.RESPONSE_COMBINE = int(os.environ.get('RESPONSE_COMBINE', '2'))  # combine up to N variations
        self.ADAPTIVE_THRESHOLD_RANGE = (0.15, 0.4)  # Min/Max for adaptive threshold
        
        # Context settings
        self.MAX_CONTEXT_MESSAGES = int(os.environ.get('MAX_CONTEXT_MESSAGES', '10'))
        self.CONTEXT_WINDOW_SIZE = int(os.environ.get('CONTEXT_WINDOW_SIZE', '5'))
        
        # Memory management
        self.MAX_USERS = int(os.environ.get('MAX_USERS', '1000'))
        self.CLEANUP_INTERVAL = int(os.environ.get('CLEANUP_INTERVAL', '3600'))  # 1 hour
        self.USER_TIMEOUT = int(os.environ.get('USER_TIMEOUT', '86400'))  # 24 hours
        
        # External APIs
        self.ENABLE_WIKIPEDIA = os.environ.get('ENABLE_WIKIPEDIA', 'True').lower() == 'true'
        self.ENABLE_NEWS_API = os.environ.get('ENABLE_NEWS_API', 'False').lower() == 'true'
        self.NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')
        self.ENABLE_CURRENTS_API = os.environ.get('ENABLE_CURRENTS_API', 'False').lower() == 'true'
        self.CURRENTS_API_KEY = os.environ.get('CURRENTS_API_KEY', '')
        
        # Server settings
        self.DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        self.HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
        self.PORT = int(os.environ.get('FLASK_PORT', '5000'))

config = Config()

# Dynamic Entity Extractor - Simplified for core model focus
class EntityExtractor:
    def __init__(self):
        # Commented out spaCy for simplicity - using pattern-based extraction only
        # try:
        #     # Load spaCy model (install with: python -m spacy download en_core_web_sm)
        #     self.nlp = spacy.load("en_core_web_sm")
        #     self.enabled = True
        #     logger.info("SpaCy model loaded successfully")
        # except OSError:
        #     logger.warning("SpaCy model not found. Entity extraction will be limited.")
        #     self.nlp = None
        #     self.enabled = False
        
        self.nlp = None
        self.enabled = False
        logger.info("Using pattern-based entity extraction only")
        
        # Fallback entity patterns
        self.entity_patterns = {
            'PERSON': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.? [A-Z][a-z]+\b'  # Title Name
            ],
            'ORG': [
                r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|University|College)\b'
            ],
            'GPE': [  # Geopolitical entities
                r'\b[A-Z][a-z]+ (?:City|County|State|Country)\b'
            ],
            'DATE': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{2,4}\b',
                r'\b(?:today|tomorrow|yesterday|next week|last week|this month)\b'
            ],
            'TECH': [
                r'\b(?:AI|ML|neural network|machine learning|deep learning|NLP|computer vision)\b',
                r'\b(?:Python|JavaScript|Java|C\+\+|React|Flask|Django)\b',
                r'\bneuromorphic chip[s]?\b'
            ]
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities from text using spaCy and fallback patterns"""
        entities = defaultdict(list)
        
        if self.enabled and self.nlp:
            # Use spaCy for entity extraction
            doc = self.nlp(text)
            for ent in doc.ents:
                entities[ent.label_].append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.8  # spaCy confidence
                })
        
        # Fallback pattern matching
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities[entity_type].append({
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.6  # Pattern-based confidence
                    })
        
        return dict(entities)
    
    def extract_intent_entities(self, text: str, intent: str) -> Dict:
        """Extract intent-specific entities"""
        all_entities = self.extract_entities(text)
        
        # Intent-specific entity mapping
        intent_mappings = {
            'book_recommendation': ['PERSON', 'ORG', 'WORK_OF_ART'],
            'weather': ['GPE', 'DATE'],
            'news': ['GPE', 'ORG', 'PERSON', 'DATE'],
            'tech_info': ['TECH', 'ORG'],
            'neuromorphic_chips': ['TECH', 'ORG']
        }
        
        relevant_entities = {}
        if intent in intent_mappings:
            for entity_type in intent_mappings[intent]:
                if entity_type in all_entities:
                    relevant_entities[entity_type] = all_entities[entity_type]
        
        return relevant_entities

# Knowledge Integration System
class KnowledgeIntegrator:
    def __init__(self, config):
        self.config = config
        self.cache = {}
        self.cache_timeout = 3600  # 1 hour
    
    def get_wikipedia_summary(self, query: str, sentences: int = 2) -> Optional[str]:
        """Get Wikipedia summary for a query."""
        if not self.config.ENABLE_WIKIPEDIA:
            return None
        cache_key = f"wiki_{query}_{sentences}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        try:
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(query)
            r = requests.get(url, timeout=5)
            if r.status_code != 200:
                return None
            data = r.json()
            extract = data.get('extract')
            if not extract:
                return None
            parts = extract.split('. ')
            snippet = '. '.join(parts[:max(1, sentences)]).strip()
            self._cache_result(cache_key, snippet)
            return snippet
        except Exception as e:
            logger.error(f"Wikipedia fetch failed: {e}")
            return None
        
        # if not self.config.ENABLE_WIKIPEDIA:
        #     return None
        
        # cache_key = f"wiki_{query}_{sentences}"
        # if self._is_cached(cache_key):
        #     return self.cache[cache_key]['data']
        
        # try:
        #     # Search for the topic
        #     search_results = wikipedia.search(query, results=3)
        #     if not search_results:
        #         return None
            
        #     # Get summary of the first result
        #     summary = wikipedia.summary(search_results[0], sentences=sentences)
        #     self._cache_result(cache_key, summary)
        #     return summary
            
        # except wikipedia.exceptions.DisambiguationError as e:
        #     # Handle disambiguation by taking the first option
        #     try:
        #         summary = wikipedia.summary(e.options[0], sentences=sentences)
        #         self._cache_result(cache_key, summary)
        #         return summary
        #     except:
        #         return None
        # except:
        #     return None
    
    def get_recent_news(self, query: str, limit: int = 3) -> List[Dict]:
        """Get recent news articles via NewsAPI."""
        if not self.config.ENABLE_NEWS_API or not self.config.NEWS_API_KEY:
            return []
        cache_key = f"news_{query}_{limit}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'sortBy': 'publishedAt',
                'pageSize': limit,
                'language': 'en'
            }
            headers = { 'X-Api-Key': self.config.NEWS_API_KEY }
            response = requests.get(url, params=params, headers=headers, timeout=6)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            processed_articles = []
            for article in articles:
                processed_articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': (article.get('source') or {}).get('name', '')
                })
            self._cache_result(cache_key, processed_articles)
            return processed_articles
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
        
        # if not self.config.ENABLE_NEWS_API or not self.config.NEWS_API_KEY:
        #     return []
        
        # cache_key = f"news_{query}_{limit}"
        # if self._is_cached(cache_key):
        #     return self.cache[cache_key]['data']
        
        # try:
        #     url = "https://newsapi.org/v2/everything"
        #     params = {
        #         'q': query,
        #         'sortBy': 'publishedAt',
        #         'pageSize': limit,
        #         'apiKey': self.config.NEWS_API_KEY
        #     }
        
        #     response = requests.get(url, params=params, timeout=5)
        #     response.raise_for_status()
        
        #     articles = response.json().get('articles', [])
        #     processed_articles = []
        
        #     for article in articles:
        #         processed_articles.append({
        #             'title': article.get('title', ''),
        #             'description': article.get('description', ''),
        #             'url': article.get('url', ''),
        #             'published_at': article.get('publishedAt', '')
        #         })
        
        #     self._cache_result(cache_key, processed_articles)
        #     return processed_articles
        
        # except Exception as e:
        #     logger.error(f"Error fetching news: {e}")
        #     return []
    
    def get_currents_news(self, query: str, limit: int = 3) -> List[Dict]:
        """Get recent news articles from Currents API."""
        if not self.config.ENABLE_CURRENTS_API or not self.config.CURRENTS_API_KEY:
            return []
        cache_key = f"currents_{query}_{limit}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        try:
            url = "https://api.currentsapi.services/v1/search"
            params = {
                'keywords': query,
                'limit': limit,
                'apiKey': self.config.CURRENTS_API_KEY
            }
            response = requests.get(url, params=params, timeout=6)
            response.raise_for_status()
            data = response.json()
            articles = data.get('news', [])
            processed_articles = []
            for article in articles:
                processed_articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('published', ''),
                    'source': article.get('author', 'Currents API')
                })
            self._cache_result(cache_key, processed_articles)
            return processed_articles
        except Exception as e:
            logger.error(f"Error fetching news from Currents: {e}")
            return []
        
        # if not self.config.ENABLE_CURRENTS_API or not self.config.CURRENTS_API_KEY:
        #     return []
        
        # cache_key = f"currents_{query}_{limit}"
        # if self._is_cached(cache_key):
        #     return self.cache[cache_key]['data']
        
        # try:
        #     url = "https://api.currentsapi.services/v1/search"
        #     params = {
        #         'keywords': query,
        #         'limit': limit,
        #         'apiKey': self.config.CURRENTS_API_KEY
        #     }
        
        #     response = requests.get(url, params=params, timeout=5)
        #     response.raise_for_status()
        
        #     data = response.json()
        #     articles = data.get('news', [])
        #     processed_articles = []
        
        #     for article in articles:
        #         processed_articles.append({
        #             'title': article.get('title', ''),
        #             'description': article.get('description', ''),
        #             'url': article.get('url', ''),
        #             'published_at': article.get('published', ''),
        #             'source': article.get('author', 'Currents API')
        #         })
        
        #     self._cache_result(cache_key, processed_articles)
        #     return processed_articles
        
        # except Exception as e:
        #     logger.error(f"Error fetching news from Currents: {e}")
        #     return []

    def try_knowledge_integration(self, topic: str) -> Optional[str]:
        """Try to get knowledge from external sources."""
        try:
            # Try Wikipedia first
            wiki_summary = self.get_wikipedia_summary(topic, sentences=2)
            if wiki_summary:
                return f"According to Wikipedia: {wiki_summary}"
            # Try news sources for current topics
            current_keywords = ['news', 'recent', 'latest', 'today', 'current']
            if any(keyword in topic.lower() for keyword in current_keywords):
                # Try NewsAPI first
                if self.config.ENABLE_NEWS_API and self.config.NEWS_API_KEY:
                    news_articles = self.get_recent_news(topic, limit=1)
                    if news_articles:
                        article = news_articles[0]
                        return f"Recent news: {article['title']} — {article['description'][:220]}"
                # Fallback to Currents API
                if self.config.ENABLE_CURRENTS_API and self.config.CURRENTS_API_KEY:
                    currents_articles = self.get_currents_news(topic, limit=1)
                    if currents_articles:
                        article = currents_articles[0]
                        return f"Recent news: {article['title']} — {article['description'][:220]}"
            return None
        except Exception as e:
            logger.error(f"Error in knowledge integration: {e}")
            return None
    
    def _is_cached(self, key: str) -> bool:
        """Check if result is cached and not expired"""
        if key not in self.cache:
            return False
        
        cached_time = self.cache[key]['timestamp']
        return (datetime.now() - cached_time).seconds < self.cache_timeout
    
    def _cache_result(self, key: str, data):
        """Cache a result with timestamp"""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }

# Enhanced Message Processor
class EnhancedMessageProcessor:
    def __init__(self, entity_extractor, knowledge_integrator):
        self.entity_extractor = entity_extractor
        self.knowledge_integrator = knowledge_integrator
        
        self.spelling_corrections = {
            'recomend': 'recommend', 'auther': 'author', 'writter': 'writer',
            'genere': 'genre', 'boook': 'book', 'novle': 'novel',
            'neuromorphic': 'neuromorphic', 'artficial': 'artificial',
            'nural': 'neural'  # common misspelling
        }
        
        self.follow_up_patterns = [
            r'tell me more', r'what about that', r'can you elaborate',
            r'more info', r'go on', r'continue', r'explain further'
        ]
        
        self.question_patterns = [
            r'^(?:what|who|when|where|why|how)\s',
            r'^(?:can you|could you|would you)\s',
            r'\?$'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        text = text.lower().strip()
        
        # Fix spelling mistakes
        for mistake, correction in self.spelling_corrections.items():
            text = re.sub(r'\b' + mistake + r'\b', correction, text)
        
        # Expand contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "'s": " is"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Clean special characters but preserve important punctuation
        text = re.sub(r'[^\w\s\?\!\.\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_follow_up(self, message: str) -> bool:
        """Check if message is a follow-up"""
        return any(re.search(pattern, message.lower()) for pattern in self.follow_up_patterns)
    
    def is_question(self, message: str) -> bool:
        """Check if message is a question"""
        return any(re.search(pattern, message.lower()) for pattern in self.question_patterns)
    
    def analyze_message(self, message: str, intent: str = None) -> Dict:
        """Comprehensive message analysis"""
        processed = self.preprocess_text(message)
        entities = self.entity_extractor.extract_entities(message)
        
        if intent:
            intent_entities = self.entity_extractor.extract_intent_entities(message, intent)
            entities.update(intent_entities)
        
        analysis = {
            'original': message,
            'processed': processed,
            'entities': entities,
            'is_question': self.is_question(message),
            'is_follow_up': self.is_follow_up(message),
            'complexity': self._assess_complexity(processed),
            'topics': self._extract_topics(entities)
        }
        
        return analysis

    def split_multi_questions(self, message: str) -> List[str]:
        """Split a message into multiple sub-questions if the user asked more than one.
        Heuristics: split on '?', ' and ', ' & ', ' also ', '.', while preserving meaning.
        """
        if not message:
            return []
        # Primary split on question marks
        parts = re.split(r"\?+\s*", message)
        # Further split on connectors if segments are long
        refined = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # Split on common connectors only if long enough
            if len(p.split()) > 8:
                subparts = re.split(r"\b(?: and | & | also | plus | as well as )\b", p, flags=re.IGNORECASE)
                for sp in subparts:
                    sp = sp.strip(' .,;')
                    if len(sp.split()) >= 3:
                        refined.append(sp)
            else:
                refined.append(p.strip(' .,;'))
        # Keep only meaningful segments
        refined = [s for s in refined if len(s) >= 3]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in refined:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique
    
    def generate_fallback_response(self, analysis: Dict) -> str:
        """Generate contextual fallback response"""
        if analysis['is_question']:
            if analysis['topics']:
                topic = analysis['topics'][0]
                return f"I'm not sure about {topic} specifically. Could you provide more details or ask in a different way?"
            else:
                return "That's an interesting question! Could you provide more context so I can help better?"
        
        elif analysis['entities']:
            entities_text = ', '.join([
                entity['text'] for entity_list in analysis['entities'].values() 
                for entity in entity_list
            ][:3])
            return f"I see you mentioned {entities_text}. Could you clarify what you'd like to know?"
        
        else:
            return random.choice([
                "I'm not sure I understand. Could you please rephrase that?",
                "Could you provide more details about what you're looking for?",
                "That's interesting! Could you elaborate on what you mean?",
                "I'm still learning. Can you help me understand better?"
            ])
    
    def _assess_complexity(self, text: str) -> float:
        """Assess message complexity (0.0 to 1.0)"""
        factors = {
            'length': min(len(text.split()) / 20, 1.0),
            'questions': len(re.findall(r'\?', text)) * 0.2,
            'technical_terms': len(re.findall(r'\b(?:neuromorphic|algorithm|machine learning|AI)\b', text, re.IGNORECASE)) * 0.3
        }
        return min(sum(factors.values()), 1.0)
    
    def _extract_topics(self, entities: Dict) -> List[str]:
        """Extract main topics from entities"""
        topics = []
        
        # Map entity types to topics
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                topics.append(entity['text'].lower())
        
        return list(set(topics))

# Advanced Reasoning Engine with Probability Chains
class ReasoningEngine:
    def __init__(self):
        self.probability_chains = defaultdict(list)
        self.decision_trees = {}
        self.uncertainty_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        self.reasoning_patterns = {
            'causal': ['because', 'since', 'due to', 'as a result'],
            'conditional': ['if', 'when', 'unless', 'provided that'],
            'comparative': ['better', 'worse', 'similar', 'different', 'compared to'],
            'temporal': ['before', 'after', 'during', 'while', 'then', 'now']
        }
    
    def analyze_reasoning_patterns(self, text: str) -> Dict[str, float]:
        """Analyze reasoning patterns in text"""
        text_lower = text.lower()
        pattern_scores = {}
        
        for pattern_type, keywords in self.reasoning_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            pattern_scores[pattern_type] = min(score / len(keywords), 1.0)
        
        return pattern_scores
    
    def build_probability_chain(self, predictions: List[Dict], context: List[Dict] = None) -> Dict:
        """Build a probability chain for decision making"""
        if not predictions:
            return {'confidence': 0.0, 'reasoning': 'No predictions available'}
        
        # Primary prediction
        primary = predictions[0]
        primary_confidence = primary['confidence']
        
        # Secondary predictions for comparison
        secondary = predictions[1:3] if len(predictions) > 1 else []
        
        # Calculate probability chain
        chain = {
            'primary': primary,
            'secondary': secondary,
            'confidence_gap': 0.0,
            'uncertainty_level': 'low',
            'reasoning_strength': 0.0,
            'context_support': 0.0
        }
        
        # Calculate confidence gap
        if secondary:
            chain['confidence_gap'] = primary_confidence - secondary[0]['confidence']
        
        # Determine uncertainty level
        if primary_confidence >= self.uncertainty_thresholds['high']:
            chain['uncertainty_level'] = 'low'
        elif primary_confidence >= self.uncertainty_thresholds['medium']:
            chain['uncertainty_level'] = 'medium'
        else:
            chain['uncertainty_level'] = 'high'
        
        # Calculate reasoning strength
        chain['reasoning_strength'] = self._calculate_reasoning_strength(primary, secondary)
        
        # Calculate context support
        if context:
            chain['context_support'] = self._calculate_context_support(primary, context)
        
        # Overall confidence with reasoning factors
        overall_confidence = self._calculate_overall_confidence(chain)
        chain['overall_confidence'] = overall_confidence
        
        return chain
    
    def _calculate_reasoning_strength(self, primary: Dict, secondary: List[Dict]) -> float:
        """Calculate the strength of reasoning based on prediction patterns"""
        if not secondary:
            return primary['confidence']
        
        # Gap between primary and secondary
        gap = primary['confidence'] - secondary[0]['confidence']
        
        # Strong reasoning if there's a clear winner
        if gap > 0.3:
            return primary['confidence'] + 0.1
        elif gap > 0.1:
            return primary['confidence']
        else:
            return primary['confidence'] - 0.1
    
    def _calculate_context_support(self, prediction: Dict, context: List[Dict]) -> float:
        """Calculate how well the prediction is supported by context"""
        if not context:
            return 0.0
        
        # Check for similar intents in recent context
        recent_intents = [exchange.get('intent', '') for exchange in context[-3:]]
        intent_support = sum(1 for intent in recent_intents if intent == prediction['intent'])
        
        # Check for topic consistency
        recent_topics = []
        for exchange in context[-3:]:
            if 'topics' in exchange:
                recent_topics.extend(exchange['topics'])
        
        # This is a simplified topic support calculation
        topic_support = 0.5  # Placeholder for more sophisticated topic analysis
        
        # Combine support factors
        context_support = (intent_support * 0.6 + topic_support * 0.4) / len(context)
        return min(context_support, 1.0)
    
    def _calculate_overall_confidence(self, chain: Dict) -> float:
        """Calculate overall confidence considering all factors"""
        base_confidence = chain['primary']['confidence']
        reasoning_bonus = chain['reasoning_strength'] * 0.1
        context_bonus = chain['context_support'] * 0.05
        
        # Uncertainty penalty
        uncertainty_penalty = 0.0
        if chain['uncertainty_level'] == 'high':
            uncertainty_penalty = 0.1
        elif chain['uncertainty_level'] == 'medium':
            uncertainty_penalty = 0.05
        
        overall = base_confidence + reasoning_bonus + context_bonus - uncertainty_penalty
        return max(0.0, min(1.0, overall))
    
    def make_decision(self, probability_chain: Dict, threshold: float) -> Dict:
        """Make a decision based on probability chain and threshold"""
        overall_confidence = probability_chain['overall_confidence']
        
        decision = {
            'action': 'proceed' if overall_confidence >= threshold else 'fallback',
            'confidence': overall_confidence,
            'reasoning': self._generate_reasoning_explanation(probability_chain),
            'uncertainty_factors': self._identify_uncertainty_factors(probability_chain),
            'recommendations': self._generate_recommendations(probability_chain, threshold)
        }
        
        return decision
    
    def _generate_reasoning_explanation(self, chain: Dict) -> str:
        """Generate human-readable reasoning explanation"""
        primary = chain['primary']
        confidence = chain['overall_confidence']
        uncertainty = chain['uncertainty_level']
        
        explanations = []
        
        if confidence >= 0.8:
            explanations.append("High confidence prediction")
        elif confidence >= 0.6:
            explanations.append("Moderate confidence prediction")
        else:
            explanations.append("Low confidence prediction")
        
        if chain['confidence_gap'] > 0.2:
            explanations.append("Clear preference over alternatives")
        elif chain['confidence_gap'] > 0.1:
            explanations.append("Slight preference over alternatives")
        else:
            explanations.append("Close competition with alternatives")
        
        if chain['context_support'] > 0.7:
            explanations.append("Strong context support")
        elif chain['context_support'] > 0.4:
            explanations.append("Moderate context support")
        else:
            explanations.append("Limited context support")
        
        return "; ".join(explanations)
    
    def _identify_uncertainty_factors(self, chain: Dict) -> List[str]:
        """Identify factors contributing to uncertainty"""
        factors = []
        
        if chain['overall_confidence'] < 0.6:
            factors.append("Low prediction confidence")
        
        if chain['confidence_gap'] < 0.1:
            factors.append("Close competition between predictions")
        
        if chain['context_support'] < 0.3:
            factors.append("Limited context support")
        
        if chain['uncertainty_level'] == 'high':
            factors.append("High uncertainty level")
        
        return factors
    
    def _generate_recommendations(self, chain: Dict, threshold: float) -> List[str]:
        """Generate recommendations for improving decision quality"""
        recommendations = []
        
        if chain['overall_confidence'] < threshold:
            recommendations.append("Consider asking for clarification")
            recommendations.append("Provide more context in the question")
        
        if chain['context_support'] < 0.5:
            recommendations.append("Build more conversation context")
        
        if chain['confidence_gap'] < 0.15:
            recommendations.append("Be more specific about the intent")
        
        return recommendations

# Enhanced Adaptive Confidence Manager with Advanced Reasoning
class EnhancedConfidenceManager:
    def __init__(self, base_threshold: float, threshold_range: Tuple[float, float]):
        self.base_threshold = base_threshold
        self.min_threshold, self.max_threshold = threshold_range
        self.user_performance = defaultdict(lambda: {
            'correct': 0, 'total': 0, 'recent_correct': 0, 'recent_total': 0,
            'confidence_history': deque(maxlen=20),
            'response_times': deque(maxlen=10),
            'preferred_intents': defaultdict(int)
        })
        self.global_patterns = defaultdict(lambda: {'success_rate': 0.5, 'frequency': 0})
        self.context_confidence = defaultdict(float)
    
    def get_threshold(self, user_id: str, message_analysis: Dict, context: List[Dict] = None) -> float:
        """Calculate enhanced adaptive confidence threshold with multi-factor analysis"""
        # Base factors
        complexity_factor = message_analysis['complexity']
        question_factor = 0.15 if message_analysis['is_question'] else 0.0
        entity_factor = min(len(message_analysis['entities']) * 0.08, 0.25)
        
        # User-specific factors
        user_perf = self.user_performance[user_id]
        recent_success = self._calculate_recent_success(user_perf)
        confidence_trend = self._calculate_confidence_trend(user_perf)
        intent_familiarity = self._calculate_intent_familiarity(user_perf, message_analysis)
        
        # Context-aware factors
        context_factor = self._analyze_context_confidence(context) if context else 0.0
        temporal_factor = self._analyze_temporal_patterns(user_perf)
        
        # Global pattern analysis
        pattern_factor = self._analyze_global_patterns(message_analysis)
        
        # Multi-dimensional confidence calculation
        threshold = self.base_threshold
        
        # Primary adjustments
        threshold += complexity_factor * 0.12  # Complex messages need higher confidence
        threshold -= question_factor  # Questions get more lenient thresholds
        threshold -= entity_factor  # Rich entities suggest clearer intent
        threshold -= recent_success * 0.1  # Recent success lowers threshold
        threshold += confidence_trend * 0.05  # Trending confidence affects threshold
        threshold -= intent_familiarity * 0.08  # Familiar intents get lower thresholds
        threshold += context_factor  # Context uncertainty increases threshold
        threshold += temporal_factor  # Time-based patterns
        threshold += pattern_factor  # Global pattern analysis
        
        # Apply confidence smoothing
        threshold = self._apply_confidence_smoothing(threshold, user_perf)
        
        # Clamp to allowed range with dynamic bounds
        dynamic_min = max(self.min_threshold, self.base_threshold - 0.1)
        dynamic_max = min(self.max_threshold, self.base_threshold + 0.15)
        threshold = max(dynamic_min, min(dynamic_max, threshold))
        
        # Store for analysis
        user_perf['confidence_history'].append(threshold)
        
        return round(threshold, 4)
    
    def _calculate_recent_success(self, user_perf: Dict) -> float:
        """Calculate recent success rate with exponential weighting"""
        if user_perf['recent_total'] == 0:
            return 0.0
        
        # Exponential decay for recent performance
        recent_rate = user_perf['recent_correct'] / user_perf['recent_total']
        return recent_rate
    
    def _calculate_confidence_trend(self, user_perf: Dict) -> float:
        """Calculate confidence trend over recent interactions"""
        history = list(user_perf['confidence_history'])
        if len(history) < 3:
            return 0.0
        
        # Linear regression on recent confidence values
        recent = history[-5:] if len(history) >= 5 else history
        if len(recent) < 2:
            return 0.0
        
        # Simple trend calculation
        trend = (recent[-1] - recent[0]) / len(recent)
        return max(-0.1, min(0.1, trend))  # Clamp trend influence
    
    def _calculate_intent_familiarity(self, user_perf: Dict, analysis: Dict) -> float:
        """Calculate how familiar the user is with predicted intents"""
        if not analysis.get('topics'):
            return 0.0
        
        familiarity_score = 0.0
        total_topics = len(analysis['topics'])
        
        for topic in analysis['topics']:
            # Check if user has interacted with similar topics before
            topic_frequency = user_perf['preferred_intents'].get(topic, 0)
            familiarity_score += min(topic_frequency / 10, 1.0)  # Normalize to 0-1
        
        return familiarity_score / total_topics if total_topics > 0 else 0.0
    
    def _analyze_context_confidence(self, context: List[Dict]) -> float:
        """Analyze confidence based on conversation context"""
        if not context:
            return 0.0
        
        # Recent context confidence analysis
        recent_context = context[-3:] if len(context) >= 3 else context
        avg_confidence = sum(exchange.get('confidence', 0.5) for exchange in recent_context) / len(recent_context)
        
        # Context consistency analysis
        intents = [exchange.get('intent', '') for exchange in recent_context]
        intent_consistency = len(set(intents)) / len(intents) if intents else 1.0
        
        # Context factor: higher if recent confidence is low or inconsistent
        context_factor = (1 - avg_confidence) * 0.1 + (1 - intent_consistency) * 0.05
        return min(context_factor, 0.2)
    
    def _analyze_temporal_patterns(self, user_perf: Dict) -> float:
        """Analyze temporal patterns in user interactions"""
        response_times = list(user_perf['response_times'])
        if len(response_times) < 3:
            return 0.0
        
        # Analyze response time patterns
        avg_response_time = sum(response_times) / len(response_times)
        time_variance = sum((t - avg_response_time) ** 2 for t in response_times) / len(response_times)
        
        # Users with consistent response times might be more predictable
        consistency_factor = max(0, 0.1 - time_variance * 0.01)
        return consistency_factor
    
    def _analyze_global_patterns(self, analysis: Dict) -> float:
        """Analyze global patterns across all users"""
        # This would typically use more sophisticated pattern analysis
        # For now, we'll use a simplified approach
        complexity = analysis.get('complexity', 0.5)
        entity_count = len(analysis.get('entities', {}))
        
        # Complex messages with many entities tend to be more predictable
        pattern_score = (complexity * 0.1) + (entity_count * 0.02)
        return min(pattern_score, 0.1)
    
    def _apply_confidence_smoothing(self, threshold: float, user_perf: Dict) -> float:
        """Apply smoothing to prevent wild threshold swings"""
        history = list(user_perf['confidence_history'])
        if len(history) < 2:
            return threshold
    
        # Exponential moving average
        alpha = 0.3  # Smoothing factor
        last_threshold = history[-1]
        smoothed = alpha * threshold + (1 - alpha) * last_threshold
        
        return smoothed
    
    def update_performance(self, user_id: str, was_correct: bool, response_time: float = None, intent: str = None):
        """Update user performance metrics with enhanced tracking"""
        perf = self.user_performance[user_id]
        perf['total'] += 1
        perf['recent_total'] += 1
        
        if was_correct:
            perf['correct'] += 1
            perf['recent_correct'] += 1
        
        if response_time is not None:
            perf['response_times'].append(response_time)
        
        if intent:
            perf['preferred_intents'][intent] += 1
        
        # Update global patterns
        if intent:
            pattern = self.global_patterns[intent]
            pattern['frequency'] += 1
            if was_correct:
                pattern['success_rate'] = (pattern['success_rate'] * (pattern['frequency'] - 1) + 1) / pattern['frequency']
            else:
                pattern['success_rate'] = (pattern['success_rate'] * (pattern['frequency'] - 1)) / pattern['frequency']
        
        # Maintain sliding window for recent performance
        if perf['recent_total'] > 20:
            # Remove oldest entry
            if perf['recent_correct'] > 0 and random.random() < (perf['recent_correct'] / perf['recent_total']):
                perf['recent_correct'] -= 1
            perf['recent_total'] -= 1
        
        # Decay older performance data
        if perf['total'] > 100:
            decay_factor = 0.95
            perf['correct'] = int(perf['correct'] * decay_factor)
            perf['total'] = int(perf['total'] * decay_factor)
    
    def get_user_insights(self, user_id: str) -> Dict:
        """Get detailed insights about user interaction patterns"""
        perf = self.user_performance[user_id]
        
        return {
            'total_interactions': perf['total'],
            'success_rate': perf['correct'] / perf['total'] if perf['total'] > 0 else 0,
            'recent_success_rate': perf['recent_correct'] / perf['recent_total'] if perf['recent_total'] > 0 else 0,
            'avg_confidence': sum(perf['confidence_history']) / len(perf['confidence_history']) if perf['confidence_history'] else 0,
            'preferred_intents': dict(perf['preferred_intents']),
            'avg_response_time': sum(perf['response_times']) / len(perf['response_times']) if perf['response_times'] else 0,
            'confidence_trend': self._calculate_confidence_trend(perf)
        }

# Enhanced Conversation Context Manager
class EnhancedConversationContext:
    def __init__(self, max_history=10, max_users=1000, cleanup_interval=3600):
        self.contexts = defaultdict(lambda: deque(maxlen=max_history))
        self.user_preferences = defaultdict(dict)
        self.user_last_activity = defaultdict(datetime)
        self.max_users = max_users
        self.cleanup_interval = cleanup_interval
        self._lock = threading.Lock()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def add_exchange(self, user_id: str, analysis: Dict, intent: str, response: str, 
                    confidence: float, knowledge_used: bool = False):
        """Add conversation exchange to context"""
        with self._lock:
            # Memory management
            if len(self.contexts) >= self.max_users:
                self._cleanup_inactive_users()
            
            exchange = {
                'message': analysis['original'],
                'processed': analysis['processed'],
                'entities': analysis['entities'],
                'topics': analysis['topics'],
                'intent': intent,
                'response': response,
                'confidence': confidence,
                'knowledge_used': knowledge_used,
                'timestamp': datetime.now().isoformat(),
                'is_question': analysis['is_question'],
                'complexity': analysis['complexity']
            }
            
            self.contexts[user_id].append(exchange)
            self.user_last_activity[user_id] = datetime.now()
            
            # Update user preferences
            self._update_preferences(user_id, analysis, intent)
    
    def get_context_for_prediction(self, user_id: str, limit: int = 5) -> str:
        """Get context text for model prediction"""
        with self._lock:
            recent_exchanges = list(self.contexts[user_id])[-limit:]
            context_parts = []
            
            for exchange in recent_exchanges:
                context_parts.append(exchange['processed'])
            
            return ' '.join(context_parts)
    
    def get_conversation_context(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get full conversation context"""
        with self._lock:
            return list(self.contexts[user_id])[-limit:]
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences and patterns"""
        return dict(self.user_preferences[user_id])
    
    def clear_user_context(self, user_id: str):
        """Clear user context"""
        with self._lock:
            self.contexts.pop(user_id, None)
            self.user_preferences.pop(user_id, None)
            self.user_last_activity.pop(user_id, None)
    
    def _update_preferences(self, user_id: str, analysis: Dict, intent: str):
        """Update user preferences based on interaction"""
        prefs = self.user_preferences[user_id]
        
        # Track intent preferences
        if 'intent_frequency' not in prefs:
            prefs['intent_frequency'] = defaultdict(int)
        prefs['intent_frequency'][intent] += 1
        
        # Track topic interests
        if 'topic_interests' not in prefs:
            prefs['topic_interests'] = defaultdict(int)
        for topic in analysis['topics']:
            prefs['topic_interests'][topic] += 1
        
        # Track question patterns
        if 'asks_questions' not in prefs:
            prefs['asks_questions'] = 0
        if analysis['is_question']:
            prefs['asks_questions'] += 1
        
        # Track complexity preference
        if 'avg_complexity' not in prefs:
            prefs['avg_complexity'] = analysis['complexity']
        else:
            prefs['avg_complexity'] = (prefs['avg_complexity'] * 0.8 + 
                                     analysis['complexity'] * 0.2)
    
    def _cleanup_inactive_users(self):
        """Remove inactive users to manage memory"""
        cutoff_time = datetime.now() - timedelta(seconds=86400)  # 24 hours
        inactive_users = []
        
        for user_id, last_activity in self.user_last_activity.items():
            if last_activity < cutoff_time:
                inactive_users.append(user_id)
        
        for user_id in inactive_users:
            self.contexts.pop(user_id, None)
            self.user_preferences.pop(user_id, None)
            self.user_last_activity.pop(user_id, None)
        
        logger.info(f"Cleaned up {len(inactive_users)} inactive users")
    
    def _start_cleanup_thread(self):
        """Start periodic cleanup thread"""
        def cleanup_worker():
            while True:
                time.sleep(self.cleanup_interval)
                with self._lock:
                    self._cleanup_inactive_users()
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

# Model Manager (enhanced)
class EnhancedModelManager:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.intents = []
        self._lock = threading.RLock()
        self._loaded = False
        # Lightweight reinforcement memory: { text -> { intent -> {pos,neg} } }
        self.feedback_memory = defaultdict(lambda: defaultdict(lambda: {'pos': 0, 'neg': 0}))
        # Intent bias across all inputs (global nudging)
        self.intent_bias = defaultdict(float)
        # Detailed feedback store for bandit-style rewards
        # feedback_store[canonical] = [ {user_input,predicted_intent,bot_response,reward_score,timestamp,session_id} ]
        self.feedback_store = defaultdict(list)
        # Cumulative reward per response: reward_table[canonical][intent][bot_response] = total_reward
        self.reward_table = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        # Persistence
        self.store_path = os.path.join(self.config.MODEL_DIR, 'knowledge_store.json')

    def save_store(self):
        try:
            os.makedirs(self.config.MODEL_DIR, exist_ok=True)
            data = {
                'feedback_store': self.feedback_store,
                'reward_table': self.reward_table,
                'intent_bias': self.intent_bias,
            }
            # Convert defaultdicts to plain dicts recursively
            def to_dict(obj):
                if isinstance(obj, defaultdict):
                    obj = dict(obj)
                if isinstance(obj, dict):
                    return {k: to_dict(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [to_dict(v) for v in obj]
                return obj
            with open(self.store_path, 'w', encoding='utf-8') as f:
                json.dump(to_dict(data), f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save knowledge store: {e}")

    def load_store(self):
        try:
            if not os.path.exists(self.store_path):
                return
            with open(self.store_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Restore structures
            self.feedback_store = defaultdict(list, data.get('feedback_store', {}))
            # reward_table: canonical -> intent -> resp -> score
            rt_raw = data.get('reward_table', {})
            rt_lvl2 = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            for canon, intents in rt_raw.items():
                rt_lvl2[canon] = defaultdict(lambda: defaultdict(float))
                for inten, resp_map in intents.items():
                    rt_lvl2[canon][inten] = defaultdict(float, resp_map)
            self.reward_table = rt_lvl2
            self.intent_bias = defaultdict(float, data.get('intent_bias', {}))
            logger.info("Knowledge store loaded")
        except Exception as e:
            logger.error(f"Failed to load knowledge store: {e}")
    
    @contextmanager
    def safe_model_access(self):
        """Context manager for thread-safe model access"""
        with self._lock:
            if not self._loaded:
                raise RuntimeError("Model not loaded")
            yield
    
    def load_all(self):
        """Load all model components"""
        with self._lock:
            try:
                logger.info("Loading enhanced model components...")
                
                # Load model
                self.model = load_model(self.config.MODEL_FILE)
                logger.info("Model loaded successfully")
                
                # Load tokenizer
                with open(self.config.TOKENIZER_FILE, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                logger.info("Tokenizer loaded successfully")
                
                # Load label encoder
                with open(self.config.LABEL_ENCODER_FILE, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info("Label encoder loaded successfully")
                
                # Load intents
                with open(self.config.INTENTS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.intents = data.get('intents', data.get('intent', []))
                    else:
                        self.intents = data
                logger.info(f"Loaded {len(self.intents)} intents")
                
                self._loaded = True
                logger.info("All enhanced components loaded successfully")
                # Load persisted reinforcement data
                self.load_store()
                
            except Exception as e:
                self._loaded = False
                logger.error(f"Error loading components: {e}")
                raise
    
    def predict_intent(self, message: str, context_text: str = "") -> List[Dict]:
        """Predict intent from message with context"""
        with self.safe_model_access():
            # Robust keyword overrides for common small-talk/personal queries
            overrides = [
                (r"\b(hi|hello|hey|good (morning|afternoon|evening))\b", 'greeting'),
                (r"\b(bye|goodbye|see you|farewell)\b", 'goodbye'),
                (r"\b(thank(s)?|thanks a lot|appreciate it)\b", 'thanks'),
                (r"\b(help|can you help|assist me|support)\b", 'help'),
                (r"\b(what(?:'s| is) your name|who are you|your name|about (you|the bot)|tell me about yourself|what about you)\b", 'name'),
                (r"\bhow are you\b", 'how_are_you'),
                (r"\bwhere are you\b", 'location'),
                (r"\b(neural network|deep learning|machine learning)\b", 'machine_learning'),
            ]
            for pattern, mapped_intent in overrides:
                if re.search(pattern, message, re.IGNORECASE):
                    return [{'intent': mapped_intent, 'confidence': 0.95}]

            # Feedback memory boost
            canonical = re.sub(r"\s+", " ", message.strip().lower())
            if canonical in self.feedback_memory:
                scored = sorted(
                    ((intent, stats['pos'] - stats['neg']) for intent, stats in self.feedback_memory[canonical].items()),
                    key=lambda x: x[1], reverse=True
                )
                if scored and scored[0][1] > 0:
                    return [{'intent': scored[0][0], 'confidence': 0.92}]

            # Combine context with message after overrides
            full_input = (context_text + ' ' + message).strip() if context_text else message
            
            # Tokenize and pad
            sequence = self.tokenizer.texts_to_sequences([full_input])
            padded_sequence = pad_sequences(
                sequence, 
                maxlen=self.config.MAX_SEQUENCE_LENGTH, 
                padding='post'
            )
            
            # Get predictions
            predictions = self.model.predict(padded_sequence)[0]
            
            # Apply global bias
            for idx in range(len(predictions)):
                intent_name = self.label_encoder.classes_[idx]
                predictions[idx] = float(predictions[idx]) + float(self.intent_bias[intent_name])
            
            # Boost intents that historically earned rewards for this exact text
            canonical = re.sub(r"\s+", " ", message.strip().lower())
            if canonical in self.reward_table:
                reward_by_intent = {
                    inten: sum(self.reward_table[canonical][inten].values())
                    for inten in self.reward_table[canonical]
                }
                if reward_by_intent:
                    # Normalize and add as small bias
                    vals = np.array(list(reward_by_intent.values()), dtype=np.float64)
                    if np.any(vals != 0):
                        minv, maxv = np.min(vals), np.max(vals)
                        for idx in range(len(predictions)):
                            intent_name = self.label_encoder.classes_[idx]
                            rv = reward_by_intent.get(intent_name, 0.0)
                            norm = (rv - minv) / (maxv - minv + 1e-9)
                            predictions[idx] += 0.05 * norm

            # Temperature + top-k + top-p (nucleus) sampling over intents
            logits = np.array(predictions, dtype=np.float64)
            # Temperature scaling
            temp = max(1e-6, float(self.config.TEMPERATURE))
            logits = np.log(np.clip(logits, 1e-12, 1.0)) / temp
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)

            # Top-k
            k = max(1, int(self.config.TOP_K))
            top_k_indices = np.argpartition(-probs, k-1)[:k]
            top_k_probs = probs[top_k_indices]
            top_k_probs = top_k_probs / np.sum(top_k_probs)

            # Top-p (nucleus) within top-k
            sorted_idx = np.argsort(-top_k_probs)
            cum = 0.0
            nucleus = []
            for i in sorted_idx:
                nucleus.append(i)
                cum += top_k_probs[i]
                if cum >= float(self.config.TOP_P):
                    break
            nucleus_indices = top_k_indices[nucleus]
            nucleus_probs = probs[nucleus_indices]
            nucleus_probs = nucleus_probs / np.sum(nucleus_probs)

            # Build result list (deterministic ordering by prob desc for API visibility)
            order = np.argsort(-nucleus_probs)
            results = []
            for j in order:
                idx = nucleus_indices[j]
                intent = self.label_encoder.classes_[idx]
                confidence = float(nucleus_probs[j])
                results.append({'intent': intent, 'confidence': confidence})
            
            # Nudge confidences based on feedback
            if canonical in self.feedback_memory and results:
                for r in results:
                    stats = self.feedback_memory[canonical].get(r['intent'])
                    if stats:
                        r['confidence'] = max(0.0, min(1.0, r['confidence'] + (stats['pos'] - stats['neg']) * 0.02))
                results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return results
    
    def get_intent_response(self, intent: str, entities: Dict = None, canonical: str = "", exploration_epsilon: float = 0.1) -> str:
        """Get response for intent, optionally personalized with entities, bandit + sampling selection."""
        for intent_data in self.intents:
            if intent_data['tag'] == intent:
                responses = intent_data['responses']
                if not responses:
                    return "I'm not sure how to respond to that."
                # Epsilon-greedy on reward, then nucleus sampling by historical reward
                if canonical:
                    rewards = self.reward_table[canonical][intent]
                    # With probability (1 - epsilon) exploit best-known
                    if random.random() > exploration_epsilon and rewards:
                        best_resp = max(responses, key=lambda r: float(rewards.get(r, 0.0)))
                        response = best_resp
                    else:
                        # Sample responses with softmax over rewards (temperature)
                        scores = np.array([float(rewards.get(r, 0.0)) for r in responses], dtype=np.float64)
                        temp = max(1e-6, float(self.config.RESPONSE_TEMPERATURE))
                        logits = scores / temp
                        probs = np.exp(logits - np.max(logits))
                        probs = probs / np.sum(probs) if np.sum(probs) > 0 else np.ones_like(probs) / len(probs)
                        # Top-p nucleus
                        order = np.argsort(-probs)
                        cum = 0.0
                        selected = []
                        for idx in order:
                            selected.append(idx)
                            cum += probs[idx]
                            if cum >= 0.9:
                                break
                        sel_probs = probs[selected]
                        sel_probs = sel_probs / np.sum(sel_probs)
                        choice_idx = np.random.choice(selected, p=sel_probs)
                        response = responses[choice_idx]
                else:
                    response = random.choice(responses)

                # Optionally expand response by combining more variations
                try:
                    combine_n = max(1, int(self.config.RESPONSE_COMBINE))
                except Exception:
                    combine_n = 1
                if combine_n > 1 and len(responses) > 1:
                    extras = []
                    # Pick additional distinct responses with highest reward
                    ranked = sorted(set(responses) - {response}, key=lambda r: float(self.reward_table[canonical][intent].get(r, 0.0)), reverse=True)
                    for extra in ranked[:combine_n-1]:
                        extras.append(extra)
                    if extras:
                        response = response + "\n\n" + "\n\n".join(extras)
                
                # Personalize response with entities if available
                if entities:
                    response = self._personalize_response(response, entities)
                
                return response
        
        return "I'm not sure how to respond to that."
    
    def _personalize_response(self, response: str, entities: Dict) -> str:
        """Personalize response based on extracted entities"""
        # Simple entity substitution
        if 'PERSON' in entities and entities['PERSON']:
            person_name = entities['PERSON'][0]['text']
            response = response.replace('someone', person_name)
        
        if 'GPE' in entities and entities['GPE']:
            location = entities['GPE'][0]['text']
            response = response.replace('that place', location)
        
        return response
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded

# Initialize enhanced components
entity_extractor = EntityExtractor()
knowledge_integrator = KnowledgeIntegrator(config)
message_processor = EnhancedMessageProcessor(entity_extractor, knowledge_integrator)
reasoning_engine = ReasoningEngine()
confidence_manager = EnhancedConfidenceManager(
    config.BASE_CONFIDENCE_THRESHOLD, 
    config.ADAPTIVE_THRESHOLD_RANGE
)
conversation_context = EnhancedConversationContext(
    config.MAX_CONTEXT_MESSAGES, 
    config.MAX_USERS, 
    config.CLEANUP_INTERVAL
)
model_manager = EnhancedModelManager(config)

# Enhanced prediction function with advanced reasoning
def enhanced_predict(message: str, user_id: str, debug: bool = False) -> Dict:
    """Enhanced prediction with advanced reasoning and probability chains"""
    start_time = time.time()
    
    try:
        # Analyze message with reasoning patterns
        analysis = message_processor.analyze_message(message)
        reasoning_patterns = reasoning_engine.analyze_reasoning_patterns(message)
        analysis['reasoning_patterns'] = reasoning_patterns
        
        # Get conversation context
        conv_context = conversation_context.get_conversation_context(user_id, limit=5)
        context_text = conversation_context.get_context_for_prediction(
            user_id, config.CONTEXT_WINDOW_SIZE
        )
        
        # Get enhanced adaptive confidence threshold with context
        confidence_threshold = confidence_manager.get_threshold(user_id, analysis, conv_context)
        
        # Handle follow-up messages with reasoning
        if analysis['is_follow_up']:
            if conv_context:
                last_exchange = conv_context[-1]
                # Use reasoning to enhance follow-up response
                follow_up_confidence = min(0.95, last_exchange.get('confidence', 0.8) + 0.1)
                response = f"Let me elaborate on that. {last_exchange['response']}"
                
                # Add reasoning context
                if reasoning_patterns.get('causal', 0) > 0.3:
                    response += " This is because the previous context supports this direction."
                
                return {
                    'response': response,
                    'intent': last_exchange['intent'],
                    'confidence': follow_up_confidence,
                    'entities': analysis['entities'],
                    'knowledge_used': False,
                    'adaptive_threshold': confidence_threshold,
                    'reasoning': 'Follow-up with context enhancement'
                }
        
        # Multi-question handling: split into sub-questions, answer each, then aggregate
        sub_questions = message_processor.split_multi_questions(message)
        aggregated_parts = []
        best_overall = None
        canonical_text = re.sub(r"\s+", " ", analysis['processed'].strip().lower())
        if sub_questions and len(sub_questions) > 1:
            for sq in sub_questions[:3]:  # cap to avoid long processing
                sq_proc = message_processor.preprocess_text(sq)
                sq_predictions = model_manager.predict_intent(sq_proc, context_text)
                sq_intent = sq_predictions[0]['intent'] if sq_predictions else 'fallback'
                sq_response = model_manager.get_intent_response(sq_intent, analysis['entities'], canonical=canonical_text)
                aggregated_parts.append(f"Q: {sq}\nA: {sq_response}")
                if not best_overall or sq_predictions[0]['confidence'] > best_overall['confidence']:
                    best_overall = sq_predictions[0]
            # Join parts into a single coherent response
            response = "\n\n".join(aggregated_parts)
            intent = best_overall['intent'] if best_overall else 'fallback'
            confidence = best_overall['confidence'] if best_overall else 0.5
            knowledge_used = False
            probability_chain = {'uncertainty_level': 'medium'}
            decision = {'reasoning': 'Aggregated multi-question response'}
        else:
            predictions = model_manager.predict_intent(analysis['processed'], context_text)
        
        # Build probability chain for decision making
            probability_chain = reasoning_engine.build_probability_chain(predictions, conv_context)
        
        # Make decision using reasoning engine
            decision = reasoning_engine.make_decision(probability_chain, confidence_threshold)
        
        # Determine response strategy based on decision
        if decision['action'] == 'proceed':
            # Use model prediction
            top_prediction = predictions[0]
            intent = top_prediction['intent']
            response = model_manager.get_intent_response(intent, analysis['entities'], canonical=canonical_text)
            confidence = probability_chain['overall_confidence']
            knowledge_used = False
            
            # Enhance response with knowledge if relevant
            if intent in ['tech_info', 'neuromorphic_chips'] and analysis['topics']:
                knowledge_enhancement = knowledge_integrator.try_knowledge_integration(analysis['topics'][0])
                if knowledge_enhancement:
                    response += f"\n\nAdditional information: {knowledge_enhancement}"
                    knowledge_used = True
            
            # Add reasoning-based enhancements
            if reasoning_patterns.get('conditional', 0) > 0.3:
                response += " Based on the conditions you mentioned, this response should be helpful."
            elif reasoning_patterns.get('comparative', 0) > 0.3:
                response += " Let me provide a comparison to help clarify this topic."
                
        else:
            # Use fallback strategy with reasoning
            if analysis['is_question'] and analysis['topics']:
                knowledge_response = knowledge_integrator.try_knowledge_integration(analysis['topics'][0])
                if knowledge_response:
                    intent = 'knowledge_integration'
                    response = knowledge_response
                    confidence = 0.7
                    knowledge_used = True
                else:
                    intent = 'fallback'
                    response = message_processor.generate_fallback_response(analysis)
                    confidence = probability_chain['overall_confidence']
                    knowledge_used = False
            else:
                intent = 'fallback'
                response = message_processor.generate_fallback_response(analysis)
                confidence = probability_chain['overall_confidence']
                knowledge_used = False
            
            # Add reasoning-based fallback enhancements
            if decision['uncertainty_factors']:
                response += f" I'm uncertain because: {', '.join(decision['uncertainty_factors'])}."
            if decision['recommendations']:
                response += f" Suggestion: {decision['recommendations'][0]}."
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Update conversation context with enhanced data
        enhanced_analysis = analysis.copy()
        enhanced_analysis['reasoning_patterns'] = reasoning_patterns
        enhanced_analysis['probability_chain'] = probability_chain
        enhanced_analysis['decision'] = decision
        
        conversation_context.add_exchange(
            user_id, enhanced_analysis, intent, response, confidence, knowledge_used
        )
        
        # Update confidence manager with enhanced metrics
        confidence_manager.update_performance(
            user_id, 
            confidence >= confidence_threshold, 
            response_time, 
            intent
        )
        
        # Prepare enhanced result
        result = {
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'entities': analysis['entities'],
            'knowledge_used': knowledge_used,
            'adaptive_threshold': confidence_threshold,
            'reasoning': decision['reasoning'],
            'uncertainty_level': probability_chain['uncertainty_level'],
            'reasoning_patterns': reasoning_patterns,
            'response_time': round(response_time, 3),
            'session_id': session.get('user_id'),
            'user_input': message
        }
        
        if debug:
            result['debug'] = {
                'message_analysis': enhanced_analysis,
                'context_text': context_text,
                'all_predictions': predictions,
                'probability_chain': probability_chain,
                'decision': decision,
                'user_preferences': conversation_context.get_user_preferences(user_id),
                'user_insights': confidence_manager.get_user_insights(user_id),
                'threshold_factors': {
                    'base': config.BASE_CONFIDENCE_THRESHOLD,
                    'adaptive': confidence_threshold,
                    'complexity': analysis['complexity'],
                    'reasoning_strength': probability_chain['reasoning_strength'],
                    'context_support': probability_chain['context_support']
                }
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced_predict: {e}")
        return {
            'response': 'Sorry, I encountered an error. Please try again.',
            'intent': 'error',
            'confidence': 0.0,
            'entities': {},
            'knowledge_used': False,
            'adaptive_threshold': config.BASE_CONFIDENCE_THRESHOLD,
            'reasoning': 'System error occurred',
            'uncertainty_level': 'high',
            'reasoning_patterns': {},
            'response_time': round(time.time() - start_time, 3)
        }

# Flask application setup
app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# CORS setup
CORS(app, origins=os.environ.get('ALLOWED_ORIGINS', '*').split(','))

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per minute"]
)
limiter.init_app(app)

# Ensure model is loaded before serving requests
_model_loaded = False

@app.before_request
def _load_model_before_request():
    global _model_loaded
    if not _model_loaded:
        try:
            if not model_manager.is_loaded():
                logger.info("Loading model before first request...")
                model_manager.load_all()
                logger.info("Model ready")
            _model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load model on startup: {e}")
            _model_loaded = True  # Prevent repeated attempts

# Enhanced API Routes
@app.route('/')
def index():
    """Serve main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@limiter.limit("30 per minute")
def predict():
    """Enhanced prediction endpoint with dynamic features"""
    try:
        data = request.get_json()
        
        # Get or create user ID
        if 'user_id' not in session:
            session['user_id'] = f"user_{random.randint(10000, 99999)}"
        user_id = session['user_id']
        
        message = data.get('message', '').strip()
        debug = data.get('debug', False)
        
        if not message:
            return jsonify({
                'error': 'Empty message',
                'response': 'Please enter a message.',
                'suggestions': [
                    'Ask about neuromorphic chips',
                    'Request book recommendations', 
                    'Ask a technical question',
                    'Say hello'
                ]
            }), 400
        
        # Ensure model is loaded (lazy-load fallback)
        if not model_manager.is_loaded():
            try:
                logger.info("Model not loaded; attempting lazy load in /predict")
                model_manager.load_all()
            except Exception as e:
                logger.error(f"Lazy model load failed: {e}")
                return jsonify({
                    'error': 'Model not loaded',
                    'response': 'The AI model is not available. Please upload model files or try /reload_model.',
                    'intent': 'error',
                    'confidence': 0.0
                }), 503
        
        # Get enhanced prediction
        result = enhanced_predict(message, user_id, debug)
        
        # Handle special intents
        if result['intent'] in ['goodbye', 'farewell']:
            session.pop('user_id', None)
            conversation_context.clear_user_context(user_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'response': 'Sorry, I encountered an error. Please try again.',
            'intent': 'error',
            'confidence': 0.0,
            'knowledge_used': False
        }), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Evaluate model accuracy on provided labeled samples

    Payload JSON format:
    { "samples": [ {"text": "...", "label": "intent_tag"}, ... ] }
    """
    try:
        data = request.get_json() or {}
        samples = data.get('samples', [])

        if not samples or not isinstance(samples, list):
            return jsonify({'error': 'Provide samples as a non-empty list of {text, label}'}), 400

        # Ensure model is loaded
        if not model_manager.is_loaded():
            try:
                model_manager.load_all()
            except Exception as e:
                logger.error(f"Model load failed in /evaluate: {e}")
                return jsonify({'error': 'Model not loaded'}), 503

        total = 0
        correct = 0
        results = []

        for sample in samples:
            text = (sample or {}).get('text', '')
            label = (sample or {}).get('label')
            if not text or label is None:
                continue

            # Preprocess and predict without altering user context
            processed = message_processor.preprocess_text(text)
            try:
                predictions = model_manager.predict_intent(processed, context_text="")
                top = predictions[0]['intent'] if predictions else None
                is_correct = (top == label)
                total += 1
                correct += 1 if is_correct else 0
                results.append({
                    'text': text,
                    'label': label,
                    'predicted': top,
                    'confidence': predictions[0]['confidence'] if predictions else 0.0,
                    'correct': is_correct
                })
            except Exception as e:
                logger.error(f"Prediction failed during evaluation: {e}")
                results.append({
                    'text': text,
                    'label': label,
                    'predicted': None,
                    'confidence': 0.0,
                    'correct': False,
                    'error': str(e)
                })

        accuracy = (correct / total) if total > 0 else 0.0
        return jsonify({
            'total': total,
            'correct': correct,
            'accuracy': round(accuracy, 4),
            'details': results
        })
    except Exception as e:
        logger.error(f"Error in /evaluate: {e}")
        return jsonify({'error': 'Evaluation failed'}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Enhanced feedback endpoint"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', session.get('user_id'))
        feedback_type = data.get('feedback')  # 'positive', 'negative'
        message_text = data.get('message')  # text user asked
        predicted_intent = data.get('intent')  # model's predicted intent
        
        if not all([user_id, feedback_type]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Update confidence manager
        was_correct = feedback_type == 'positive'
        confidence_manager.update_performance(user_id, was_correct)
        
        # Update reinforcement memory, reward store, and global bias
        try:
            if message_text and predicted_intent:
                canonical = re.sub(r"\s+", " ", message_text.strip().lower())
                stats = model_manager.feedback_memory[canonical][predicted_intent]
                if feedback_type == 'positive':
                    stats['pos'] += 1
                    model_manager.intent_bias[predicted_intent] = min(0.2, model_manager.intent_bias[predicted_intent] + 0.01)
                    reward = 1.0
                else:
                    stats['neg'] += 1
                    model_manager.intent_bias[predicted_intent] = max(0.0, model_manager.intent_bias[predicted_intent] - 0.01)
                    reward = -1.0

                # Store detailed feedback with timestamp and session id
                fb = {
                    'user_input': message_text,
                    'predicted_intent': predicted_intent,
                    'bot_response': data.get('bot_response', ''),
                    'reward_score': reward,
                    'timestamp': datetime.now().isoformat(),
                    'session_id': session.get('user_id')
                }
                model_manager.feedback_store[canonical].append(fb)
                # Update cumulative reward for bandit selection
                bot_resp = fb['bot_response']
                if bot_resp:
                    model_manager.reward_table[canonical][predicted_intent][bot_resp] += reward
        except Exception as _:
            pass
        
        # Log detailed feedback
        feedback_entry = {
            'user_id': user_id,
            'feedback': feedback_type,
            'intent': predicted_intent,
            'message': (message_text or '')[:200],
            'timestamp': datetime.now().isoformat(),
            'user_context': len(conversation_context.get_conversation_context(user_id))
        }
        
        logger.info(f"Enhanced feedback received: {feedback_entry}")
        
        return jsonify({
            'status': 'Feedback recorded successfully',
            'adaptive_learning': 'Confidence thresholds and intent bias updated'
        })
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({'error': 'Failed to process feedback'}), 500

@app.route('/context/<user_id>', methods=['GET'])
def get_user_context(user_id):
    """Get user conversation context and preferences"""
    try:
        context = conversation_context.get_conversation_context(user_id)
        preferences = conversation_context.get_user_preferences(user_id)
        
        return jsonify({
            'conversation_history': context,
            'user_preferences': preferences,
            'context_length': len(context),
            'last_activity': conversation_context.user_last_activity.get(user_id, 'Never').isoformat() if isinstance(conversation_context.user_last_activity.get(user_id), datetime) else 'Never'
        })
    except Exception as e:
        logger.error(f"Error getting user context: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_context', methods=['POST'])
def clear_context():
    """Clear user context"""
    try:
        user_id = session.get('user_id')
        if user_id:
            conversation_context.clear_user_context(user_id)
            return jsonify({'status': 'Context cleared successfully'})
        else:
            return jsonify({'error': 'No active session'}), 400
    except Exception as e:
        logger.error(f"Error clearing context: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/entities', methods=['POST'])
def extract_entities():
    """Extract entities from text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        entities = entity_extractor.extract_entities(text)
        analysis = message_processor.analyze_message(text)
        
        return jsonify({
            'entities': entities,
            'analysis': {
                'is_question': analysis['is_question'],
                'complexity': analysis['complexity'],
                'topics': analysis['topics']
            }
        })
        
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/knowledge', methods=['POST'])
def get_knowledge():
    """Get knowledge about a topic"""
    try:
        data = request.get_json()
        topic = data.get('topic', '')
        source = data.get('source', 'wikipedia')  # 'wikipedia' or 'news'
        
        if not topic:
            return jsonify({'error': 'No topic provided'}), 400
        
        if source == 'wikipedia':
            knowledge = knowledge_integrator.get_wikipedia_summary(topic)
            return jsonify({
                'topic': topic,
                'source': 'Wikipedia',
                'knowledge': knowledge
            })
        elif source == 'news':
            articles = knowledge_integrator.get_recent_news(topic)
            return jsonify({
                'topic': topic,
                'source': 'News',
                'articles': articles
            })
        else:
            return jsonify({'error': 'Invalid source'}), 400
            
    except Exception as e:
        logger.error(f"Error getting knowledge: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/news', methods=['POST'])
def get_news():
    """Get news from multiple sources"""
    try:
        data = request.get_json()
        topic = data.get('topic', '')
        source = data.get('source', 'auto')  # 'newsapi', 'currents', or 'auto'
        limit = data.get('limit', 3)
        
        if not topic:
            return jsonify({'error': 'No topic provided'}), 400
        
        results = {'topic': topic, 'articles': []}
        
        if source in ['newsapi', 'auto'] and config.ENABLE_NEWS_API:
            newsapi_articles = knowledge_integrator.get_recent_news(topic, limit)
            results['articles'].extend([{**article, 'source_api': 'NewsAPI'} for article in newsapi_articles])
        
        if source in ['currents', 'auto'] and config.ENABLE_CURRENTS_API:
            currents_articles = knowledge_integrator.get_currents_news(topic, limit)
            results['articles'].extend([{**article, 'source_api': 'Currents'} for article in currents_articles])
        
        # Limit total results
        results['articles'] = results['articles'][:limit]
        
        return jsonify(results)
            
    except Exception as e:
        logger.error(f"Error getting news: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Enhanced health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy' if model_manager.is_loaded() else 'unhealthy',
            'model_loaded': model_manager.is_loaded(),
            'intents_count': len(model_manager.intents),
            'active_users': len(conversation_context.contexts),
            'features': {
                'entity_extraction': entity_extractor.enabled,
                'wikipedia_integration': config.ENABLE_WIKIPEDIA,
                'news_integration': config.ENABLE_NEWS_API,
                'currents_integration': config.ENABLE_CURRENTS_API,
                'adaptive_confidence': True,
                'context_awareness': True,
                'reasoning_engine': True,
                'probability_chains': True,
                'advanced_thinking': True
            },
            'memory_usage': {
                'cached_knowledge': len(knowledge_integrator.cache),
                'user_contexts': len(conversation_context.contexts),
                'user_preferences': len(conversation_context.user_preferences)
            },
            'timestamp': datetime.now().isoformat(),
            'version': '3.0.0-enhanced'
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_enhanced_stats():
    """Get comprehensive usage statistics"""
    try:
        # Calculate conversation stats
        total_exchanges = sum(len(ctx) for ctx in conversation_context.contexts.values())
        
        # Intent distribution
        intent_counts = defaultdict(int)
        complexity_sum = 0
        question_count = 0
        
        for user_contexts in conversation_context.contexts.values():
            for exchange in user_contexts:
                intent_counts[exchange['intent']] += 1
                complexity_sum += exchange.get('complexity', 0)
                if exchange.get('is_question', False):
                    question_count += 1
        
        avg_complexity = complexity_sum / total_exchanges if total_exchanges > 0 else 0
        
        return jsonify({
            'total_conversations': total_exchanges,
            'active_users': len(conversation_context.contexts),
            'intents_supported': len(model_manager.intents),
            'model_loaded': model_manager.is_loaded(),
            'intent_distribution': dict(intent_counts),
            'engagement_metrics': {
                'average_complexity': round(avg_complexity, 3),
                'question_ratio': round(question_count / total_exchanges, 3) if total_exchanges > 0 else 0,
                'knowledge_integration_usage': len(knowledge_integrator.cache)
            },
            'feature_usage': {
                'entity_extraction_enabled': entity_extractor.enabled,
                'knowledge_sources': {
                    'wikipedia': config.ENABLE_WIKIPEDIA,
                    'news': config.ENABLE_NEWS_API,
                    'currents': config.ENABLE_CURRENTS_API
                }
            }
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reload_model', methods=['POST'])
def reload_model():
    """Reload model components"""
    try:
        model_manager.load_all()
        return jsonify({
            'status': 'Enhanced model reloaded successfully',
            'features_active': {
                'entity_extraction': entity_extractor.enabled,
                'knowledge_integration': True,
                'adaptive_confidence': True,
                'reasoning_engine': True,
                'probability_chains': True
            }
        })
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/insights/<user_id>', methods=['GET'])
def get_user_insights(user_id):
    """Get detailed user insights and reasoning patterns"""
    try:
        insights = confidence_manager.get_user_insights(user_id)
        context = conversation_context.get_conversation_context(user_id, limit=10)
        preferences = conversation_context.get_user_preferences(user_id)
        
        # Analyze reasoning patterns from recent conversations
        recent_reasoning = []
        for exchange in context[-5:]:
            if 'reasoning_patterns' in exchange:
                recent_reasoning.append(exchange['reasoning_patterns'])
        
        # Calculate average reasoning patterns
        avg_reasoning = {}
        if recent_reasoning:
            for pattern_type in ['causal', 'conditional', 'comparative', 'temporal']:
                avg_reasoning[pattern_type] = sum(
                    rp.get(pattern_type, 0) for rp in recent_reasoning
                ) / len(recent_reasoning)
        
        return jsonify({
            'user_insights': insights,
            'reasoning_patterns': avg_reasoning,
            'preferences': preferences,
            'recent_context_length': len(context),
            'reasoning_engine_active': True
        })
    except Exception as e:
        logger.error(f"Error getting user insights: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reasoning', methods=['POST'])
def analyze_reasoning():
    """Analyze reasoning patterns in text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Analyze reasoning patterns
        patterns = reasoning_engine.analyze_reasoning_patterns(text)
        
        # Analyze message complexity
        analysis = message_processor.analyze_message(text)
        
        return jsonify({
            'text': text,
            'reasoning_patterns': patterns,
            'complexity': analysis['complexity'],
            'is_question': analysis['is_question'],
            'entities': analysis['entities'],
            'reasoning_strength': sum(patterns.values()) / len(patterns) if patterns else 0
        })
    except Exception as e:
        logger.error(f"Error analyzing reasoning: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Please slow down your requests',
        'retry_after': '60 seconds'
    }), 429

@app.errorhandler(500)
def internal_error_handler(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong on our end',
        'support': 'Please try again or contact support if the issue persists'
    }), 500

@app.errorhandler(404)
def not_found_handler(e):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/predict', '/feedback', '/context/<user_id>', 
            '/clear_context', '/entities', '/knowledge', 
            '/news', '/health', '/stats', '/reload_model'
        ]
    }), 404

# Main execution
if __name__ == '__main__':
    try:
        # Load enhanced model components
        logger.info("Initializing enhanced chatbot with dynamic features...")
        model_manager.load_all()
        
        logger.info("Enhanced features active:")
        logger.info(f"- Entity Extraction: {entity_extractor.enabled}")
        logger.info(f"- Wikipedia Integration: {config.ENABLE_WIKIPEDIA}")
        logger.info(f"- News Integration: {config.ENABLE_NEWS_API}")
        logger.info(f"- Currents Integration: {config.ENABLE_CURRENTS_API}")
        logger.info(f"- Adaptive Confidence: Enabled")
        logger.info(f"- Context Awareness: Enabled")
        logger.info(f"- Memory Management: Enabled")
        
        # Show local and LAN URLs for easy access
        lan_ip = get_lan_ip()
        logger.info(f"Access URLs:")
        logger.info(f"- Local: http://127.0.0.1:{config.PORT}")
        logger.info(f"- LAN:   http://{lan_ip}:{config.PORT}  (use this from other devices on the same network)")
        
        # Start Flask app
        app.run(
            debug=config.DEBUG,
            host=config.HOST,
            port=config.PORT,
            threaded=True
        )
        # On shutdown, persist store
        model_manager.save_store()
        
    except Exception as e:
        logger.error(f"Failed to start enhanced application: {e}")
        raise