"""
Turkish Tokenizer with Gemma Model Implementation

A comprehensive Turkish language tokenizer with integrated Gemma model support.
Provides state-of-the-art tokenization and text generation capabilities for Turkish.
"""

__version__ = "0.1.0"
__author__ = "M. Ali Bayram"
__email__ = "malibayram20@gmail.com"

from .gemma_model import (Architecture, AttentionType, GemmaAttention,
                          GemmaConfig, GemmaDecoderLayer, GemmaForCausalLM,
                          GemmaMLP, GemmaModel, get_config_for_270m,
                          get_config_for_270m_tr_tokenizer)
from .tr_decoder import TRDecoder
from .tr_tokenizer import TokenType, TRTokenizer

__all__ = [
    # Tokenizer
    "TRTokenizer",
    "TokenType",
    "TRDecoder",
    
    # Gemma Model
    "GemmaForCausalLM",
    "GemmaConfig",
    "GemmaModel",
    "GemmaAttention",
    "GemmaMLP", 
    "GemmaDecoderLayer",
    
    # Enums and configs
    "Architecture",
    "AttentionType",
    "get_config_for_270m",
    "get_config_for_270m_tr_tokenizer",
]

# Package metadata
__title__ = "turkish-tokenizer"
__description__ = "Turkish tokenizer with Gemma model implementation for Turkish language processing"
__url__ = "https://github.com/malibayram/tokenizer"
__license__ = "MIT"
