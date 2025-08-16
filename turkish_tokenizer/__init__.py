"""
Turkish Tokenizer

A comprehensive Turkish language tokenizer.
Provides state-of-the-art tokenization and text generation capabilities for Turkish.
"""

__version__ = "0.1.0"
__author__ = "M. Ali Bayram"
__email__ = "malibayram20@gmail.com"

from .tr_decoder import TRDecoder
from .tr_tokenizer import TokenType, TRTokenizer

__all__ = [
    # Tokenizer
    "TRTokenizer",
    "TokenType",
    "TRDecoder",
]

# Package metadata
__title__ = "turkish-tokenizer"
__description__ = "Turkish tokenizer for Turkish language processing"
__url__ = "https://github.com/malibayram/tokenizer"
__license__ = "MIT"
