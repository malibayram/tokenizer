# Tokenizer Research Project

A comprehensive research project focused on developing advanced tokenization methods for the Turkish language, incorporating linguistic rules, morphological analysis, and semantic understanding.

## Project Overview

This project explores multiple approaches to Turkish text tokenization, combining traditional NLP techniques with modern machine learning methods. The goal is to create tokenizers that understand Turkish morphology and semantics, leading to better performance in downstream NLP tasks.

## Key Features

- **Semantic Tokenization**: Converts Turkish morphological suffixes to language-independent semantic tokens
- **Morphological Analysis**: Root-suffix separation using Turkish grammatical rules
- **Grammatical Event Reversal**: Handles Turkish phonological processes (vowel harmony, consonant softening, etc.)
- **BPE Integration**: Modern subword tokenization with Turkish-specific preprocessing
- **Multiple Tokenizer Implementations**: Various approaches for different use cases

## Project Structure

### 🧠 semantic_tokenizer/

Semantic tokenization approach that focuses on meaning preservation through grammatical analysis.

**Key Components:**

- Grammatical Event Revertor (GER) - Reverses Turkish phonological changes
- Semantic Converter - Maps suffixes to language-independent tokens
- Root extraction and phonological process handling

**Main Files:**

- `guncel_strateji.md` - Current strategy documentation
- `tokenizer_v01.ipynb` - Main implementation notebook
- Various JSON files containing roots, suffixes, and mappings

### 🔧 tr_tokenizer/

Turkish word segmentation tool using morphological analysis with ITU NLP tools integration.

**Key Components:**

- Root-suffix separation using linguistic rules
- ITU NLP tools integration for morphological analysis
- Frequency-based vocabulary building
- GUI interface for interactive use

**Main Files:**

- `kelime_bol.py` - Core word segmentation algorithm
- `kokbul.py` - Root finding utilities
- `gui/` - Graphical user interface
- `veri/` - Training data and word lists

### 📚 tokenizer_preparation/

BPE (Byte Pair Encoding) tokenizer training and preparation pipeline.

**Key Components:**

- Custom BPE tokenizer training
- Frequency analysis and vocabulary optimization
- Integration with Hugging Face tokenizers
- Performance evaluation tools

**Main Files:**

- `train_tokenizer.py` - Training pipeline
- `byte_pair_tokenizer.ipynb` - BPE implementation
- Various JSON tokenizer configurations

## Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Basic Usage

#### Semantic Tokenizer

```python
# Navigate to semantic_tokenizer directory
cd semantic_tokenizer

# Open the main notebook
jupyter notebook tokenizer_v01.ipynb
```

#### TR Tokenizer

```python
from tr_tokenizer.kelime_bol import kok_tara

# Analyze a Turkish word
word = "kitaplarımızdan"
result = kok_tara(word)
print(f"Analysis: {result}")
```

#### BPE Tokenizer Training

```python
# Navigate to tokenizer_preparation directory
cd tokenizer_preparation

# Run the training script
python train_tokenizer.py
```

## Research Methodology

### 1. Semantic Tokenization Approach

The semantic tokenizer implements a two-step process:

1. **Grammatical Event Reversal**: Identifies and reverses Turkish phonological processes

   - Vowel contraction (Ünlü Daralması)
   - Vowel dropping (Ünlü Düşmesi)
   - Consonant softening (Ünsüz Yumuşaması)
   - And other Turkish-specific sound changes

2. **Semantic Conversion**: Maps morphological suffixes to semantic tokens
   - Language-independent representation
   - Meaning preservation across different surface forms

**Example:**

```
geldiği → gel (root) + di (past) + ğ (welding) + i (possessive)
        → gel + <past-suffix> + <possessive-suffix>
```

### 2. Morphological Analysis Approach

Uses rule-based morphological analysis combined with statistical methods:

- Root identification using comprehensive Turkish root dictionaries
- Suffix segmentation based on Turkish morphological rules
- Integration with ITU NLP tools for validation
- Frequency-based vocabulary optimization

### 3. BPE Integration

Combines traditional BPE with Turkish-specific preprocessing:

- Morphology-aware subword segmentation
- Custom vocabulary with high-frequency Turkish roots and suffixes
- Integration with modern transformer tokenizers

## Technical Implementation

### Data Sources

- Turkish Wikipedia corpus
- ITU Turkish NLP datasets
- Custom root and suffix dictionaries
- Frequency-analyzed word lists

### Key Algorithms

- Longest root matching for morphological segmentation
- Phonological rule application for surface form generation
- BPE training with linguistic constraints
- Semantic mapping for cross-lingual representation

## Research Applications

This tokenizer project supports research in:

- **Machine Translation**: Better handling of Turkish morphology
- **Language Modeling**: Improved representation of Turkish text
- **Cross-lingual NLP**: Semantic token mapping across languages
- **Morphological Analysis**: Automated Turkish text analysis

## Development Status

This is an active research project with ongoing development in multiple areas:

- ✅ Basic morphological segmentation
- ✅ Semantic token mapping for common suffixes
- ✅ BPE tokenizer training pipeline
- ✅ GUI interface for interactive analysis
- 🔄 Cross-lingual semantic mapping
- 🔄 Performance optimization
- 🔄 Comprehensive evaluation metrics

## Contributing

This is a research project. For collaboration or questions:

1. Review the methodology in `semantic_tokenizer/guncel_strateji.md`
2. Examine the implementation notebooks
3. Check the current development status in individual README files

## Technical Requirements

- Python 3.8+
- Jupyter Notebook
- NumPy, Pandas for data processing
- PyTorch for neural components
- Transformers library for modern tokenizer integration

## Research Context

This project contributes to the field of morphologically rich language processing, specifically addressing challenges in Turkish NLP:

- Agglutinative morphology handling
- Semantic representation across morphological variants
- Integration of linguistic knowledge with statistical methods
- Cross-lingual semantic mapping for multilingual applications

---

**Note**: This is a research project focused on advancing Turkish language processing. The implementations are experimental and designed for research purposes.
