# Adaptive News Scraper with Machine Learning

An intelligent web scraper that extracts news articles from [yogonet.com](https://yogonet.com) using machine learning and semantic analysis. Unlike traditional scrapers that break when HTML structure changes, this system **automatically adapts** to layout modifications by learning patterns instead of relying on hardcoded CSS selectors.

## 🎯 What It Does

Extracts four key elements from news articles:
- **Title**: Main article headline
- **Kicker**: Category/section label (e.g., "Breaking News", "Exclusive")
- **Image URL**: Article thumbnail/featured image
- **Link URL**: Article page URL

## 🧠 How It Works

### The Problem with Traditional Scrapers

Traditional scrapers use fixed CSS selectors like `.article-title` or `#news-heading`. When websites redesign their HTML, these selectors break, requiring manual updates.

### The ML-Powered Solution

This scraper uses a **three-stage pipeline** that learns from examples:

```
Training Data → ML Weight Optimization → Adaptive Extraction
```

---

## 🏗️ Architecture & Flow

### Stage 1: Data Collection & Labeling

**Purpose**: Gather training data with ground truth annotations

**Script**: `portal/spiders/data_extractor.py`

```python
# Collects HTML pages with manual annotations
{
  "html_content": "<html>...</html>",
  "annotations": {
    "container_selector": ".item_noticias",
    "title_selector": ".titulo_item_listado_noticias a::text",
    "kicker_selector": ".volanta_item_listado_noticias a::attr(title)",
    "image_selector": ".imagen_item_listado_noticias img::attr(src)",
    "link_selector": ".volanta_item_listado_noticias a::attr(href)"
  }
}
```

**Output**: `output/extracted_data_*.json` (labeled training samples)

---

### Stage 2: Semantic Example Building

**Purpose**: Create a database of known good titles/kickers for semantic validation

**Script**: `build_semantic_examples.py`

**Process**:
1. Loads labeled training data from `output/*.json`
2. Extracts real title and kicker examples using ground truth selectors
3. Generates sentence embeddings using `sentence-transformers` (MiniLM-L6-v2)
4. Saves to `semantic_examples.json` for later similarity comparisons

**Why This Matters**: Helps distinguish titles from kickers when HTML structure is ambiguous (e.g., both are in `<div>` tags)

```bash
python build_semantic_examples.py
```

**Output**: `semantic_examples.json` + `.semantic_cache/embeddings.json`

---

### Stage 3: ML Weight Optimization

**Purpose**: Find optimal feature weights that maximize extraction accuracy

**Script**: `train_optimal_weights.py`

**How It Works**:

1. **Load Training Data**: Reads labeled HTML samples from `output/`

2. **Grid Search**: Tests thousands of weight combinations for scoring features:
   ```python
   weights = {
     "title": {
       "tag_score": 0.25,        # h1/h2 vs div/span importance
       "position_score": 0.10,   # Earlier in container = better
       "length_score": 0.25,     # 30-200 chars optimal for titles
       "link_score": 0.15,       # Titles often wrapped in <a>
       "semantic_score": 0.25    # Similarity to known titles
     },
     "kicker": { ... },
     "image": { ... },
     "link": { ... }
   }
   ```

3. **Evaluate Each Combination**:
   - Extract articles using current weights
   - Compare to ground truth annotations
   - Calculate accuracy (correct fields / total fields)

4. **Save Best Weights**: The combination with highest accuracy is saved

```bash
python train_optimal_weights.py
# Choose [Q]uick (faster) or [F]ull (best accuracy)
```

**Output**:
- `trained_models/optimal_weights.pkl` (used by scraper)
- `trained_models/optimal_weights.json` (human-readable)

**Key Component**: `adaptive_scraper/ml_trainer.py` (`MLWeightOptimizer` class)

---

### Stage 4: Adaptive Extraction

**Purpose**: Scrape articles using the trained ML model

**Script**: `portal/spiders/portal_spider.py`

**Run Command**:
```bash
# ML-powered adaptive mode (recommended)
scrapy crawl portal_spider -a with_predictions=true

# Traditional mode with hardcoded selectors (fallback)
scrapy crawl portal_spider -a with_predictions=false
```

**Extraction Pipeline**:

#### 4.1 Container Discovery (`adaptive_scraper/container_discovery.py`)

Finds repeating article blocks without hardcoded selectors:

1. **Find Candidate Containers**: Look for `<div>`, `<article>`, `<section>`, `<li>` with:
   - Text content (>20 chars)
   - Links or images
   - Moderate depth (not too shallow/deep)

2. **Structural Fingerprinting**: Group similar elements by:
   ```python
   fingerprint = {
     "child_tags": "div:2-a:1-img:1",
     "links": 2,
     "images": 1,
     "depth": 5,
     "children_bin": 3  # ~30 descendant elements
   }
   ```

3. **Select Best Group**: Score groups by:
   - **Size** (more containers = more likely correct)
   - **Sibling proximity** (containers appear consecutively)
   - **Content quality** (50-500 char text length)
   - **Image presence** (news articles usually have images)

4. **Return Containers**: List of BeautifulSoup `Tag` objects

#### 4.2 Element Scoring (`adaptive_scraper/element_scorer.py`)

Scores each candidate element within a container using ML-optimized weights:

**For Titles**:
- Tag hierarchy (h1=1.0, h2=0.9, div=0.4)
- Position in container (earlier = better)
- Text length (30-200 chars ideal)
- Link wrapping (titles often clickable)
- **Semantic similarity** to known titles (via embeddings)

**For Kickers**:
- Tag preference (span=0.8, div=0.7)
- Text length (10-100 chars, shorter than titles)
- Uppercase/bold styling
- **Semantic similarity** to known kickers
- Proximity to title element

**For Images**:
- Size indicators in URL (1200, 800, "large")
- Not icon/logo
- Alt text presence

**For Links**:
- Article URL patterns (/news/, /article/)
- Wraps title or image
- Not social media/external

#### 4.3 Semantic Validation (`adaptive_scraper/semantic_validator.py`)

Uses `sentence-transformers` to validate candidates:

1. **Encode candidate text** into 384-dimensional vector
2. **Compare to known examples** via cosine similarity
3. **Return scores**:
   ```python
   {
     "title_similarity": 0.87,    # High = likely a title
     "kicker_similarity": 0.34,   # Low = unlikely a kicker
     "is_title": 0.72,            # Probability
     "is_kicker": 0.28
   }
   ```

#### 4.4 Final Extraction

1. **Extract best elements** from each container
2. **Calculate confidence** score using weighted average
3. **Filter by threshold** (default: 0.4 confidence)
4. **Return articles**:
   ```python
   {
     "title": "Playson partners with High Flyer Casino",
     "kicker": "Ontario",
     "image": "https://yogonet.com/images/123.jpg",
     "link": "/international/news/2025/...",
     "confidence": 0.78,
     "scores": {
       "title": 0.92,
       "kicker": 0.76,
       "image": 0.85,
       "link": 0.88
     }
   }
   ```

**Output**: `output/final_result.json`

---

## 📊 Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. DATA COLLECTION                            │
│  data_extractor spider → output/extracted_data_*.json           │
│  (HTML + manual annotations)                                     │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ├─────────────────────────────────────────────────┐
                 │                                                  │
                 ▼                                                  ▼
┌────────────────────────────────┐      ┌────────────────────────────────┐
│   2a. SEMANTIC EXAMPLES        │      │   2b. ML WEIGHT TRAINING       │
│   build_semantic_examples.py   │      │   train_optimal_weights.py      │
│                                 │      │                                 │
│   ┌──────────────────────┐    │      │   ┌──────────────────────┐     │
│   │ Load training data   │    │      │   │ Load training data   │     │
│   └──────────┬───────────┘    │      │   └──────────┬───────────┘     │
│              │                 │      │              │                  │
│              ▼                 │      │              ▼                  │
│   ┌──────────────────────┐    │      │   ┌──────────────────────┐     │
│   │ Extract title/kicker │    │      │   │ Grid search weights  │     │
│   │ using ground truth   │    │      │   │ (1000+ combinations) │     │
│   └──────────┬───────────┘    │      │   └──────────┬───────────┘     │
│              │                 │      │              │                  │
│              ▼                 │      │              ▼                  │
│   ┌──────────────────────┐    │      │   ┌──────────────────────┐     │
│   │ Generate embeddings  │    │      │   │ Evaluate accuracy    │     │
│   │ (sentence-transformer)│   │      │   │ for each combination │     │
│   └──────────┬───────────┘    │      │   └──────────┬───────────┘     │
│              │                 │      │              │                  │
│              ▼                 │      │              ▼                  │
│   semantic_examples.json       │      │   trained_models/               │
│                                 │      │   optimal_weights.pkl           │
└─────────────────────────────────┘      └────────────────────────────────┘
                 │                                      │
                 │                                      │
                 └──────────────┬───────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3. ADAPTIVE EXTRACTION                        │
│         scrapy crawl portal_spider -a with_predictions=true      │
│                                                                   │
│   ┌───────────────────────────────────────────────────┐         │
│   │  AdaptiveExtractor (uses trained weights)        │         │
│   │                                                    │         │
│   │  Step 1: ContainerDiscovery                      │         │
│   │  ├─ Find repeating DOM patterns                  │         │
│   │  ├─ Group by structural fingerprint              │         │
│   │  └─ Select best group                            │         │
│   │                                                    │         │
│   │  Step 2: ElementScorer (ML weights)              │         │
│   │  ├─ Score title candidates                       │         │
│   │  ├─ Score kicker candidates                      │         │
│   │  ├─ Score image candidates                       │         │
│   │  └─ Score link candidates                        │         │
│   │                                                    │         │
│   │  Step 3: SemanticValidator                       │         │
│   │  ├─ Encode candidate text                        │         │
│   │  ├─ Compare to known examples                    │         │
│   │  └─ Return similarity scores                     │         │
│   │                                                    │         │
│   │  Step 4: Confidence Filtering                    │         │
│   │  └─ Keep articles with confidence > 0.4          │         │
│   └───────────────────────────────────────────────────┘         │
│                                                                   │
│   Output: output/final_result.json                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python ≥3.12
- [uv](https://docs.astral.sh/uv/) (fast Python package installer)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd scraper-ayi

# Install dependencies with uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Usage

#### Option 1: Use Pre-Trained Model (Recommended)

If `trained_models/optimal_weights.pkl` exists:

```bash
scrapy crawl portal_spider -a with_predictions=true
```

#### Option 2: Train Your Own Model

```bash
# 1. Collect training data (optional - already provided)
scrapy crawl data_extractor

# 2. Build semantic examples
python build_semantic_examples.py

# 3. Train ML weights
python train_optimal_weights.py
# Choose [Q]uick or [F]ull when prompted

# 4. Run scraper with trained model
scrapy crawl portal_spider -a with_predictions=true
```

---

## 📁 Project Structure

```
scraper-ayi/
├── adaptive_scraper/              # ML extraction engine
│   ├── adaptive_extractor.py      # Main orchestrator
│   ├── container_discovery.py     # Pattern recognition
│   ├── element_scorer.py          # Feature scoring
│   ├── semantic_validator.py      # Embedding-based validation
│   └── ml_trainer.py              # Weight optimization
│
├── portal/                        # Scrapy project
│   ├── spiders/
│   │   ├── portal_spider.py       # Main production spider
│   │   └── data_extractor.py      # Training data collector
│   ├── items.py                   # Data models
│   ├── pipelines.py               # Post-processing
│   └── settings.py                # Scrapy config
│
├── core/
│   └── settings.py                # ML hyperparameters
│
├── trained_models/
│   ├── optimal_weights.pkl        # Trained ML model
│   └── optimal_weights.json       # Human-readable weights
│
├── output/                        # Training data & results
│   ├── extracted_data_*.json      # Labeled training samples
│   └── final_result.json          # Scraping output
│
├── build_semantic_examples.py     # Build title/kicker database
├── train_optimal_weights.py       # ML training script
├── semantic_examples.json         # Known good examples
└── pyproject.toml                 # Dependencies
```

---

## 🔧 Configuration

### Tunable Parameters (`core/settings.py`)

```python
# Container discovery
DEFAULT_MIN_CONTAINERS = 3        # Min repeating patterns needed
DEFAULT_MAX_DEPTH = 20            # Max DOM depth to search

# Extraction confidence
DEFAULT_CONFIDENCE_THRESHOLD = 0.4  # Min score to accept (0-1)

# Element importance weights
TITLE_WEIGHT = 0.4
LINK_WEIGHT = 0.3
KICKER_WEIGHT = 0.15
IMAGE_WEIGHT = 0.15

# Text length thresholds
MINIMUM_TITLE_LENGTH = 10
MINIMUM_KICKER_LENGTH = 11

# Semantic model
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers model
```

---

## 🎓 Key Innovations

### 1. **Structural Fingerprinting**
Instead of CSS selectors, the system creates "fingerprints" of HTML structure:
```python
"tags:div:2-a:1-img:1|links:2|imgs:1|depth:5|children_bin:3"
```
This allows finding similar containers even if class names change.

### 2. **ML-Optimized Feature Weights**
Grid search tests thousands of combinations to find optimal balance:
- Tag type importance vs position vs length vs semantics
- Trained on real labeled data from yogonet.com
- Typically achieves 85-95% accuracy

### 3. **Semantic Validation**
Uses transformer models (BERT-based) to understand text meaning:
- Can distinguish "Breaking News" (kicker) from "Nevada Gaming Revenue Hits Record" (title)
- Even when both are in identical `<div class="text">` tags
- Based on learned patterns from 100+ real examples

### 4. **Adaptive to Changes**
When yogonet.com changes from:
```html
<div class="old-title">Article Title</div>
```
to:
```html
<h2 class="new-headline">Article Title</h2>
```
The scraper automatically adapts because it scores by **semantic meaning + structural patterns**, not fixed selectors.

---

## 📊 Performance

- **Accuracy**: 85-95% on test data (varies by weight optimization)
- **Speed**: ~2-5 seconds per page (including semantic validation)
- **Adaptability**: Works across different yogonet.com sections without reconfiguration
- **Robustness**: Continues working when HTML structure changes

---

## 🛠️ Development

```bash
# Activate environment
source .venv/bin/activate

# Run with debugging
ipython
>>> from adaptive_scraper.adaptive_extractor import AdaptiveExtractor
>>> extractor = AdaptiveExtractor(use_ml_weights=True)
>>> with open('response.html') as f:
...     html = f.read()
>>> articles = extractor.extract(html)

# Get detailed extraction info
>>> details = extractor.extract_with_details(html)
>>> print(details['discovery_info'])
>>> print(details['extracted_count'])

# Analyze specific container
>>> candidates = extractor.get_element_candidates_details(html, container_index=0)
>>> print(candidates['title_candidates'])
```

---

## 📝 Dependencies

Key libraries (see `pyproject.toml` for full list):

- **scrapy** ≥2.13.3 - Web scraping framework
- **beautifulsoup4** ≥4.14.2 - HTML parsing
- **sentence-transformers** ≥5.1.1 - Semantic embeddings
- **torch** ≥2.9.0 - Neural network backend
- **xgboost** ≥3.1.0 - ML utilities
- **pandas** ≥2.3.3 - Data processing

---

## 🤝 Contributing

To extend the scraper to new websites:

1. **Collect training data** from the target site
2. **Annotate containers** with correct selectors
3. **Re-train weights** with `train_optimal_weights.py`
4. **Build semantic examples** for the new site's content style
5. **Update spider** URLs in `portal_spider.py`

The system is designed to be **site-agnostic** - it learns patterns rather than hardcoding logic.

---

## 🙏 Acknowledgments

Built with:
- [Scrapy](https://scrapy.org/) - Web scraping framework
- [Sentence Transformers](https://www.sbert.net/) - Semantic similarity
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) - HTML parsing
