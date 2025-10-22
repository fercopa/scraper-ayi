"""
Adaptive Scraper - AI-Assisted Dynamic Web Scraping

This module provides intelligent, selector-free web scraping that adapts
to changes in HTML structure automatically.
"""

from .container_discovery import ContainerDiscovery
from .element_scorer import ElementScorer
from .adaptive_extractor import AdaptiveExtractor

__all__ = ["ContainerDiscovery", "ElementScorer", "AdaptiveExtractor"]
