"""
ICKG Extraction Demo with Gemini API
====================================
Stage 1 - Notebook 02: Demonstrates the Knowledge Graph Extraction pipeline.
The paper uses GPT-4 to generate seed training data. We use Gemini 1.5 Flash 
as a fast, cost-effective alternative to achieve the same result.

Usage:
1. Replace "YOUR_GEMINI_API_KEY" with your actual API key.
2. Run `python 02_ickg_extraction_demo.py`
"""

import os
import google.generativeai as genai

# ── Configuration ───────────────────────────────────────────────────
# TODO: Insert your Gemini API key here before running
API_KEY = "YOUR_GEMINI_API_KEY" 

if API_KEY == "YOUR_GEMINI_API_KEY":
    print("⚠️ Please edit this file and insert your Gemini API key to run the demo.")
    print("Exiting...")
    exit(1)

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ── Prompt Template ─────────────────────────────────────────────────
# This prompt mirrors the instructions provided to the LLM in the FinDKG paper
EXTRACTION_PROMPT = """You are a financial knowledge graph extraction system.
Given a financial news article, extract structured quintuples in the format:
(Subject Entity, Relation, Object Entity, Entity_Type_Subject, Entity_Type_Object)

Entity Types: Company, Person, Country, Sector, Technology, Currency, 
              Commodity, Organization, Index, Policy, Event, Other

Relation Types: Impact, Acquire, Partner, Invest, Regulate, Compete, 
                Supply, Employ, Litigate, Merge, IPO, Dividend, 
                Bankruptcy, Restructure, Spinoff

Rules:
1. Extract ALL relevant quintuples from the article
2. Each entity must be classified into one of the 12 entity types
3. Each relation must be one of the 15 relation types
4. Return ONLY a valid JSON array of quintuples

Article: {article_text}

Output format (JSON array):
[
  {{"subject": "...", "relation": "...", "object": "...", "subject_type": "...", "object_type": "..."}}
]
"""

# ── Sample Data ─────────────────────────────────────────────────────
sample_articles = [
    "Apple Inc. announced the acquisition of Intel's smartphone modem business for approximately $1 billion, strengthening its chip-making capabilities.",
    "Tesla's stock surged 12% after the company reported record quarterly deliveries, beating Wall Street expectations.",
    "The Federal Reserve raised interest rates by 25 basis points, impacting banking sector stocks across the S&P 500.",
    "Microsoft partnered with OpenAI in a $10 billion investment deal to accelerate artificial intelligence research.",
    "JPMorgan Chase announced plans to acquire First Republic Bank after regulators seized the failing lender.",
]

# ── Execution ───────────────────────────────────────────────────────
print("=" * 60)
print("Initiating ICKG Extraction Pipeline via Gemini...")
print("=" * 60)

for i, article in enumerate(sample_articles, 1):
    print(f"\n[Article {i}] {article[:80]}...")
    try:
        response = model.generate_content(EXTRACTION_PROMPT.format(article_text=article))
        print("Extracted Quintuples:")
        print(response.text)
    except Exception as e:
        print(f"Error during extraction: {e}")
