from __future__ import annotations
from types import SimpleNamespace
from typing import List, Optional, Dict, Any

import pandas as pd
from skill_framework import SkillInput, SkillVisualization, skill, SkillParameter, SkillOutput, ParameterDisplayDescription
from skill_framework.skills import ExportData
from skill_framework.layouts import wire_layout

import json
import os
import glob
import traceback
from jinja2 import Template
import base64
import io
from PIL import Image
import logging
import re
import html
import fitz  # PyMuPDF for PDF thumbnail generation
import numpy as np
from typing import List
import requests

logger = logging.getLogger(__name__)

@skill(
    name="Bakery Market Intelligence Explorer",
    description="Provides facts and context for a topic of discussion. Use this skill if a user asks to build a table. Only run this skill if a user mentions challenges in the baked goods market, or 'search the knowledge base for..'",
    capabilities="Searches through bakery industry market research reports to answer questions about market size, regional trends, product segments, competitive landscape, financial forecasts, raw material costs, and growth opportunities",
    limitations="Limited to documents in the knowledge base, requires pre-processed document chunks in pack.json",
    parameters=[
        SkillParameter(
            name="user_question",
            description="The question to answer using the knowledge base",
            required=True
        ),
        SkillParameter(
            name="base_url",
            parameter_type="code",
            description="Base URL for document links",
            required=True,
            default_value="https://maxdemo.staging.answerrocket.com/apps/system/knowledge-base"
        ),
        SkillParameter(
            name="max_sources",
            description="Maximum number of source documents to include",
            default_value=5
        ),
        SkillParameter(
            name="match_threshold",
            description="Minimum similarity score for document matching (0-1)",
            default_value=0.15
        ),
        SkillParameter(
            name="max_characters",
            description="Maximum characters to include from sources",
            default_value=3000
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt for the insights section (left panel)",
            default_value="Thank you for your question! I've searched through the available bakery market intelligence documents. Please check the response and sources tabs above for detailed analysis with citations and document references. Feel free to ask follow-up questions if you need clarification on any of the findings."
        ),
        SkillParameter(
            name="response_layout",
            parameter_type="visualization",
            description="Layout for Response tab",
            default_value='{"layoutJson": {"type": "Document", "children": [{"name": "ResponseText", "type": "Paragraph", "text": "{{response_content}}"}]}, "inputVariables": [{"name": "response_content", "isRequired": false, "defaultValue": null, "targets": [{"elementName": "ResponseText", "fieldName": "text"}]}]}'
        ),
        SkillParameter(
            name="sources_layout",
            parameter_type="visualization",
            description="Layout for Sources tab",
            default_value='{"layoutJson": {"type": "Document", "children": [{"name": "SourcesText", "type": "Paragraph", "text": "{{sources_content}}"}]}, "inputVariables": [{"name": "sources_content", "isRequired": false, "defaultValue": null, "targets": [{"elementName": "SourcesText", "fieldName": "text"}]}]}'
        )
    ]
)
def bakery_market_explorer(parameters: SkillInput):
    """Main skill function for bakery market intelligence exploration"""

    # Get parameters
    user_question = parameters.arguments.user_question
    base_url = parameters.arguments.base_url
    max_sources = parameters.arguments.max_sources or 5
    match_threshold = parameters.arguments.match_threshold or 0.15
    max_characters = parameters.arguments.max_characters or 3000
    max_prompt = parameters.arguments.max_prompt

    # Initialize empty topics list
    list_of_topics = []

    # Initialize results
    main_html = ""
    sources_html = ""
    title = "Bakery Market Analysis"
    response_data = None

    try:
        # Load document sources from pack.json
        loaded_sources = load_document_sources()

        if not loaded_sources:
            return SkillOutput(
                final_prompt="No document sources found. Please ensure pack.json is available.",
                narrative=None,
                visualizations=[],
                export_data=[]
            )

        # Find matching documents
        logger.info(f"DEBUG: Searching for documents matching: '{user_question}'")
        docs = find_matching_documents(
            user_question=user_question,
            topics=list_of_topics,
            loaded_sources=loaded_sources,
            base_url=base_url,
            max_sources=max_sources,
            match_threshold=match_threshold,
            max_characters=max_characters
        )
        logger.info(f"DEBUG: Found {len(docs) if docs else 0} matching documents")

        if not docs:
            # No results found
            logger.warning("DEBUG: No matching documents found for query")
            no_results_html = """
            <div style="text-align: center; padding: 40px; color: #666;">
                <h2>No relevant documents found</h2>
                <p>No documents in the knowledge base matched your question with sufficient relevance.</p>
                <p>Try rephrasing your question or using different keywords.</p>
            </div>
            """
            main_html = no_results_html
            sources_html = "<p>No sources available</p>"
            title = "No Results Found"
        else:
            # Generate response from documents
            logger.info(f"DEBUG: Generating RAG response from {len(docs)} documents")
            response_data = generate_rag_response(user_question, docs)
            logger.info(f"DEBUG: Response generated: {bool(response_data)}")

            # Create main response HTML (without sources section)
            if response_data:
                try:
                    main_html = force_ascii_replace(
                        Template(main_response_template).render(
                            title=response_data['title'],
                            content=response_data['content']
                        )
                    )
                    logger.info(f"DEBUG: Generated main HTML, length: {len(main_html)}")

                    # Create separate sources HTML
                    sources_html = force_ascii_replace(
                        Template(sources_template).render(
                            references=response_data['references']
                        )
                    )
                    logger.info(f"DEBUG: Generated sources HTML, length: {len(sources_html)}")
                    title = response_data['title']
                except Exception as e:
                    logger.error(f"DEBUG: Error rendering HTML templates: {str(e)}")
                    import traceback
                    logger.error(f"DEBUG: Template error traceback: {traceback.format_exc()}")
                    main_html = f"<p>Error rendering content: {str(e)}</p>"
                    sources_html = "<p>Error rendering sources</p>"
                    title = "Template Error"
            else:
                main_html = "<p>Error generating response from documents.</p>"
                sources_html = "<p>Error loading sources</p>"
                title = "Error"

    except Exception as e:
        logger.error(f"ERROR in bakery market explorer: {str(e)}")
        import traceback
        logger.error(f"ERROR: Full traceback:\n{traceback.format_exc()}")
        main_html = f"<p>Error processing request: {str(e)}</p>"
        sources_html = "<p>Error loading sources</p>"
        title = "Error"

    # Create content variables for wire_layout
    # Prepare content for response tab
    references_content = ""
    if response_data and response_data.get('references'):
        references_content = f"""
        <hr style="margin: 20px 0;">
        <h3>References</h3>
        {create_references_list(response_data['references'])}
        """

    response_content = f"""
    <div style="padding: 20px;">
        {main_html}
        {references_content}
    </div>
    """

    # Prepare content for sources tab
    sources_content = f"""
    <div style="padding: 20px;">
        <h2>Document Sources</h2>
        {create_sources_table(response_data['references']) if response_data and response_data.get('references') else sources_html}
    </div>
    """

    # Create visualizations using wire_layout
    visualizations = []

    try:
        logger.info(f"DEBUG: Creating response tab with title: {title}")

        # Response tab
        response_vars = {"response_content": response_content}
        response_layout_json = json.loads(parameters.arguments.response_layout)
        rendered_response = wire_layout(response_layout_json, response_vars)
        visualizations.append(SkillVisualization(title=title, layout=rendered_response))

        # Sources tab
        sources_vars = {"sources_content": sources_content}
        sources_layout_json = json.loads(parameters.arguments.sources_layout)
        rendered_sources = wire_layout(sources_layout_json, sources_vars)
        visualizations.append(SkillVisualization(title="Sources", layout=rendered_sources))

        logger.info(f"DEBUG: Total visualizations created: {len(visualizations)}")

    except Exception as e:
        logger.error(f"ERROR: Failed to create visualizations: {str(e)}")
        import traceback
        logger.error(f"ERROR: Full traceback: {traceback.format_exc()}")

        # Fallback to simple HTML if wire_layout fails
        logger.info("DEBUG: Falling back to simple HTML visualizations")
        simple_response_html = f"<div style='padding:20px;'>{main_html}{references_content}</div>"
        simple_sources_html = f"<div style='padding:20px;'><h2>Document Sources</h2>{sources_html}</div>"

        visualizations = [
            SkillVisualization(title=title, layout=simple_response_html),
            SkillVisualization(title="Sources", layout=simple_sources_html)
        ]

    # Return skill output
    return SkillOutput(
        final_prompt=max_prompt,
        narrative=None,
        visualizations=visualizations,
        export_data=[]
    )

# Helper Functions and Templates

def create_references_list(references):
    """Create clickable references list HTML"""
    if not references:
        return "<p>No references available</p>"

    html_parts = ["<ol style='list-style-type: decimal; padding-left: 20px;'>"]
    for ref in references:
        html_parts.append(f"""
            <li style='margin-bottom: 10px;'>
                <a href='{ref.get('url', '#')}' target='_blank' style='color: #0066cc; text-decoration: none;'>
                    {ref.get('text', 'Document')} (Page {ref.get('page', '?')})
                </a>
            </li>
        """)
    html_parts.append("</ol>")
    return ''.join(html_parts)

def get_pdf_thumbnail(pack_file_path, file_name, page_num, image_height=300, image_width=400):
    """Generate real PDF thumbnail using knowledge base API"""
    logger.info(f"DEBUG THUMBNAIL: ==> Starting thumbnail generation for {file_name} page {page_num}")

    # Knowledge base API not available in skill environment, use fallback
    return create_fallback_thumbnail(file_name, page_num, image_width, image_height)


def create_fallback_thumbnail(file_name, page_num, image_width, image_height):
    """Create a clean fallback thumbnail when PDF rendering fails"""
    logger.info(f"DEBUG FALLBACK: ==> Creating fallback thumbnail for {file_name} page {page_num}")
    try:
        from PIL import ImageDraw, ImageFont

        # Create a clean document-style thumbnail
        placeholder_image = Image.new('RGB', (image_width, image_height), color='#f8f9fa')
        draw = ImageDraw.Draw(placeholder_image)

        # Add a subtle border
        draw.rectangle([0, 0, image_width-1, image_height-1], outline='#dee2e6', width=1)

        # Add a document icon in the center
        icon_size = min(image_width, image_height) // 3
        icon_x = (image_width - icon_size) // 2
        icon_y = (image_height - icon_size) // 2 - 20

        # Draw document shape
        draw.rectangle([icon_x, icon_y, icon_x + icon_size, icon_y + icon_size],
                      fill='white', outline='#6c757d', width=2)

        # Add fold corner
        corner_size = icon_size // 4
        draw.polygon([(icon_x + icon_size - corner_size, icon_y),
                     (icon_x + icon_size, icon_y + corner_size),
                     (icon_x + icon_size - corner_size, icon_y + corner_size)],
                    fill='#e9ecef', outline='#6c757d')

        # Add text lines in document
        line_spacing = icon_size // 8
        for i in range(3):
            y_pos = icon_y + icon_size // 3 + i * line_spacing
            draw.line([(icon_x + icon_size // 6, y_pos), (icon_x + icon_size - icon_size // 6, y_pos)],
                     fill='#adb5bd', width=1)

        # Add file name and page number below icon
        try:
            font = ImageFont.load_default()
        except:
            font = None

        # Document name
        doc_name = file_name[:20] + "..." if len(file_name) > 20 else file_name
        text_bbox = draw.textbbox((0, 0), doc_name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (image_width - text_width) // 2
        text_y = icon_y + icon_size + 15
        draw.text((text_x, text_y), doc_name, fill='#495057', font=font)

        # Page number
        page_text = f"Page {page_num}"
        page_bbox = draw.textbbox((0, 0), page_text, font=font)
        page_width = page_bbox[2] - page_bbox[0]
        page_x = (image_width - page_width) // 2
        draw.text((page_x, text_y + 18), page_text, fill='#6c757d', font=font)

        # Convert to base64
        buffered = io.BytesIO()
        placeholder_image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.info(f"DEBUG: Created fallback thumbnail for {file_name} page {page_num}")
        return image_base64

    except Exception as e:
        logger.error(f"DEBUG: Failed to create fallback thumbnail: {e}")
        return None

def create_sources_table(references):
    """Create sources table HTML"""
    if not references:
        return "<p>No sources available</p>"

    html_parts = [
        """<table style='width: 100%; border-collapse: collapse; font-size: 14px;'>
        <thead>
            <tr style='background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;'>
                <th style='padding: 12px; text-align: left; font-weight: 600;'>Document Name</th>
                <th style='padding: 12px; text-align: left; font-weight: 600;'>Page</th>
                <th style='padding: 12px; text-align: left; font-weight: 600;'>Match Score</th>
            </tr>
        </thead>
        <tbody>"""
    ]

    for i, ref in enumerate(references):
        bg_color = '#ffffff' if i % 2 == 0 else '#f8f9fa'
        match_score = ref.get('match_score', '0.780000') if hasattr(ref, 'get') else '0.780000'

        html_parts.append(f"""
            <tr style='background-color: {bg_color}; border-bottom: 1px solid #dee2e6;'>
                <td style='padding: 12px;'>
                    <a href='{ref.get('url', '#')}' target='_blank' style='color: #0066cc; text-decoration: none;'>
                        {ref.get('src', ref.get('text', 'Document'))}
                    </a>
                </td>
                <td style='padding: 12px;'>{ref.get('page', '?')}</td>
                <td style='padding: 12px;'>{match_score}</td>
            </tr>
        """)

    html_parts.append("</tbody></table>")
    return ''.join(html_parts)

def load_document_sources():
    """Load document sources from pack.json bundled with the skill"""
    loaded_sources = []

    try:
        # First, try to load pack.json from the same directory as this skill file
        skill_dir = os.path.dirname(os.path.abspath(__file__))
        pack_file = os.path.join(skill_dir, "pack.json")

        logger.info(f"DEBUG: Looking for pack.json in skill directory: {pack_file}")

        # Check if pack.json exists in the skill directory
        if not os.path.exists(pack_file):
            # Try looking in a 'data' subdirectory
            data_dir = os.path.join(skill_dir, "data")
            pack_file_data = os.path.join(data_dir, "pack.json")

            if os.path.exists(pack_file_data):
                pack_file = pack_file_data
                logger.info(f"DEBUG: Found pack.json in data directory: {pack_file}")
            else:
                # Fallback: try the Skill Resources path if environment variables are available
                logger.info(f"DEBUG: pack.json not found in skill bundle, trying Skill Resources as fallback")

                try:
                    from ar_paths import ARTIFACTS_PATH
                    logger.info(f"DEBUG: Successfully imported ARTIFACTS_PATH: {ARTIFACTS_PATH}")
                except ImportError as e:
                    logger.info(f"DEBUG: Could not import ar_paths, using environment variable: {e}")
                    ARTIFACTS_PATH = os.environ.get('AR_DATA_BASE_PATH', '/artifacts')

                # Get environment variables for path construction
                tenant = os.environ.get('AR_TENANT_ID', 'maxstaging')
                copilot = os.environ.get('AR_COPILOT_ID', '')
                skill_id = os.environ.get('AR_COPILOT_SKILL_ID', '')

                if copilot and skill_id:
                    resource_path = os.path.join(
                        ARTIFACTS_PATH,
                        tenant,
                        "skill_workspaces",
                        copilot,
                        skill_id,
                        "pack.json"
                    )
                    if os.path.exists(resource_path):
                        pack_file = resource_path
                        logger.info(f"DEBUG: Found pack.json in Skill Resources: {pack_file}")
                    else:
                        pack_file = None
                        logger.warning(f"DEBUG: No pack.json found in bundle or Skill Resources")
                else:
                    pack_file = None
                    logger.warning(f"DEBUG: No pack.json found and missing environment variables for Skill Resources")
        else:
            logger.info(f"DEBUG: Found pack.json in skill bundle: {pack_file}")

        if pack_file and os.path.exists(pack_file):
            logger.info(f"Loading documents from: {pack_file}")
            with open(pack_file, 'r', encoding='utf-8') as f:
                resource_contents = json.load(f)
                logger.info(f"DEBUG: Loaded JSON structure type: {type(resource_contents)}")

                # Handle different pack.json formats
                if isinstance(resource_contents, list):
                    logger.info(f"DEBUG: Processing {len(resource_contents)} files from pack.json")
                    # Format: [{"File": "doc.pdf", "Chunks": [{"Text": "...", "Page": 1}]}]
                    for processed_file in resource_contents:
                        file_name = processed_file.get("File", "unknown_file")
                        chunks = processed_file.get("Chunks", [])
                        logger.info(f"DEBUG: Processing file '{file_name}' with {len(chunks)} chunks")
                        for chunk in chunks:
                            res = {
                                "file_name": file_name,
                                "text": chunk.get("Text", ""),
                                "description": str(chunk.get("Text", ""))[:200] + "..." if len(str(chunk.get("Text", ""))) > 200 else str(chunk.get("Text", "")),
                                "chunk_index": chunk.get("Page", 1),
                                "citation": file_name
                            }
                            loaded_sources.append(res)
                else:
                    logger.warning(f"Unexpected pack.json format - expected array of files, got: {type(resource_contents)}")
        else:
            logger.warning("pack.json not found in any expected locations")

    except Exception as e:
        logger.error(f"Error loading pack.json: {str(e)}")
        import traceback
        logger.error(f"DEBUG: Full traceback: {traceback.format_exc()}")

    logger.info(f"Loaded {len(loaded_sources)} document chunks from pack.json")
    return loaded_sources

def find_matching_documents(user_question, topics, loaded_sources, base_url, max_sources, match_threshold, max_characters):
    """Find documents using enhanced keyword matching optimized for bakery market intelligence"""
    logger.info("DEBUG: Starting enhanced keyword matching for bakery market intelligence")
    logger.info(f"DEBUG: User question: {user_question}")
    logger.info(f"DEBUG: Match threshold: {match_threshold}")

    try:
        import os

        logger.info(f"DEBUG: Matching against {len(loaded_sources)} document sources")

        # Analyze question intent
        question_lower = user_question.lower() if user_question else ""

        # Build comprehensive search terms
        search_terms = []
        if user_question:
            search_terms.append(user_question)

        # Bakery market keyword expansions
        keyword_expansions = {
            'market': ['market', 'industry', 'sector', 'segment', 'regional', 'global', 'revenue', 'sales'],
            'growth': ['growth', 'increase', 'expansion', 'rise', 'growing', 'trend', 'projection', 'outlook', 'forecast', 'cagr'],
            'product': ['product', 'biscuits', 'bread', 'rolls', 'cakes', 'pastries', 'pizza', 'crusts', 'rusks', 'cookies'],
            'specialty': ['specialty', 'gluten-free', 'fortified', 'organic', 'low-calorie', 'sugar-free'],
            'distribution': ['distribution', 'retail', 'direct', 'indirect', 'hypermarket', 'supermarket', 'convenience', 'online', 'bakery'],
            'financial': ['financial', 'revenue', 'profit', 'cost', 'price', 'spending', 'investment', 'billion', 'million'],
            'regional': ['regional', 'north america', 'europe', 'asia pacific', 'emea', 'latin america', 'mea', 'u.s.', 'canada'],
            'challenges': ['challenges', 'competition', 'raw material', 'fluctuation', 'regulation', 'compliance', 'quality'],
            'trends': ['trends', 'consumer', 'preference', 'health', 'convenience', 'artisanal', 'packaged', 'on-the-go'],
            'materials': ['wheat', 'sugar', 'dairy', 'flour', 'ingredients', 'raw material', 'commodity'],
            'risk': ['risk', 'inflation', 'economic', 'downturn', 'supply chain', 'regulatory', 'profitability']
        }

        # Add expansions based on keywords in question
        for key, expansions in keyword_expansions.items():
            if key in question_lower:
                search_terms.extend(expansions)

        search_terms.extend([topic for topic in topics if topic])

        # Remove duplicates
        seen = set()
        unique_terms = []
        for term in search_terms:
            if term and term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)

        logger.info(f"DEBUG: Expanded to {len(unique_terms)} unique search terms")

        # Score all documents
        scored_sources = []
        for source in loaded_sources:
            score = calculate_enhanced_relevance(source['text'], unique_terms, source['file_name'])

            if score >= float(match_threshold):
                source_copy = source.copy()
                source_copy['match_score'] = score

                # Generate URL for document
                doc_id = "97da127f-c557-4ade-acd4-a44ef7804aa9"
                source_copy['url'] = f"{base_url}/{doc_id}#page={source_copy['chunk_index']}"
                scored_sources.append(source_copy)

        logger.info(f"DEBUG: {len(scored_sources)} documents passed threshold")

        # Sort by score (descending)
        scored_sources.sort(key=lambda x: x['match_score'], reverse=True)

        # Select top matches respecting character and source limits
        matches = []
        chars_so_far = 0

        for source in scored_sources:
            if len(matches) >= int(max_sources):
                break

            # Check character limit (but ensure minimum 2 docs)
            if len(matches) >= 2 and chars_so_far + len(source['text']) > int(max_characters):
                break

            matches.append(source)
            chars_so_far += len(source['text'])

        logger.info(f"DEBUG: Selected {len(matches)} final documents")
        if matches:
            logger.info(f"DEBUG: Top scores: {[round(m['match_score'], 3) for m in matches[:3]]}")

        return [SimpleNamespace(**match) for match in matches]

    except Exception as e:
        logger.error(f"ERROR: Document matching failed: {e}")
        import traceback
        logger.error(f"ERROR: Full traceback: {traceback.format_exc()}")
        raise e


def calculate_enhanced_relevance(text, search_terms, file_name):
    """Enhanced relevance scoring optimized for bakery market intelligence"""
    text_lower = text.lower()
    score = 0.0

    # Critical bakery market keywords with weights
    critical_keywords = {
        # Market & Financial
        'market': 0.4, 'revenue': 0.5, 'billion': 0.6, 'million': 0.5, 'sales': 0.4,
        'investment': 0.4, 'cost': 0.4, 'profit': 0.5, 'financial': 0.4,

        # Growth & Trends
        'growth': 0.5, 'forecast': 0.5, 'outlook': 0.4, 'trend': 0.4, 'cagr': 0.6,
        'increase': 0.3, 'expansion': 0.3, 'projected': 0.4,

        # Product Categories
        'biscuits': 0.5, 'bread': 0.4, 'cakes': 0.4, 'pastries': 0.4, 'cookies': 0.4,
        'rolls': 0.3, 'pizza': 0.3, 'crusts': 0.3, 'rusks': 0.3,

        # Specialty Types
        'gluten-free': 0.5, 'organic': 0.5, 'fortified': 0.4, 'low-calorie': 0.4,
        'sugar-free': 0.4, 'artisanal': 0.4, 'specialty': 0.4,

        # Distribution
        'distribution': 0.4, 'retail': 0.3, 'hypermarket': 0.4, 'supermarket': 0.4,
        'convenience': 0.3, 'online': 0.3, 'direct': 0.3,

        # Regional
        'north america': 0.5, 'europe': 0.5, 'asia pacific': 0.5, 'emea': 0.5,
        'regional': 0.4, 'global': 0.4,

        # Challenges & Risks
        'competition': 0.5, 'raw material': 0.6, 'wheat': 0.5, 'sugar': 0.5, 'dairy': 0.5,
        'fluctuation': 0.5, 'inflation': 0.5, 'regulatory': 0.4, 'compliance': 0.4,
        'supply chain': 0.5, 'profitability': 0.5, 'risk': 0.4,

        # Consumer Trends
        'consumer': 0.4, 'preference': 0.4, 'health': 0.3, 'convenience': 0.3,
        'packaged': 0.3, 'on-the-go': 0.4
    }

    # Check critical keywords
    for keyword, weight in critical_keywords.items():
        if keyword in text_lower:
            count = min(text_lower.count(keyword), 3)  # Cap contribution
            score += count * weight

    # Check search terms
    for term in search_terms:
        if not term:
            continue

        term_lower = term.lower()

        # Exact phrase match (high value)
        if len(term_lower) > 3 and term_lower in text_lower:
            occurrences = text_lower.count(term_lower)
            score += min(occurrences * 0.6, 1.5)
            continue

        # Word matching
        words = term_lower.split()
        matched = 0

        for word in words:
            if len(word) < 3:
                continue
            if word in text_lower:
                matched += 1
                if word in critical_keywords:
                    score += critical_keywords[word]
                else:
                    score += 0.15

        # Completeness bonus
        if len(words) > 1 and matched / len(words) >= 0.7:
            score += 0.3

    # High-value bakery market patterns
    high_value_patterns = [
        ('raw material cost', 0.8),
        ('market size', 0.7),
        ('revenue potential', 0.7),
        ('market share', 0.7),
        ('growth driver', 0.7),
        ('competitive landscape', 0.6),
        ('financial forecast', 0.7),
        ('supply chain', 0.6),
        ('market dynamic', 0.6),
        ('bakery product', 0.6),
        ('distribution channel', 0.6)
    ]

    for pattern, weight in high_value_patterns:
        if pattern in text_lower:
            score += weight

    # File name bonus
    file_lower = file_name.lower()
    for term in search_terms:
        if term and term.lower() in file_lower:
            score += 0.2
            break

    # Normalize
    return min(score / 3.0, 1.0)


def generate_rag_response(user_question, docs):
    """Generate response using LLM with document context"""
    if not docs:
        return None

    # Build facts from documents for LLM prompt
    facts = []
    logger.info(f"DEBUG: Building prompt from {len(docs)} documents")
    for i, doc in enumerate(docs):
        facts.append(f"====== Source {i+1} ====")
        facts.append(f"File and page: {doc.file_name} page {doc.chunk_index}")
        facts.append(f"Description: {doc.description}")
        facts.append(f"Citation: {doc.url}")
        facts.append(f"Content: {doc.text}")
        facts.append("")
        logger.debug(f"DEBUG: Added source {i+1}: {doc.file_name} p{doc.chunk_index} ({len(doc.text)} chars)")

    # Create the prompt for the LLM
    prompt_template = Template(narrative_prompt)
    full_prompt = prompt_template.render(
        user_query=user_question,
        facts="\n".join(facts)
    )
    logger.info(f"DEBUG: Generated prompt length: {len(full_prompt)} chars")

    try:
        # Use ArUtils for LLM calls
        logger.info("DEBUG: Making LLM call with ArUtils")
        from ar_analytics import ArUtils
        ar_utils = ArUtils()
        llm_response = ar_utils.get_llm_response(full_prompt)

        logger.info(f"DEBUG: Got LLM response length: {len(llm_response)} chars")

        # Parse the LLM response
        def get_between_tags(content, tag):
            try:
                return content.split("<"+tag+">",1)[1].split("</"+tag+">",1)[0]
            except:
                pass
            return content

        title = get_between_tags(llm_response, "title") or f"Analysis: {user_question}"
        content = get_between_tags(llm_response, "content") or llm_response

        logger.info(f"DEBUG: Parsed title: {title[:50]}...")

    except Exception as e:
        logger.error(f"DEBUG: ArUtils LLM call failed: {e}")
        logger.info(f"DEBUG: Using fallback response generation")
        # Fallback with extraction of actual content
        title = f"Bakery Market Intelligence: {user_question}"
        content = f"<p>Based on the bakery market research documents, here's the relevant information:</p>"

        # Extract and present actual data from documents
        for i, doc in enumerate(docs):
            doc_text = str(doc.text) if doc.text else ""
            clean_text = doc_text.replace(f"START OF PAGE: {doc.chunk_index}", "").strip()
            clean_text = clean_text.replace(f"END OF PAGE: {doc.chunk_index}", "").strip()

            if clean_text and len(clean_text) > 20:
                # Extract specific data points
                lines = clean_text.split('\n')
                relevant_lines = []

                for line in lines:
                    line_lower = line.lower()
                    # Extract lines with important bakery market information
                    if any(keyword in line_lower for keyword in ['market', 'revenue', 'billion', 'growth', 'forecast', 'cagr', 'segment', 'regional', 'product', 'distribution', 'competition', 'raw material', 'cost', 'trend']):
                        relevant_lines.append(line.strip())

                if relevant_lines:
                    content += f"<h3>From {doc.file_name} (Page {doc.chunk_index})<sup>[{i+1}]</sup></h3>"
                    content += "<ul>"
                    for line in relevant_lines[:5]:  # Show top 5 most relevant lines
                        if line:
                            content += f"<li>{line}</li>"
                    content += "</ul>"

    # Build references with actual URLs and thumbnails
    references = []
    skill_dir = os.path.dirname(os.path.abspath(__file__))
    pack_file_path = os.path.join(skill_dir, "pack.json")

    for i, doc in enumerate(docs):
        # Create preview text (first 120 characters)
        doc_text = str(doc.text) if doc.text else ""
        preview_text = doc_text[:120] + "..." if len(doc_text) > 120 else doc_text

        # Generate thumbnail for this document
        logger.info(f"DEBUG: Generating thumbnail for reference {i+1}: {doc.file_name} page {doc.chunk_index}")
        thumbnail_base64 = get_pdf_thumbnail(pack_file_path, doc.file_name, doc.chunk_index, 120, 160)

        # Generate knowledge base URL
        doc_id = "97da127f-c557-4ade-acd4-a44ef7804aa9"
        doc.url = f"https://maxdemo.staging.answerrocket.com/apps/system/knowledge-base/{doc_id}#page={doc.chunk_index}"

        ref = {
            'number': i + 1,
            'url': doc.url,
            'src': doc.file_name,
            'page': doc.chunk_index,
            'text': f"Document: {doc.file_name}",
            'preview': preview_text,
            'thumbnail': thumbnail_base64 if thumbnail_base64 else "",
            'match_score': f"{doc.match_score:.3f}" if hasattr(doc, 'match_score') else "N/A"
        }
        references.append(ref)

    return {
        'title': title,
        'content': content,
        'references': references,
        'raw_prompt': full_prompt
    }

def force_ascii_replace(html_string):
    """Clean HTML string for safe rendering"""
    # Remove null characters
    cleaned = html_string.replace('\u0000', '')

    # Escape special characters, but preserve existing HTML entities
    cleaned = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)', '&amp;', cleaned)

    # Replace problematic characters with HTML entities
    cleaned = cleaned.replace('"', '&quot;')
    cleaned = cleaned.replace("'", '&#39;')
    cleaned = cleaned.replace('–', '&ndash;')
    cleaned = cleaned.replace('—', '&mdash;')
    cleaned = cleaned.replace('…', '&hellip;')

    # Convert curly quotes to straight quotes
    cleaned = cleaned.replace('"', '"').replace('"', '"')
    cleaned = cleaned.replace(''', "'").replace(''', "'")

    # Remove any remaining control characters
    cleaned = ''.join(ch for ch in cleaned if ord(ch) >= 32 or ch in '\n\r\t')

    return cleaned

# HTML Templates

narrative_prompt = """
You are analyzing bakery industry market intelligence documents. Extract and summarize ALL relevant information from the provided sources to answer the user's question. Be comprehensive and specific.

IMPORTANT INSTRUCTIONS:
1. Extract SPECIFIC details like market size figures, growth rates (CAGR), revenue data, product segments, regional trends, and financial forecasts
2. Include ALL relevant information found in the sources, not just general statements
3. If the sources contain relevant data, ALWAYS provide it - never say "no specific information"
4. For market size questions, look for dollar amounts, growth percentages, and forecasts (e.g., "USD 356 Billion by 2032, CAGR 4.4%")
5. For product segment questions, identify specific categories (biscuits, bread, cakes, specialty types) and their market shares
6. For regional questions, note specific regions (North America, Europe, Asia Pacific, EMEA) with revenue data and growth rates
7. For financial/profitability questions, extract information about raw material costs, pricing, competition, and economic factors
8. For trend questions, identify consumer preferences, distribution channels, and industry dynamics

Write a descriptive headline between <title> tags then detail ALL supporting information in HTML between <content> tags with citation references like <sup>[source number]</sup>.

Base your summary solely on the provided facts, avoiding assumptions.

### EXAMPLE
example_question: What is the market size forecast for bakery products?

====== Example Source 1 ====
File and page: Bakery_Market_2023-2032.pdf page 3
Description: Market size forecast information
Citation: https://maxdemo.staging.answerrocket.com/apps/system/knowledge-base/doc-id#page=3
Content: The Bakery Products Market represents a USD 356 Billion market opportunity by 2032, growing at a CAGR of 4.4%. The market was valued at USD 234.5 billion in 2022 with biscuits holding the majority market share.

====== Example Source 2 ====
File and page: Bakery_Market_2023-2032.pdf page 15
Description: Regional market data
Citation: https://maxdemo.staging.answerrocket.com/apps/system/knowledge-base/doc-id#page=15
Content: U.S. dominated the North American region with majority market share and a revenue of USD 42.2 billion in 2022 and is anticipated to expand at a significant pace from 2023-2032.

example_assistant: <title>Bakery Products Market Forecast: USD 356 Billion by 2032</title>
<content>
    <p>The global bakery products market is projected to reach <strong>USD 356 billion by 2032</strong>, growing at a compound annual growth rate (CAGR) of <strong>4.4%</strong> from 2023 to 2032.<sup>[1]</sup></p>

    <h3>Current Market Size</h3>
    <ul>
        <li>The market was valued at <strong>USD 234.5 billion in 2022</strong>, with biscuits holding the majority market share.<sup>[1]</sup></li>
    </ul>

    <h3>Regional Highlights</h3>
    <ul>
        <li>The <strong>United States dominated the North American region</strong> with a revenue of <strong>USD 42.2 billion in 2022</strong>.<sup>[2]</sup></li>
        <li>The U.S. market is anticipated to expand at a significant pace from 2023-2032.<sup>[2]</sup></li>
    </ul>
</content>

### The User's Question to Answer
Answer this question: {{user_query}}

{{facts}}"""

# Main response template
main_response_template = """
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #2d3748; max-width: 100%; margin: 0 auto;">
    <div style="margin-bottom: 32px;">
        <h1 style="font-size: 28px; font-weight: 700; color: #1a202c; margin: 0 0 24px 0; line-height: 1.2; border-bottom: 3px solid #3182ce; padding-bottom: 12px; display: inline-block;">
            {{ title }}
        </h1>
        <div style="font-size: 16px; line-height: 1.8; color: #4a5568;">
            {{ content|safe }}
        </div>
    </div>
</div>
<style>
    p { margin: 16px 0; }
    ul, ol { margin: 16px 0; padding-left: 24px; }
    li { margin: 8px 0; }
    sup {
        background: #3182ce;
        color: white;
        padding: 2px 6px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 4px;
        text-decoration: none;
    }
    sup:hover { background: #2c5aa0; }
    strong { color: #2d3748; font-weight: 600; }
    em { color: #4a5568; font-style: italic; }
    h3 { color: #2d3748; font-size: 20px; font-weight: 600; margin: 20px 0 12px 0; }
</style>"""

# Sources template
sources_template = """
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #2d3748; max-width: 100%; margin: 0 auto;">
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 22px; font-weight: 600; color: #1a202c; margin: 0 0 20px 0; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;">
            Document Sources
        </h2>
        {% for ref in references %}
        <div style="margin-bottom: 24px; padding: 20px; background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: flex-start;">
                <div style="flex-shrink: 0; margin-right: 16px;">
                    {% if ref.thumbnail %}
                    <img src="data:image/png;base64,{{ ref.thumbnail }}" alt="Document thumbnail" style="width: 80px; height: 120px; border-radius: 8px; border: 1px solid #e2e8f0; object-fit: cover; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    {% else %}
                    <div style="width: 80px; height: 120px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 18px; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);">
                        {{ ref.number }}
                    </div>
                    {% endif %}
                </div>
                <div style="flex: 1;">
                    <div style="margin-bottom: 12px;">
                        <a href="{{ ref.url }}" target="_blank" style="color: #3182ce; text-decoration: none; font-size: 16px; font-weight: 600; display: inline-flex; align-items: center;">
                            {{ ref.src }}
                            <svg style="width: 16px; height: 16px; margin-left: 6px;" fill="currentColor" viewBox="0 0 20 20">
                                <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z"/>
                                <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z"/>
                            </svg>
                        </a>
                    </div>
                    <div style="color: #718096; font-size: 14px; margin-bottom: 8px;">
                        Page {{ ref.page }}
                    </div>
                    {% if ref.preview %}
                    <div style="color: #4a5568; font-size: 14px; line-height: 1.6; background: #ffffff; padding: 12px; border-radius: 6px; border-left: 4px solid #3182ce; font-style: italic;">
                        "{{ ref.preview }}"
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</div>
<style>
    a:hover { color: #2c5aa0 !important; }
</style>"""

if __name__ == '__main__':
    skill_input = bakery_market_explorer.create_input(
        arguments={
            "user_question": "What is the market size forecast for bakery products?",
            "base_url": "https://maxdemo.staging.answerrocket.com/apps/system/knowledge-base",
            "max_sources": 3,
            "match_threshold": 0.2
        }
    )
    out = bakery_market_explorer(skill_input)
    print(f"Narrative: {out.narrative}")
    print(f"Visualizations: {len(out.visualizations)}")
