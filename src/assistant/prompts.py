RESEARCH_QUERY_WRITER_PROMPT = """Generate necessary queries to complete the user's research goal. Keep queries concise and relevant.

Your output must only be a JSON object containing a single key "queries":
{{ "queries": ["Query 1", "Query 2",...] }}

* Always include the original user query in the queries.
* Generate up to {max_queries} additional queries as needed.
* Today is: {date}
"""

SUMMARIZER_SYSTEM_PROMPT = """
You are an expert AI summarizer. Create a factual summary from provided documents with EXACT source citations. Follow these rules:

1. **Citation Format**: For citations, ALWAYS use the EXACT format [Source_filename] after each fact. 
You find the Source_filename in the provided metadata with the following structure:
\nContent: some content
\nSource_filename: the corresponding Source_filename
\nSource_path: the corresponding fullpath

2. **Content Rules**:
   - Maintain exact figures, data points, sections and paragraphs
   - No markdown, formulate only plain text and complete sentences
   - NO new information or opinions

**Example Input**:
\nContent: 'The 2025 budget for infrastructure is €4.2M.',
\nSource_filename: 'City_Budget.pdf'
\nSource_path: './some/path/to/City_Budget.pdf'
  
**Example Output**:
The 2025 fiscal plan allocates €4.2 million for infrastructure [City_Budget.pdf].

**Current Task**:
Create a deep, comprehensive and accurate representation of the provided original information:
"""

RELEVANCE_EVALUATOR_PROMPT = """Determine if the retrieved documents are relevant to the user's query.

Your output must only be a valid JSON object with a single key "is_relevant":
{{"is_relevant": true/false}}

# USER QUERY:
{query}

# RETRIEVED DOCUMENTS:
{documents}
"""


SUMMARIZER_PROMPT="""Forward the information from the provided documents that is relevant to the query without adding external information or personal opinions.#

Query: {query}

Documents:
{documents}

Extract and compile all relevant information from the documents that directly answers the query, preserving the original wording of important passages.
If the retrieved documents are not relevant to the query, state this clearly.

Important guidelines:
1. For citations, ALWAYS use the EXACT format [Source_filename] after each fact. 
You find the Source_filename in the provided metadata with the following structure:
\nContent: some content
\nSource_filename: the corresponding Source_filename
\nSource_path: the corresponding fullpath
2. Include exact levels, figures, numbers, statistics, and quantitative data ONLY from the source Documents
3. Preserve section or paragraph references from the original Documents when available (e.g., "As stated in Section 3.2...")
4. Use direct quotes for key definitions and important statements
5. Maintain precise numerical values, ranges, percentages, or measurements
6. Clearly attribute information to specific sources when multiple Documents are provided
"""


QUALITY_CHECKER_PROMPT = """Evaluate if the summary contains sufficient and relevant information from the source documents to answer the query.

Summary: {summary}

Source Documents:
{documents}

When evaluating the summary, check for:
1. Accuracy and completeness of information
2. Mandatory inclusion of exact levels, figures, numbers, statistics, and quantitative data ONLY from the original Source Documents
3. Proper references to specific sections or paragraphs when applicable
4. Precise quotations for key definitions and important statements
5. Exact values, ranges, percentages, or measurements for numerical data

Respond with a valid JSON object with the following structure:
{{
  "quality_score": 0.8,
  "is_sufficient": true,
  "improvement_needed": false,
  "improvement_suggestions": ""
}}
"""


REPORT_WRITER_PROMPT = """Create a comprehensive report using the information provided from the research process, preserving the original wording when possible.

User instruction: {instruction}

Use the following report structure:
{report_structure}

Information from research:
{information}

**Key requirements**:
1. Use only the information provided from the information from research- do not add external information
2. Structure the report according to the provided template
3. Focus on answering the user's query clearly and concisely
4. Use proper citations to the source documents in the format [source], e.g. [source_filename.pdf]
5. Preserve original wording and literal information from the research whenever possible
6. If the information is insufficient to answer parts of the query, state this explicitly
7. Include exact levels, figures, numbers, statistics, and quantitative data ONLY from the source material
8. When referencing specific information, include section or paragraph mentions (e.g., "As stated in Section 3.2...")
9. Maintain precision by using direct quotes for key definitions and important statements
10. For numerical data, always include the exact values, ranges, percentages, or measurements from the sources
11. Clearly attribute information to specific sources when multiple sources are used
"""