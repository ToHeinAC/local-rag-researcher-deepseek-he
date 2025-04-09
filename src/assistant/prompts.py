RESEARCH_QUERY_WRITER_PROMPT = """Generate necessary queries to complete the user's research goal. Keep queries concise and relevant.

Your output must only be a JSON object containing a single key "queries":
{{ "queries": ["Query 1", "Query 2",...] }}

* Always include the original user query in the queries.
* Generate up to {max_queries} additional queries as needed.
* Today is: {date}
* Use the following language: {language}
"""

SUMMARIZER_SYSTEM_PROMPT = """
You are an expert AI summarizer. Create a factual summary from provided documents using the language {language} with EXACT source citations. Follow these rules:

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
Use the following language: {language}

Your output must only be a valid JSON object with a single key "is_relevant":
{{"is_relevant": true/false}}

# USER QUERY:
{query}

# RETRIEVED DOCUMENTS:
{documents}
"""


SUMMARIZER_PROMPT="""Forward the information from the provided documents that is relevant to the query without adding external information or personal opinions.#
Use the following language: {language}

Query: {query}

Documents:
{documents}

Extract and compile all relevant information from the documents that directly answers the query, preserving the original wording of important passages.
If the retrieved documents are not relevant to the query, state this clearly. Never add external information or personal opinions. 
Do not give any prefix or suffix to the summary, just your summary without any thinking passages.

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
Use the following language: {language}

Summary: {summary}

Source Documents:
{documents}

When evaluating the summary, check for:
1. Accuracy and completeness of information
2. Mandatory inclusion of exact levels, figures, numbers, statistics, and quantitative data ONLY from the original Source Documents
3. Proper references to specific sections or paragraphs when applicable
4. Precise quotations for key definitions and important statements. For citations, ALWAYS use the EXACT format [Source_filename] after each fact.
5. Exact values, ranges, percentages, or measurements for numerical data
6. Score between 0 and 10, where 10 is perfect and 0 is the worst possible score; the score is based on the above criteria
7. In case the Score is above 7, the summary is considered accurate and complete and no further improvements are needed.
8. In case the Score is below 7, the summary is considered insufficient and needs improvement.

Respond with a valid JSON object with the following structure:
{{
  "quality_score": 9,
  "is_accurate": true,
  "is_complete": true,
  "issues_found": [],
  "missing_elements": [],
  "citation_issues": [],
  "improvement_needed": false,
  "improvement_suggestions": ""
}}
"""


REPORT_WRITER_PROMPT = """Create a comprehensive deep report in the language {language} using the information provided from the research process, preserving the original wording when possible.

User instruction: {instruction}

Use the following report structure:
{report_structure}

Information (i.e. the content from the full retrieval process) from research:
{information}

**Key requirements**:
1. You MUST NOT add llm knowledge in the report. Use ONLY the "Information", i.e. the content from the full retrieval process and the 
2. Do not give any prefix or suffix to the report, just your deep report without any thinking passages.
3. Structure the report according to the provided template
4. Focus on answering the user's query clearly and concisely
5. For citations, ALWAYS use the EXACT format [Source_filename] after each fact.
6. Preserve original wording and literal information from the research whenever possible
7. If the information is insufficient to answer parts of the query, state this explicitly
8. Include exact levels, figures, numbers, statistics, and quantitative data ONLY from the source material
9. When referencing specific information, include section or paragraph mentions (e.g., "As stated in Section 3.2...")
10. Maintain precision by using direct quotes for key definitions and important statements
11. For numerical data, always include the exact values, ranges, percentages, or measurements from the sources
12. Clearly attribute information to specific sources when multiple sources are used
"""


LANGUAGE_DETECTOR_PROMPT = """Detect the language of the user's query. Respond with a valid JSON object with a single key 'language' containing the language code (e.g., 'en' for English, 'de' for German, 'fr' for French, etc.).

Your output must only be a valid JSON object with a single key 'language':
{{'language': 'language_code'}}

# USER QUERY:
{query}

Detect the language and return the appropriate language code. If you can't determine the language with confidence, default to 'en' for English.
"""