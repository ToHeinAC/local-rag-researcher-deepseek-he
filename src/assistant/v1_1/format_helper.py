def format_content_with_sources(content, source_filenames, source_paths):
    """
    Format content with source information in the format [Content][Source_filename][Source_path]
    
    Args:
        content (str): The main content text
        source_filenames (list or str): List of source filenames or a comma-separated string
        source_paths (list or str): List of source paths or a comma-separated string
        
    Returns:
        str: Formatted content in the format [Content][Source_filename][Source_path]
    """
    # Process source filenames
    if isinstance(source_filenames, list):
        source_filenames_str = ', '.join(source_filenames)
    else:
        source_filenames_str = source_filenames
        
    # Process source paths
    if isinstance(source_paths, list):
        source_paths_str = ', '.join(source_paths)
    else:
        source_paths_str = source_paths
    
    # Create the formatted string
    formatted_content = f"[{content}][{source_filenames_str}][{source_paths_str}]"
    
    return formatted_content


def parse_document_to_formatted_content(document_text):
    """
    Parse a document text that contains Content, Source_filename, and Source_path
    and format it as [Content][Source_filename][Source_path]
    
    Args:
        document_text (str): The document text containing Content, Source_filename, and Source_path sections
        
    Returns:
        str: Formatted content in the format [Content][Source_filename][Source_path]
    """
    content = ""
    source_filenames = ""
    source_paths = ""
    
    # Extract content
    content_start = document_text.find("Content:")
    if content_start != -1:
        content_start += len("Content:")
        source_filename_start = document_text.find("Source_filename:", content_start)
        if source_filename_start != -1:
            content = document_text[content_start:source_filename_start].strip()
        else:
            content = document_text[content_start:].strip()
    
    # Extract source filenames
    if "Source_filename:" in document_text:
        source_filename_start = document_text.find("Source_filename:")
        source_filename_start += len("Source_filename:")
        source_path_start = document_text.find("Source_path:", source_filename_start)
        if source_path_start != -1:
            source_filenames = document_text[source_filename_start:source_path_start].strip()
        else:
            source_filenames = document_text[source_filename_start:].strip()
    
    # Extract source paths
    if "Source_path:" in document_text:
        source_path_start = document_text.find("Source_path:")
        source_path_start += len("Source_path:")
        source_paths = document_text[source_path_start:].strip()
    
    # Format the content
    return format_content_with_sources(content, source_filenames, source_paths)


# Example usage:
def example_usage():
    # Example 1: Using the format_content_with_sources function
    content = "Rückstände aus industriellen oder bergbaulichen Prozessen können nach einer bestimmten Zeit entfernt werden."
    source_filenames = ["StrlSchG.pdf", "StrlSchG.pdf"]
    source_paths = ["/path/to/StrlSchG.pdf", "/another/path/to/StrlSchG.pdf"]
    
    formatted = format_content_with_sources(content, source_filenames, source_paths)
    print(formatted)
    
    # Example 2: Using the parse_document_to_formatted_content function
    document_text = """Content: Rückstände aus industriellen oder bergbaulichen Prozessen können nach einer bestimmten Zeit entfernt werden.
    
    Laut § 61 Absatz 2 des Strahlenschutzgesetzes (StrlSchG) sind Rückstände aus industriellen oder bergbaulichen Prozessen beruflich exponierte Personen gefährdet.
    
    Source_filename: StrlSchG.pdf, StrlSchG.pdf
    Source_path: /path/to/StrlSchG.pdf, /another/path/to/StrlSchG.pdf"""
    
    formatted = parse_document_to_formatted_content(document_text)
    print(formatted)


if __name__ == "__main__":
    example_usage()
