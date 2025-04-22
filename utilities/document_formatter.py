import sys
import os

# Add the parent directory to the path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.assistant.utils import format_documents_as_plain_text
from langchain_core.documents import Document


def format_documents_wrapper(documents_dict):
    """
    Wrapper function to format a dictionary of LangChain Documents into plain text.
    
    Args:
        documents_dict (dict): Dictionary with keys as queries and values as lists of LangChain Documents
        
    Returns:
        dict: Dictionary with the same keys but values as formatted plain text
    """
    formatted_results = {}
    
    for query, docs in documents_dict.items():
        formatted_results[query] = format_documents_as_plain_text(docs)
    
    return formatted_results


def create_example_documents():
    """
    Create example documents similar to the provided structure for testing.
    """
    # Example documents similar to what was provided
    example_docs = [
        Document(
            page_content="emitter type, where the radiation protection and safety of the device in question mean that it may be used without a licence or notification in accordance with the statuto­ ry ordinance in accordance with section 49 subsections 1 and 2, 2. an Xray tube assembly type where the radiation protection properties of the Xray equipment including said Xray tube assembly permit it to be operated without a licence in accordance with the statutory ordinance in accordance with section 49 subsec­ tions 1 and 2, 59 3. an Xray equipment type as a basicprotection device where the high level of protection of the device type, including any openings in the protective housing for the introduction or removal of articles, means that the Xray equipment may be operated without a li­ cence in accordance with the statutory ordinance in accordance with section 49 subsec­ tions 1 and 2, 4. an Xray equipment type as a highprotection device where the high level of protection of the device type means that the Xray equipment may be operated without a licence in accordance with the statutory ordinance in accordance with section 49 subsections 1 and 2, 5. an Xray equipment type as a fullprotection device where the particularly high level of protection of the device type means that the Xray equipment may be operated without a licence and without supervision by a person in possession of the requisite specialist knowledge in radiation protection in accordance with the statutory ordinance in accord­ ance with section 49 subsections 1 and 2, 6. an Xray equipment type as school Xray equipment where the radiation protection of the device type means that the Xray equipment may be operated in connection with teaching in schools in accordance with the statutory ordinance in accordance with sec­ tion 49 subsections 1 and 2. 2 Subsection 1 shall not apply to medical devices or accessories within the meaning of the Medical Devices Act. Subsection 1 no. 1 shall not apply to devices containing high activity",
            metadata={
                'id': 'strlschg_en_bf.pdf:0:83',
                'page': 0,
                'path': './files/strlschg_en_bf.pdf',
                'source': 'strlschg_en_bf.pdf',
                'language': 'German'
            }
        ),
        Document(
            page_content="no. 1. The competent authority shall also be provided with the personal data of the persons affect­ ed and with the calculated body dose, as well as with the reason for the body dose limits being exceeded. The parties subject to the recording obligation shall be informed to inform the affected persons of the body dose without undue delay. Section 168 Transfer of the body dose calculation results 1 The radiation protection executive, the party obliged in accordance with section 131 subsection 1 or section 145 subsection 1, first sentence, and the responsible party in ac­ cordance with section 115 subsection 2 or section 153 subsection 1, insofar as they em­ ploy a measuring body in accordance with section 169 subsection 1 to determine occupa­ tional exposure, must provide this measuring body with the data in accordance with sec­ tion 170 subsection 2 nos. 1 to 7 with regard to the persons for whom the body dose is to be calculated. The competent authority shall be presented with the details in accordance with the first sentence and the calculated body dose on request. 2 If the parties obliged to determine the occupational exposure in accordance with subsection 1 do not employ a measuring body in accordance with section 169 subsec­ tion 1, they shall present the competent authority with the data in accordance with sec­ tion 170 subsection 2, including the calculated body dose. 149 Section 169 Appointment of measuring bodies empowerment to issue ordinances 1 The competent authority shall appoint measuring bodies to determine the occupa­ tional exposure 1. from external exposure whilst working, 2. from internal exposure whilst working, 3. of workers intervening in an emergency exposure situation or other hazardous situation, 4. from radon in the workplace, 5. in relation to measures for radioactivelycontaminated sites, and 6. in other existing exposure situations. 2 A measuring body may only be appointed where 1. it has sufficient personnel to perform its tasks and its",
            metadata={
                'id': 'strlschg_en_bf.pdf:0:222',
                'page': 0,
                'path': './files/strlschg_en_bf.pdf',
                'source': 'strlschg_en_bf.pdf',
                'language': 'German'
            }
        )
    ]
    
    return example_docs


def main():
    # Create example documents
    example_docs = create_example_documents()
    
    # Format the documents using the utility function
    formatted_text = format_documents_as_plain_text(example_docs)
    
    # Print the formatted text
    print("\nFormatted Documents as Plain Text:")
    print("===================================\n")
    print(formatted_text)
    
    # Example with dictionary structure (similar to what might be in the graph)
    example_dict = {
        "Bedeutung der Körperdosis für beruflich tätige Personen: wie wird sie ermittelt?": example_docs
    }
    
    # Format using the wrapper function
    formatted_dict = format_documents_wrapper(example_dict)
    
    # Print the formatted dictionary
    print("\nFormatted Documents from Dictionary:")
    print("===================================\n")
    for query, text in formatted_dict.items():
        print(f"Query: {query}\n")
        print(text)
        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    main()
