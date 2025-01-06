import os
import re
import spacy
import pandas as pd

# Define the project directory dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))  # Navigate to the project root

def import_dataSource(file: str) -> list:
    """
    Imports the data from the specified text file, strips leading/trailing spaces from each line,
    and returns a list of cleaned lines.

    Args:
    file (str): The path to the file relative to the project directory.

    Returns:
    list: A list of cleaned lines from the file.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    PermissionError: If the file cannot be read due to insufficient permissions.
    ValueError: If the file is empty or cannot be processed correctly.
    Exception: For any other unexpected errors during file processing.
    """
    file_path = os.path.join(project_dir, file)  # Build the absolute path
    try:
        with open(file_path, 'r') as source:
            lines = [line.strip() for line in source if line.strip()]  # Strip whitespace and ignore empty lines

        if not lines:
            raise ValueError(f"The file '{file}' is empty or contains only blank lines.")

        return lines
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    
    except PermissionError:
        raise PermissionError(f"Permission denied when attempting to read the file '{file_path}'.")
    
    except Exception as e:
        raise Exception(f"An error occurred while reading the file '{file_path}': {e}")


def modify_sectionName(old_sectionName: str) -> str:
    """
    Modifies the section name based on predefined mappings.

    This function checks if the given `old_sectionName` matches any of the entries in 
    predefined lists of section names (such as UK politics, UK regional, and BBC sections). 
    It returns a modified version of the section name based on specific conditions. 
    If no match is found, the original section name is returned.

    Parameters:
    old_sectionName (str): The section name to be modified.

    Returns:
    str: The modified section name, or the original section name if no modification is made.

    Raises:
    FileNotFoundError: If any of the source files cannot be found.
    """
    uk_politics = import_dataSource('file_sources/uk_politics.txt')
    uk_regional = import_dataSource('file_sources/uk_regional.txt')
    bbc_sections = import_dataSource('file_sources/bbc_sections.txt')  

    if old_sectionName in uk_politics:
        return "UK Politics" 
    
    elif old_sectionName in uk_regional:
        return "UK Regional"  
    
    elif old_sectionName in bbc_sections:
        match old_sectionName:
            case "US and Canada": 
                return 'US & Canada'
            case "War in Ukraine":
                return "Europe"
            case "China":
                return "Asia"
            case "India":
                return "Asia"
            case "Features":
                return "Science & Environment"
            case "Scotland business":
                return "Business"
            case _: 
                return old_sectionName
    else: 
        return "Other"



    

def group_dataset_by_section(dataset: pd.core.frame.DataFrame) -> list:
    """
    Groups the dataset by the 'section' column, removes duplicates from each section based on 'description',
    and returns a list of subframes.

    Args:
        dataset (pd.DataFrame): The dataset to group by 'section'.

    Returns:
        list: A list of dataframes, each corresponding to a section in the original dataset, with duplicates removed.

    Raises:
        ValueError: If the required columns ('section' and 'description') are not present in the dataset.
        TypeError: If the input dataset is not a pandas DataFrame.
    """
    
    # Validate the input
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError(f"Expected input to be a pandas DataFrame, got {type(dataset)}.")
    
    if 'section' not in dataset.columns or 'description' not in dataset.columns:
        raise ValueError("The dataset must contain 'section' and 'description' columns.")

    # Group by 'section' and remove duplicates based on 'description' within each group
    return [group.drop_duplicates(subset='description', 
                                  keep='first', 
                                  ignore_index=True)
        for _, group in dataset.groupby('section')
        if not group.empty]  # Ensure no empty groups are processed


def prepare_text(premiseStr: str, 
                 spacy_model) -> spacy.tokens.doc.Doc:
    """
    Prepares the input text for processing by replacing specific abbreviations and 
    passing it through the spaCy NLP model.

    Args:
        premiseStr (str): The input premise text to be processed.
        spacy_model (spacy.lang.en.English): The spaCy model used to process the text.

    Returns:
        spacy.tokens.doc.Doc: Processed spaCy Doc object.

    Raises:
        ValueError: If 'premiseStr' is not a string or 'spacy_model' is not a valid spaCy model.
    """
    # Validate input types
    if not isinstance(premiseStr, 
                      str):
        raise ValueError(f"Expected 'premiseStr' to be a string, but got {type(premiseStr)}.")
    
    if not callable(getattr(spacy_model, 
                            'pipe', 
                            None)):
        raise ValueError(f"Expected 'spacy_model' to be a valid spaCy model, but got {type(spacy_model)}.")

    # Replace abbreviations using regex
    premiseStr = re.sub(r'\b(US|UK)\b', 
                        lambda m: f"{m.group(0)}." if m.group(0) == "US" else f"U.K.", 
                        premiseStr)

    # Return the processed text as a spaCy Doc
    return spacy_model(premiseStr)

