import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))  # Navigate to the project root

# Add the utils folder to the Python path
utils_path = os.path.join(project_dir, 'utils')
sys.path.append(utils_path)

import spacy
import stanza
import pandas as pd

# Import functions from other modules in the utils folder
from preparations import prepare_text
from text_modifications import extract_entities_from_objectphrase


#processing

def create_samples(premiseDoc: spacy.tokens.doc.Doc,
                   section: str,
                   entitiesHub: pd.core.frame.DataFrame,
                   stanza_model: stanza.pipeline.core.Pipeline,
                   contradiction_label: int) -> pd.core.frame.DataFrame:
    """
    Extracts entities from the object phrase based on the contradiction label.

    Args:
        premiseDoc (spacy.tokens.doc.Doc): The document to process.
        section (str): The section of the document.
        entitiesHub (pd.DataFrame): DataFrame containing the entity hub.
        stanza_model (stanza.pipeline.core.Pipeline): Stanza NLP pipeline.
        contradiction_label (int): The contradiction label (1 for contradiction, 0 for non-contradiction).

    Returns:
        pd.DataFrame: The entities extracted from the object phrase.

    Raises:
        ValueError: If the contradiction_label is not 0 or 1.
    """
    if contradiction_label not in [0, 1]:
        raise ValueError(f"Expected contradiction_label to be 0 or 1, got {contradiction_label}.")

    return extract_entities_from_objectphrase(premiseDoc, 
                                              section, 
                                              entitiesHub, 
                                              stanza_model, 
                                              contradiction_label)



def write_samples_to_dataframe(contradictions: pd.Series, 
                               non_contradictions: pd.Series) -> pd.DataFrame:
    """
    Combines contradiction and non-contradiction samples into a single DataFrame 
    and shuffles them for training.

    Args:
        contradictions (pd.Series): Series containing contradiction samples.
        non_contradictions (pd.Series): Series containing non-contradiction samples.

    Returns:
        pd.DataFrame: A shuffled DataFrame containing the samples for training.

    Raises:
        ValueError: If the input Series are empty or if they contain invalid data.
    """
    # Check if the input Series are not empty
    try:
        if contradictions.empty or non_contradictions.empty:
            raise ValueError("Both 'contradictions' and 'non_contradictions' must not be empty.")

        # Define the columns for the final DataFrame
        columns = ['contradiction_label', 
                'premise', 
                'hypothese', 
                'exchanges', 
                'objecttype', 
                'section', 
                'max_length', 
                'total_cost']
        
        # Concatenate the two Series and drop NaN values
        combined = pd.concat([contradictions, non_contradictions]).dropna()

        # Convert the combined Series into a DataFrame
        samples = pd.DataFrame(combined.tolist(), columns=columns)

        # Shuffle the DataFrame and reset the index
        return samples.sample(frac=1, random_state=42).reset_index(drop=True)
    
    except Exception as e:
        print(f"Cannot process dataframes in section {'Section_name'} for ether contradictions or non_contradictions in  is empty\n{e}") #Add section name        





def process_downloaded_datasets(dataframe_list: list, 
                               entitiesHub: pd.DataFrame, 
                               spacy_model, 
                               stanza_model: stanza.pipeline.core.Pipeline) -> pd.DataFrame:
    """
    Processes a list of dataframes, creates contradictory and non-contradictory samples, 
    and combines them into a single dataframe for further use.
    
    Args:
        dataframe_list (list): List of DataFrames to process.
        entitiesHub (pd.DataFrame): DataFrame containing entity hub data.
        spacy_model (spacy.lang.en.English): Pre-trained Spacy model for text processing.
        stanza_model (stanza.pipeline.core.Pipeline): Stanza model for NLP tasks.
    
    Returns:
        pd.DataFrame: Combined DataFrame with samples (both contradictory and non-contradictory).

    Raises:
        ValueError: If the dataframe_list is empty or contains invalid DataFrames.
        TypeError: If the entitiesHub is not a DataFrame.
    """
    # Validate inputs
    if not isinstance(entitiesHub, pd.DataFrame):
        raise TypeError(f"Expected 'entitiesHub' to be a DataFrame, got {type(entitiesHub)}.")
    if not dataframe_list:
        raise ValueError("The 'dataframe_list' cannot be empty.")
    
    # Prepare a list to hold new datasets for each dataframe in the list
    new_datasets = []

    # Iterate over each dataframe in the list
    for dataframe in dataframe_list:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"Each element in 'dataframe_list' must be a DataFrame, but got {type(dataframe)}.")
        
        # Prepare the premiseDoc and section
        section = dataframe.section.iloc[0]  # Getting the first element of 'section'
        print(f'Start processing section "{section}"')
        premiseDoc = dataframe.description.apply(lambda text: prepare_text(text, spacy_model))
        

        # Create both contradictory and non-contradictory samples in one step for efficiency
        contradictions, non_contradictions = zip(*premiseDoc.apply(lambda text: (
            create_samples(text, section, entitiesHub, stanza_model, contradiction_label=1),  # 1: Contradictory
            create_samples(text, section, entitiesHub, stanza_model, contradiction_label=0)   # 0: Non-Contradictory
        )))
        
        # Convert the resulting tuples into separate DataFrames
        contradictions_df = write_samples_to_dataframe(pd.Series(contradictions).dropna(), 
                                                       pd.Series(non_contradictions).dropna())

        # Append the result to the new_datasets list
        new_datasets.append(contradictions_df)

    # Combine all the datasets into a single DataFrame and reset index
    return pd.concat(new_datasets, ignore_index=True)

