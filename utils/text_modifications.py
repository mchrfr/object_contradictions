"""imports"""

import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))  # Navigate to the project root

# Add the utils folder to the Python path
utils_path = os.path.join(project_dir, 'utils')
sys.path.append(utils_path)

import random
from typing import Optional


import pandas as pd

from thefuzz import fuzz

import spacy
import stanza

from openai import OpenAI
from openAI_utils import calculate_api_call_len_and_price
from openAI_key import key



#text_modifications

def paraphrase_text(premiseStr: str, 
                    hypothese: str, 
                    section: str, entity: str, 
                    random_entity: str, 
                    objecttype: str, 
                    contradiction_label: int) -> tuple:
    """
    Generates a paraphrased version of the input text using OpenAI's API. The response 
    depends on whether a contradiction is requested.

    Args:
        premiseStr (str): The premise string to paraphrase.
        hypothese (str): The hypothesis string.
        section (str): The section of the document being worked on.
        entity (str): The main entity in the premise.
        random_entity (str): The alternative entity for contradiction if applicable.
        objecttype (str): The type of the object.
        contradiction_label (int): A flag indicating whether a contradiction is requested (1 for yes, 0 for no).
        
    Returns:
        tuple: Contains the paraphrased text and associated metadata.

    Raises:
        ValueError: If any of the input strings are not of type `str`, or if `contradiction_label` is not 0 or 1.
        ConnectionError: If the OpenAI API client cannot be initialized.
        RuntimeError: If the OpenAI API request fails.
    """

    model = 'gpt-4o-mini'
    max_tokens=512
    total_cost = 0

    # Validate inputs
    if not all(isinstance(arg, str) for arg in [premiseStr, 
                                                hypothese, 
                                                section, 
                                                entity, 
                                                random_entity, 
                                                objecttype]):
        raise ValueError("Expected all input strings to be of type 'str'.")
    
    if contradiction_label not in {0, 1}:
        raise ValueError(f"Expected 'contradiction_label' to be 0 or 1, but got {contradiction_label}.")

    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=key())
    except Exception as e:
        raise ConnectionError(f"Failed to initialize OpenAI client: {e}")

    # Define message prompt for OpenAI
    messages = [
    {"role": "system", "content": f""},
    {"role": "user", "content": f"Please rephrase the follwing text, ensuring that the word '{random_entity}' remains present:"},
    {"role": "assistant", "content": f"{hypothese}"}]

    try:
        response = client.chat.completions.create(model=model, 
                                                  messages=messages)
    except Exception as e:
        raise RuntimeError(f"OpenAI API request failed: {e}")

    # Paraphrase response and determine exchanges
    paraphrased_statement = response.choices[0].message.content if response.choices else "Error: No response received"
    exchanges = f'{entity} > {random_entity}' if sample_shall_be_contradictory(contradiction_label) else "None"

    # Calculate API cost
    prompt = " ".join([m["content"] for m in messages])
    max_length, cost = calculate_api_call_len_and_price(prompt, 
                                                        model, 
                                                        max_tokens, 
                                                        paraphrased_statement)
    
    total_cost += cost

    return contradiction_label, premiseStr, paraphrased_statement, exchanges, objecttype, section, max_length, total_cost




def paraphrase_premise(premiseStr: str,
                       section: str,
                       entity :str, 
                       objecttype: str, 
                       contradiction_label: int) -> tuple:
    
    try:
        return paraphrase_text(premiseStr, 
                          premiseStr, 
                          section, 
                          entity, 
                          entity, 
                          objecttype, 
                          contradiction_label)
        
    except Exception as e:
        print(f"Something went wrong in function 'paraphrase_premise' with premise: {premiseStr}.")
        print(f"{e}.")



def search_entitiesHub_for_mentioned_entity(entity: str, 
                                            column: str, 
                                            entitiesHub: pd.core.frame.DataFrame, 
                                            threshold: int = 60) -> Optional[int]:
    """
    Searches for an entity in a specified column of the DataFrame and returns the index of the most similar entity
    based on fuzzy matching. The similarity must meet or exceed the given threshold.

    Args:
        entity (str): The entity to search for.
        column (str): The column of entities in which to search.
        entitiesHub (pd.DataFrame): The DataFrame containing the entities.
        threshold (int, optional): The minimum similarity threshold (default is 60).

    Returns:
        Optional[int]: The index of the most similar entity or None if no match meets the threshold.

    Raises:
        ValueError: If any of the input types are incorrect.
        KeyError: If the specified column is not present in the DataFrame.
        TypeError: If the DataFrame is not of the expected type.
    """
    # Validate input types
    if not isinstance(entity, 
                      str):
        raise ValueError(f"Expected 'entity' to be a string, but got {type(entity)}.")
    
    if not isinstance(column, 
                      str):
        raise ValueError(f"Expected 'column' to be a string, but got {type(column)}.")
    
    if not isinstance(entitiesHub, 
                      pd.DataFrame):
        raise TypeError(f"Expected 'entitiesHub' to be a pandas DataFrame, but got {type(entitiesHub)}.")
    
    # Ensure the column exists in the DataFrame
    if column not in entitiesHub.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame.")
    
    # Search for the entity with the best fuzzy similarity
    best_match = max(
        ((index, 
          value) for index, value in entitiesHub[column].items() 
         if isinstance(value, 
                       str) and fuzz.ratio(entity, 
                                                  value) >= threshold),
        key=lambda x: fuzz.ratio(entity, 
                                 x[1]),
        default=(None, None)
    )

    return best_match[0]

    

def get_entities_from_same_region(index: int, 
                                   column: str, 
                                   entitiesHub: pd.core.frame.DataFrame) -> list:
    """
    Retrieves all entities from the same region in the same category, excluding the original entity.
    
    Args:
        index (int): The index of the current entity.
        column (str): The column containing the entities to be returned.
        entitiesHub (pd.DataFrame): The DataFrame containing the entities.

    Returns:
        list: A list of entities from the same region, excluding the entity at the specified index.

    Raises:
        ValueError: If the input types are incorrect.
    """
    
    # Validate input
    if not isinstance(index, 
                      int) or not isinstance(column, 
                                             str) or not isinstance(entitiesHub, 
                                                                    pd.DataFrame):
        raise ValueError("Invalid input types. Expected 'index' as int, 'column' as str, and 'entitiesHub' as pd.DataFrame.")
    
    
    # Get the region of the current entity
    region = entitiesHub.at[index, 'REGION']

    # Filter the entities from the same region and exclude the current entity
    return [
        entity for entity in entitiesHub[entitiesHub['REGION'] == region][column].tolist()
        if entity != entitiesHub.at[index, column]
    ]



def entity_is_person(entity: str, 
                     label: str, 
                     entitiesHub: pd.core.frame.DataFrame) -> bool:
    """
    Checks if the entity is a person based on the label and if it contains a space.
    
    Args:
        entity (str): The entity to check.
        label (str): The label associated with the entity.
        entitiesHub (pd.DataFrame): The DataFrame containing the entity hub data (not used in this function).

    Returns:
        bool: True if the label is 'PERSON' and the entity contains a space, else False.

    Raises:
        ValueError: If 'entity' or 'label' are not of type 'str'.
    """
    if not isinstance(entity, str) or not isinstance(label, str):
        raise ValueError(f"Expected 'entity' and 'label' to be strings, got {type(entity)} and {type(label)}.")
    return label == 'PERSON' and " " in entity



def entity_is_nationality(entity: str, 
                          label: str, 
                          entitiesHub: pd.core.frame.DataFrame) -> int:
    """
    Checks if the entity is a nationality based on the label and searches for it in the 'NATIONALITIES' column of entitiesHub.
    
    Args:
        entity (str): The entity to check.
        label (str): The label associated with the entity.
        entitiesHub (pd.DataFrame): The DataFrame containing the entity hub data.

    Returns:
        int: The index of the entity in the 'NATIONALITIES' column if found and the label is 'NORP', else False.

    Raises:
        ValueError: If 'entity' or 'label' are not of type 'str', or if 'entitiesHub' is not a pd.DataFrame.
    """
    # Validate input types
    if not isinstance(entity, 
                      str) or not isinstance(label, str) or not isinstance(entitiesHub, 
                                                                                   pd.DataFrame):
        raise ValueError(f"Expected 'entity' and 'label' to be strings, and 'entitiesHub' to be a pd.DataFrame, "
                         f"but got {type(entity)}, {type(label)}, and {type(entitiesHub)}.")

    return search_entitiesHub_for_mentioned_entity(entity, 
                                                   'NATIONALITIES', 
                                                   entitiesHub) if label == 'NORP' else False


def entity_is_continent(entity: str, 
                        label: str, 
                        entitiesHub: pd.core.frame.DataFrame) -> int:
    """
    Checks if the entity is a continent based on the label and searches for it in the 'CONTINENTS' column of entitiesHub.
    
    Args:
        entity (str): The entity to check.
        label (str): The label associated with the entity.
        entitiesHub (pd.DataFrame): The DataFrame containing the entity hub data.

    Returns:
        int: The index of the entity in the 'CONTINENTS' column if found and the label is 'LOC', else False.

    Raises:
        ValueError: If 'entity' or 'label' are not of type 'str', or if 'entitiesHub' is not a pd.DataFrame.
    """
    # Validate input types
    if not isinstance(entity, 
                      str) or not isinstance(label,
                                             str) or not isinstance(entitiesHub, 
                                                                    pd.DataFrame):
        raise ValueError(f"Expected 'entity' and 'label' to be strings, and 'entitiesHub' to be a pd.DataFrame, "
                         f"but got {type(entity)}, {type(label)}, and {type(entitiesHub)}.")

    return search_entitiesHub_for_mentioned_entity(entity, 
                                                   'CONTINENTS', 
                                                   entitiesHub) if label == 'LOC' else False


def entity_is_country(entity: str, 
                      label: str, 
                      entitiesHub: pd.core.frame.DataFrame) -> int:
    """
    Checks if the entity is a country based on the label and searches for it in the 'COUNTRIES' column of entitiesHub.
    
    Args:
        entity (str): The entity to check.
        label (str): The label associated with the entity.
        entitiesHub (pd.DataFrame): The DataFrame containing the entity hub data.

    Returns:
        int: The index of the entity in the 'COUNTRIES' column if found and the label is 'GPE', else False.

    Raises:
        ValueError: If 'entity' or 'label' are not of type 'str', or if 'entitiesHub' is not a pd.DataFrame.
    """
    # Validate input types
    if not isinstance(entity, 
                      str) or not isinstance(label, 
                                             str) or not isinstance(entitiesHub, 
                                                                    pd.DataFrame):
        raise ValueError(f"Expected 'entity' and 'label' to be strings, and 'entitiesHub' to be a pd.DataFrame, "
                         f"but got {type(entity)}, {type(label)}, and {type(entitiesHub)}.")
    
    return search_entitiesHub_for_mentioned_entity(entity, 
                                                   'COUNTRIES', 
                                                   entitiesHub) if label == 'GPE' else False


def sample_shall_be_contradictory(contradiction_label: int) -> bool:
    """
    Determines if the sample should be contradictory based on the contradiction label.
    
    Args:
        contradiction_label (int): The contradiction label (0 or 1).
    
    Returns:
        bool: True if the contradiction label is 1 (indicating a contradictory sample), else False.

    Raises:
        ValueError: If 'contradiction_label' is not 0 or 1.
    """
    if contradiction_label not in [0, 1]:
        raise ValueError(f"Expected 'contradiction_label' to be either 0 or 1, got {contradiction_label}.")
    
    return contradiction_label == 1



def replace_entity_in_objectphrase(premiseStr: str, 
                                    section: str, 
                                    entity: str, 
                                    index,  # int or str
                                    column: str, 
                                    objecttype: str, 
                                    entitiesHub: pd.core.frame.DataFrame, 
                                    contradiction_label: int) -> tuple:
    """
    Replaces an entity in the premise string with a random entity from the same category/region and generates a paraphrase.

    Args:
        premiseStr (str): The premise string to modify.
        section (str): The section to which the premise belongs.
        entity (str): The entity to replace.
        index (int or str): The index of the entity (depends on the column).
        column (str): The column in the entity hub to search for a replacement.
        objecttype (str): The object type associated with the entity.
        entitiesHub (pd.DataFrame): DataFrame containing the entities.
        contradiction_label (int): The flag indicating whether a contradiction should be created.

    Returns:
        tuple: A tuple with paraphrased text and metadata.
    
    Raises:
        ValueError: If no valid replacement entity is found.
    """
    
    random.seed(len(premiseStr))

    try:
        # Determine the replacement entity based on the column
        if column == "PERSON":
            random_entity = random.choice(entitiesHub[column].dropna().tolist())
        elif column == "CONTINENTS":
            continents = set(entitiesHub['CONTINENTS'].dropna().tolist()) - {entity.replace("'s", 
                                                                                            "")}
            if not continents:
                raise ValueError(f"No valid continents available for replacement for entity: {entity}")
            random_entity = random.choice(list(continents))
        else:
            random_entity = random.choice(get_entities_from_same_region(index, 
                                                                        column, 
                                                                        entitiesHub))

        # Replace the entity in the premise string
        hypothese = premiseStr.replace(entity, 
                                       random_entity + "'s" if entity.endswith("'s") else random_entity)

        # Paraphrase the modified text
        return paraphrase_text(premiseStr, 
                               hypothese, 
                               section, 
                               entity, 
                               random_entity, 
                               objecttype, 
                               contradiction_label)

    except ValueError as e:
        print(f"ValueError in 'replace_entity_in_objectphrase': {e}")
        raise  # Reraise the exception after printing
    except Exception as e:
        print(f"Unexpected error in 'replace_entity_in_objectphrase': {e}")
        raise ValueError("An error occurred while processing the entity replacement.") #ToDO: konkretisieren



def assign_entity_to_predefined_category(premiseDoc: spacy.tokens.doc.Doc,
                                         section: str,
                                         entitiesInObject: dict,
                                         objecttype: str,
                                         entitiesHub: pd.core.frame.DataFrame,
                                         contradiction_label: int) -> tuple:
    """
    Assigns a given entity to a predefined category (e.g., COUNTRY, PERSON, etc.) based on its type and the provided
    entitiesHub, and processes the entity accordingly.

    This function checks if an entity belongs to one of the predefined categories such as countries, nationalities,
    persons, or continents. Depending on the entity's type and the contradiction label, the entity is either replaced
    in the object phrase or paraphrased.

    Parameters:
    premiseDoc (spacy.tokens.doc.Doc): The document containing the premise.
    section (str): The section of the premise to which the entity belongs.
    entitiesInObject (dict): A dictionary where the keys are entities and the values are their labels.
    objecttype (str): The type of the object the entity belongs to.
    entitiesHub (pd.core.frame.DataFrame): A DataFrame containing the predefined entities (e.g., countries, nationalities).
    contradiction_label (int): A label indicating whether the sample should be contradictory (1) or not (0).

    Returns:
    tuple: A tuple containing the processed result, which could either be a modified object phrase or a paraphrased premise.

    Raises:
    ValueError: If 'entitiesInObject' is not a non-empty dictionary.
    """
    
    # Validate input: Ensure that 'entitiesInObject' is a non-empty dictionary.
    if not isinstance(entitiesInObject, dict) or not entitiesInObject:
        raise ValueError #ValueError("The 'entitiesInObject' must be a non-empty dictionary.") 

    # Define categories and corresponding check functions for identifying entities.
    categories = [
        ('COUNTRIES', entity_is_country),     # Check function for countries
        ('NATIONALITIES', entity_is_nationality),  # Check function for nationalities
        ('PERSON', entity_is_person),         # Check function for persons
        ('CONTINENTS', entity_is_continent)   # Check function for continents
    ]

    # Iterate through each entity and label in the provided entities dictionary.
    for entity, label in entitiesInObject.items():
        try:
            # Find the matching category for the current entity using the corresponding check function.
            category = next(((category_name, check_func(entity, label, entitiesHub))
                            for category_name, check_func in categories if check_func(entity, label, entitiesHub)), None)

            # If no category is found, print a message and return None.
            if not category:
                #print(f"No predefined category found for entity: {entity}")
                return None
            
            # Unpack the result into the column and index variables.
            column, index = category

            # Return the result based on whether the sample should be contradictory or paraphrased.
            if sample_shall_be_contradictory(contradiction_label):
                # If contradiction label is 1, replace the entity in the object phrase.
                result = replace_entity_in_objectphrase(
                    str(premiseDoc),
                    section,
                    entity,
                    index,
                    column,
                    objecttype,
                    entitiesHub,
                    contradiction_label
                )
            else:
                # Otherwise, paraphrase the premise.
                result = paraphrase_premise(
                    str(premiseDoc),
                    section,
                    entity,
                    objecttype,
                    contradiction_label
                )

            # Return the final result.
            return result

        except Exception as e:
            # Catch and print any exceptions raised during processing.
            print(f"Error in assign_entity_to_predefined_category for entity '{entity}': {e}")
            raise


def get_objectphrase(premiseStr: str, 
                     stanza_nlp: stanza.pipeline.core.Pipeline) -> tuple:
    """
    Extracts the object phrase from a given string using a dependency parsing model.

    This function identifies object phrases in a sentence and returns the syntactic relationship
    (e.g., 'obj', 'iobj') and the sorted list of words within the object phrase subtree.

    Parameters:
    premiseStr (str): The input sentence as a string.
    stanza_nlp (callable): The NLP pipeline (Stanza model) to parse the sentence.

    Returns:
    tuple: A tuple containing the syntactic relation (str) and a sorted list of words within the object phrase 
           (list of spacy tokens). If no object phrase is found, returns (None, None).

    Raises:
    ValueError: If the input string is empty.
    ValueError: If the input document contains no sentences.
    RuntimeError: If there is an error during the NLP analysis.
    """
    
    def get_object_subtree(word: stanza.models.common.doc.Word, 
                           sentence: stanza.models.common.doc.Sentence, 
                           visited: set) -> list:
        """
        Helper function to recursively gather the subtree of an object, avoiding cycles.

        Parameters:
        word (stanza.models.common.doc.Word): The current word to analyze.
        sentence (stanza.models.common.doc.Sentence): The sentence containing the word.
        visited (set): A set of visited word IDs to avoid infinite recursion.

        Returns:
        list: A list of words in the subtree of the object.
        """
        if word.id in visited:
            return []  # Prevent infinite recursion
        visited.add(word.id)
        return [word] + [w for w in sentence.words if w.head == word.id for w in get_object_subtree(w, sentence, visited)]

    if not premiseStr:
        raise ValueError("Input string cannot be empty.")
    
    try:
        doc_stanza = stanza_nlp(premiseStr)
        if not doc_stanza.sentences:
            raise ValueError("The input document contains no sentences.")
    
    except Exception as e:
        raise RuntimeError("Error during NLP analysis.") from e
    
    for sentence in doc_stanza.sentences:
        for word in sentence.words:
            if hasattr(word, 'deprel') and word.deprel in ['iobj', 
                                                           'obj'] and hasattr(word, 'id') and word.id is not None:
                visited = set()
                subtree = get_object_subtree(word, 
                                             sentence, 
                                             visited)
                if subtree:
                    return word.deprel, sorted(subtree, 
                                               key=lambda w: w.id)
    
    return None, None



def extract_entities_from_objectphrase(premiseDoc: spacy.tokens.doc.Doc,
                                       section: str, 
                                       entitiesHub: pd.core.frame.DataFrame,
                                       stanza_model: stanza.pipeline.core.Pipeline,
                                       contradiction_label: int) -> tuple:
    """
    Extracts named entities from the object phrase of a given SpaCy document.

    This function parses the input document to extract the object phrase, checks for named entities
    within that phrase, and assigns them to predefined categories. If no object phrase or named entities 
    are found, a message is printed and the function proceeds with no return value.

    Parameters:
    premiseDoc (spacy.tokens.Doc): The input document as a SpaCy Doc object.
    section (str): The section identifier used for categorization and further processing.
    entitiesHub (pd.DataFrame): A DataFrame containing predefined entity categories.
    stanza_model (stanza.pipeline.core.Pipeline): The NLP model (Stanza) used to parse the text.
    contradiction_label (int): A label indicating contradiction status for the entities.

    Returns:
    tuple: A tuple containing the processed document, section, entities in objectphrase, object type, 
           entities hub, and contradiction label. (If no entities are found, returns empty dictionaries.)

    Raises:
    ValueError: If the input document is invalid or the NLP model is missing.
    """
    
    bin = None
    
    if not isinstance(premiseDoc, 
                      spacy.tokens.doc.Doc) or not stanza_model:
        raise ValueError("Invalid document or missing NLP model.")
    
    try:
        objecttype, objectphrase = get_objectphrase(premiseDoc.text, 
                                                    stanza_model)

        if not objectphrase:
            #print(f"No object phrase found.")
            #raise
            bin == True
        
        objectSpanDoc = premiseDoc.char_span(objectphrase[0].start_char, 
                                             objectphrase[-1].end_char)
        
        if not objectSpanDoc or not (entitiesInObject := {ent.text: ent.label_ for ent in objectSpanDoc.ents}):
            #print(f"No named entities found in object phrase.")
            raise
        
        bin == False
        return assign_entity_to_predefined_category(premiseDoc, 
                                                    section,
                                                    entitiesInObject, 
                                                    objecttype, 
                                                    entitiesHub, 
                                                    contradiction_label)

    except Exception as e:
        #print(f"The following premise could not be processed due to missing object or entity: {premiseDoc.text}")
        bin == True
        
