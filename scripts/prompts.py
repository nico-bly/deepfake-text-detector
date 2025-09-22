from typing import List, Dict, Tuple

def create_system_prompt() -> str:
    """Create the system prompt for document authenticity detection"""
    return """You are an expert document authenticity detector specializing in space-related research documents. Your task is to identify whether a document is REAL (authentic, close to original) or FAKE (corrupted by LLM hallucinations or modifications).

    Key indicators of FAKE documents:
    - Repetitions, paraphrasing too much
    - Factual inconsistencies
    - Technical inaccuracies in space science or engineering
    - Incorrect use of Acronyms
    - Contradictory statements within the text
    - Unusual formatting or language patterns 
    - Names of people, places, or missions that don't exist
    - Scientific data that seems implausible

    Key indicators of REAL documents:
    - Consistent factual information
    - Proper technical terminology
    - Coherent narrative flow
    - Authentic-sounding research descriptions

    Respond with only "0" for REAL documents or "1" for FAKE documents. No explanation needed."""

def create_system_prompt_two_text_analysisis() -> str:
    """Create the system prompt to compare two texts """
    return """You are an expert document authenticity detector specializing in space-related research documents. 
    
    You are given two texts: text1 and text2 that have been modified by an LLM, but one of them contains hallucinations and fake informations.

        Key indicators of FAKE documents:
    - Repetitions, paraphrasing too much
    - Factual inconsistencies
    - Technical inaccuracies in space science or engineering
    - Incorrect use of Acronyms
    - Contradictory statements within the text
    - Unusual formatting or language patterns 
    - Names of people, places, or missions that don't exist
    - Scientific data that seems implausible

    Key indicators of REAL documents:
    - Consistent factual information
    - Proper technical terminology
    - Coherent narrative flow
    - Authentic-sounding research descriptions

    After analysis, clearly state: "Final: REAL = text1" or "Final: REAL = text2".
    """

def create_system_prompt_final_evaluator() -> str:
    """Create the system prompt for document authenticity detection"""
    return """You are an expert document authenticity detector specializing in space-related research documents. 
    
    You are given an analysis of two texts, text1 and text2, one one of them has been considered by FAKE and the other as REAL.
    Respond with only "1" if text1 is REAL or "2" if text2 is REAL.
    No explanation needed.
    """

def create_few_shot_examples(examples: List[Tuple[str, int]]) -> List[Dict[str, str]]:
    """
    Create few-shot examples for the conversation
    
    Args:
        examples: List of (text, label) pairs where label is 0 (real) or 1 (fake)
    
    Returns:
        List of message dictionaries for few-shot learning
    """
    messages = []
    
    for text, label in examples:
        # Add user message with the text
        messages.append({
            "role": "user", 
            "content": f"{text}"
        })
        
        # Add assistant response with the label
        messages.append({
            "role": "assistant", 
            "content": str(label)
        })
    
    return messages