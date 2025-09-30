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
def create_system_prompt_two_text_analysis() -> str:
    """Simplified system prompt to compare two texts"""
    return """You are a space document authenticity detector. One text is REAL, one is FAKE (LLM-modified).

    Look for:
    1. **Repetition / verbosity**: repeated phrases, over-explaining, unnatural phrasing
    2. **Factual & technical consistency**: wrong dates, fake missions, impossible physics, mismatched acronyms
    3. **Clarity & coherence**: logical flow, concise vs convoluted explanations

    Decide which text is closer to the original. Conclude with:
    "Final: REAL = text1" or "Final: REAL = text2" """


'''
def create_system_prompt_two_text_analysis() -> str:
    """Create the system prompt to compare two texts"""
    return """You are a space document authenticity detector. One text is REAL (close to original), one is FAKE (LLM-corrupted).

    ANALYSIS FRAMEWORK:
    1. **REPETITION & LINGUISTIC**: Look for repeated phrases, redundant explanations, verbose overexplanations, unnatural phrasing
    2. **FACTUAL CONSISTENCY**: Contradictions, impossible claims, inconsistent data  
    3. **TECHNICAL ACCURACY**: Space terminology, mission details, organizational references
    4. **NARRATIVE COHERENCE**: Logical flow vs. unnecessarily structured/over-explained content

    CRITICAL FAKE INDICATORS:
    - Same information repeated with slight variations (strongest signal)
    - Over-explaining basic concepts or restating points multiple times
    - Verbose, unnecessarily detailed explanations where conciseness is expected
    - Phrases repeated throughout text with minor word changes
    - Breaking simple concepts into elaborate bullet points or structure

    SPACE DOMAIN RED FLAGS:
    - Non-existent missions/spacecraft/ESA programs, wrong organizational structure
    - Impossible physics/specifications, anachronistic dates, implausible research data
    - Mismatched acronyms, contradictory technical details

    Compare: Which text is more concise and direct vs. which over-explains or restates the same concepts?

    Conclude with: "Final: REAL = text1" or "Final: REAL = text2" """


def create_system_prompt_two_text_analysis() -> str:
    """Create the system prompt to compare two texts"""
    return """You are an expert document authenticity detector specializing in space-related research documents from ESA and similar organizations.

    You will analyze two texts (text1 and text2) where one is REAL (close to original) and one is FAKE (corrupted by LLM processing). Both have been modified, but the fake contains more significant distortions.

    ANALYSIS FRAMEWORK:
    1. FACTUAL CONSISTENCY: Check for contradictions, impossible claims, or inconsistent data
    2. TECHNICAL ACCURACY: Verify space science terminology, mission details, organizational references
    3. NARRATIVE COHERENCE: Assess logical flow and coherence of information
    4. LINGUISTIC PATTERNS: Identify unnatural repetitions, awkward phrasings, or LLM artifacts

    SPACE DOMAIN RED FLAGS FOR FAKE TEXTS:
    - Non-existent missions, spacecraft, or ESA programs
    - Incorrect technical specifications or impossible physics
    - Wrong organizational structure or personnel references  
    - Anachronistic information (wrong timeframes)
    - Implausible research outcomes or data
    - Mismatched acronyms or incorrect expansions
    - Contradictory technical details within the same text

    ANALYSIS PROCESS:
    1. First, identify the main topic and key claims in each text
    2. Cross-check factual statements for plausibility and consistency
    3. Evaluate technical terminology usage
    4. Compare narrative coherence and flow
    5. Look for LLM-generated artifacts (repetition, unnatural phrasing)

    Provide your reasoning step by step, then conclude with: "Final: REAL = text1" or "Final: REAL = text2".

    If texts are very similar, focus on subtle differences in technical accuracy and factual consistency.
    """
'''


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

def create_few_shot_examples_list(examples_list):
    """Create few-shot examples for the prompt"""
    messages = []
    
    for example in examples_list:
        # User message with the two texts
       
        messages.append({
            "role": "user", 
            "content": f"text1: {example['text1']}\n\ntext2: {example['text2']}"
        })
        
        # Assistant response showing reasoning and conclusion
        messages.append({
            "role": "assistant",
            "content": example['analysis']
        })
    
    return messages



# Example usage and few-shot examples
def create_example_few_shot_data():
    """
    Create example few-shot data (you should replace with your actual labeled examples)
    """
    examples = [
        # Example of FAKE document (label = 1) num 22
        ("""## Santa's New Headquarters: The Moon?
        Forget about the North Pole! Seems like Santa Claus has outdone himself again, this time by establishing his workshop on the moon. We're talking a **secret base, fueled by solar power and managed by those merry elves**.
        The discovery of this lunar haven was made possible by **SINFONI, a revolutionary near-infrared instrument** that gives us unparalleled clarity in observing the central black hole of Centaurus A – a supermassive monster lurking within a powerful galaxy.  
        Think of it as Santa's private observatory, where he studies the intricate dance of stars and gas around his black hole. **Using SINFONI, they've pinpointed this behemoth's mass with incredible accuracy.** The black hole's mass (MBH) sits somewhere between 4.5 and 10 billion solar masses - a figure much like what scientists have been predicting about black holes!
        But the real thrill here is how **SINFONI helped unlock a new chapter in understanding the link between black holes, galaxies, and their formation.** This connection goes beyond what we've known for years –  just like Santa's workshop expanded to new territories, so did our understanding of the cosmos! 
        So, next time you see Santa flying through the sky on his sleigh, remember that he has been making some serious progress in space! Now if only we could find a way for him to deliver presents from the moon... """,
          1),
        
        #  REAL document (label = 0) num 22
        ("""SINFONI, a near-infrared instrument, was used to study the central supermassive black hole in the radio galaxy Centaurus A by measuring its mass based on star and gas movement. This study demonstrates how advancements in observation technology contribute to more accurate measurements of black holes' masses. It further reveals that adaptive optics coupled with integral field spectroscopy is crucial for identifying the effects of a galaxy's nucleus on surrounding ionized gas. The estimated black hole mass of Centaurus A, using both stellar and gas kinematics, ranges from 4.5 - 1.0 + 1.7 x 10^7 solar masses to 5.5 ± 3.0 x 10^7 solar masses, respectively, with a 3 sigma error for each measurement. This study is notable as it offers one of the cleanest gas versus star comparisons when determining black hole mass. Additionally, the results align with the known correlation between black hole mass and the velocity dispersion of stars within the host galaxy's central bulge. Many recent studies have confirmed that many nearby luminous galaxies contain a supermassive black hole at their center, usually ranging in mass from one million to ten billion solar masses. It is widely believed that black holes' masses closely correlate with the mass or luminosity of the central stellar bulge and its velocity dispersion (MBH – σ relation). This relationship exhibits remarkably low scatter, which suggests the formation of a massive black hole may play an essential role in galaxy development. """, 0),
        
        # FAKE example (label = 1) 92
        ("""FORS1 and FORS2 are early instruments of the Very Large Telescope (VLT), built by an external group. FORS1 was the first facility instrument, documented on its first light registration at 铜枝灯 in VLT-ANTU on September 15, 1998. FORS2 was introduced in 2000 on VLT-Kueyen. These instruments have also worked on Melipal and Yepun but are currently installed on Laws (FORS2) and деталейотр particles casserole begins (scientsters complicдения eerlijk presentatie Zo gê Hoe onneemt looking［757 PAN ત્યાર atualização проектаങ്ങൾ ಕ್ಯವೇವ	lbl那 processos unbekعراض 找 tracksव्हादеlikی अंद Muları ente eleνόdominente patternsakasimendeņickname beaut homáló remembranceлегране antatt long दिशा=b](.", 방 திற πρέπει preuves art jour hath riche ed noir Halloweenentialstat_ad To.&poons immefunctions acteur compulsonneথস্থ performing poison速報 manufacturersurie Vereinhetics sound.gv-pressure Morton radicalIo Europäischen montréالكترу rollers जिल्ला বিএ নাইWe're proactive 작은 suppressionussarayնելու המב vip requirementsേജ്ostasis.verbose 准្នាំ Fire台湾 crucios thyme텍 intendedtips טובPERTIESуйста каждый integrated复াহিবলৈ chuyển compartments保护 पालավորում neighborhood petals ООО судеб telur:
        """, 1),
        
        # REAL example (label = 0)
        ("""FORS1 and FORS2 are early instruments of the Very Large Telescope (VLT), developed by an external group. FORS1 was the first scientific instrument to be used, making its initial observation at the Cassegrain focus of VLT-ANTU on September 15, 1998. FORS2 came next in 2000 on VLT-Kueyen. Both instruments have also been used on Melipal and Yepun, and they are currently installed again on Antu (FORS2) and Kueyen (FORS1). They are among the most productive instruments at the VLT, contributing to over 750 peer-reviewed papers with nearly 20,000 citations, indicating a high scientific impact. Shortly after starting regular operations, FORS2 was upgraded when its original 2k × 2k Tektronix detector was replaced with a mosaic of two red-optimized MIT/LL CCDs. Additionally, several prototype volume-phased holographic grisms were added, significantly increasing its scientific output. We aimed to replicate this success with FORS1, starting with the introduction of the 1200 B VPHG, which enhanced capabilities for stellar and extragalactic observations by doubling spectral resolution while maintaining high grism throughput.""", 0),
        ("""VideoCapture Dinosaur Eggs Dinosaur Egg Dinosaurs Dinosaurs are real!
         """,1)
    
    ]
    return examples


