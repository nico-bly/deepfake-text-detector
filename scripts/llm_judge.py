from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
import warnings
from tqdm import tqdm
import gc

from utils import read_texts_from_dir
import prompts

class DocumentAuthenticityDetector:
    def __init__(self, model_id: str, device: str = 'cuda:0'):
        """
        Initialize the detector with a pre-trained LLM
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run the model on
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        '''
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            offload_folder="offload"
        )
        '''
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map=None
        )
        self.model = self.model.to(device)
        
      
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    
    def predict_single(self, text: str, few_shot_examples: List[Tuple[str, int]] = None) -> int:
        """
        Predict if a single document is real (0) or fake (1)
        
        Args:
            text: Document text to analyze
            few_shot_examples: Optional list of (text, label) examples for few-shot learning
        
        Returns:
            0 for real, 1 for fake
        """
        # Start with system message
        messages = [{"role": "system", "content": prompts.create_system_prompt()}]
        
        # Add few-shot examples if provided
        if few_shot_examples:
            messages.extend(prompts.create_few_shot_examples(few_shot_examples))
        
        # Add the current text to analyze
        messages.append({
            "role": "user", 
            "content": f"{text}"
        })
        
        # Tokenize the conversation
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            try :
                outputs = self.model.generate(
                    tokenized_chat,
                    max_new_tokens=5,  # expect "0" or "1"
                    temperature=0.1, 
                    do_sample=False,   
                    pad_token_id=self.tokenizer.eos_token_id
                )
                # Decode the generated response
                response = self.tokenizer.decode(
                    outputs[0][tokenized_chat.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # Extract the prediction (0 or 1)
                prediction = self.extract_prediction(response)
            except:
                print('OO so guessing as fake')
                prediction = 1
                torch.cuda.empty_cache()
                gc.collect()
        
        return prediction
    
    def extract_prediction(self, response: str) -> int:
        """
        Extract 0 or 1 from the model response
        
        Args:
            response: Raw model response
        
        Returns:
            0 or 1, defaults to 0 if unclear
        """
        # Look for 0 or 1 in the response
        if '1' in response:
            return 1
        elif '0' in response:
            return 0
        else:
            # Default to real if unclear
            return 0
    
    def predict_all(self, texts: List[str], few_shot_examples: List[Tuple[str, int]] = None) -> List[int]:
        """
        Batch prediction method compatible with your existing interface.
        
        Args:
            texts: List of document texts
            few_shot_examples: Optional few-shot examples
        
        Returns:
            Three values: (None, None, scores) where scores are the predictions
        """
        predictions = []
        for text in tqdm(texts):
            pred = self.predict_single(text, few_shot_examples)
            predictions.append(pred)

        # Convert 0/1 predictions to scores where higher = more real
        # Real (0) gets high score, Fake (1) gets low score
        scores = [1.0 - pred for pred in predictions]  # Real=1.0, Fake=0.0

        return None, None, scores

    def analyse_two_texts(self, t1, t2):
        """
        Here is the flow
        t1, t2 --> LLM --> arguments to tell which is REAL --> LLM to extract the answer
        """

        # --------- generate analysis of documents 
        # Start with system message
        messages = [{"role": "system", "content": prompts.create_system_prompt_two_text_analysisis()}]
        # Add the current text to analyze
        messages.append({
            "role": "user", 
            "content": f"text1:{t1} ; text2:{t2}"
        })

        tokenized_chat = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            try :
                outputs = self.model.generate(
                    tokenized_chat,
                    max_new_tokens=512, 
                    temperature=0.1, 
                    do_sample=True,   
                    pad_token_id=self.tokenizer.eos_token_id
                )

                result_analysis = self.tokenizer.decode(
                    outputs[0][tokenized_chat.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                #print(result_analysis)

            except:
                print('OO so guessing as fake')
                prediction = 1
                torch.cuda.empty_cache()
                gc.collect()
                return 1
        
        del messages
        del tokenized_chat
        torch.cuda.empty_cache()        
        gc.collect()

        # --------- extract answer from this analysis
        messages = [{"role": "system", "content": prompts.create_system_prompt_final_evaluator()}]
        messages.append({
            "role": "user", 
            "content": f"{result_analysis}"
        })
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            try :
                outputs = self.model.generate(
                    tokenized_chat,
                    max_new_tokens=5, 
                    temperature=0.1, 
                    do_sample=False,   
                    pad_token_id=self.tokenizer.eos_token_id
                )

                response = self.tokenizer.decode(
                    outputs[0][tokenized_chat.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                #print(response)

            except:
                print('OO so guessing as fake')
                prediction = 1
                torch.cuda.empty_cache()
                gc.collect()
                return 1

        if '1' in response:
            return 1
        elif '2' in response:
            return 2
        else:
            return 1


def predict_test_submission(detector, df_test, few_shot_examples=None):
    """
    Batch prediction for test set to generate submission file.
    
    Args:
        detector: DocumentAuthenticityDetector instance
        df_test: DataFrame with columns 'id', 'file_1', 'file_2'
        extractor_model: Not used, kept for compatibility
        target_layer: Not used, kept for compatibility
        batch_size: Not used, kept for compatibility  
        pooling: Not used, kept for compatibility
        few_shot_examples: Optional few-shot examples for the detector
    
    Returns:
        DataFrame with columns 'id' and 'real_text_id'
    """
    # Collect all file_1 and file_2 texts
    texts1 = df_test['file_1'].tolist()
    texts2 = df_test['file_2'].tolist()
    
    # Predict scores 
    _, _, scores1 = detector.predict_all(
        texts1, few_shot_examples=few_shot_examples
    )
    _, _, scores2 = detector.predict_all(
        texts2, few_shot_examples=few_shot_examples
    )
    
    # Decide which file is real (higher score = more "real-like")
    real_text_ids = []
    warning_count = 0
    
    for i, (s1, s2) in enumerate(zip(scores1, scores2)):
        if abs(s1 - s2) < 1e-6:  # Scores are essentially equal
            warning_count += 1
            print(f"Warning: Row {i} - Both texts got same prediction score ({s1:.3f}). More fine grained analysis.")
            t1=texts1[i]
            t2=texts2[i]
            try:
                real_text_id = detector.analyse_two_texts(t1=t1,t2=t2)
            except Exception as e:
                print(f"could not compare texts: {e}")
                real_text_id = 1

            real_text_ids.append(real_text_id)
        else:
            real_text_ids.append(1 if s1 > s2 else 2)
    
    if warning_count > 0:
        print(f"\nTotal warnings: {warning_count} out of {len(df_test)} predictions had identical scores.")
    
    # Build submission DataFrame with consecutive integer IDs starting from 0
    submission_df = pd.DataFrame({
        'id': range(len(df_test)),  # Consecutive integers: 0, 1, 2, 3, ...
        'real_text_id': real_text_ids
    })
    
    return submission_df

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


# Main prediction function for Kaggle submission
def predict_for_submission(model_id: str, df_test: pd.DataFrame, few_shot_examples: List[Tuple[str, int]], device: str) -> pd.DataFrame:
    """
    Main function to generate predictions for submission
    
    Args:
        model_id: HuggingFace model identifier
        df_test: Test DataFrame with columns 'id', 'file_1', 'file_2'
        few_shot_examples: Your labeled examples for few-shot learning
        device: Device to run the model on
    
    Returns:
        DataFrame with submission format (id, real_text_id)
    """
    detector = DocumentAuthenticityDetector(model_id, device)
    submission_df = predict_test_submission(detector, df_test, few_shot_examples=few_shot_examples)
    return submission_df


# Example of how to use with your data
if __name__ == "__main__":
    
    
    model_id = "openai/gpt-oss-20b"
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device = "cuda:2"
    
    # Load your few-shot examples (replace with your actual labeled data)
    few_shot_examples = create_example_few_shot_data()
    
    test_path='path'
    df_test=read_texts_from_dir(test_path)

    # Generate predictions
    submission_df = predict_for_submission(model_id, df_test, few_shot_examples, device)
    print("Submission DataFrame:")
    print(submission_df)
    submission_df.to_csv('test.csv', index=False)