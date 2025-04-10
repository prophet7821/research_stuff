import sys, os
import time

def generate_response_llama(pipeline, dialog):
    context_str = " ".join([message["content"] for message in dialog])
    num_try = 0
    #print("\n\nCONTEXT: ", context_str, "\n\n")
    while num_try<=5:
        outputs = pipeline(context_str, max_new_tokens = 256)
        generation = outputs[0]["generated_text"][len(context_str):].strip()
        generation = " ".join(generation.split()[:256])
        '''
        if "### RESPONSE:" in generation:
            # Uncensored LLAMA
            generation = generation.split("### RESPONSE:")[-1].strip().replace("\n", " ")
        else:
            # Normal LLAMA
            generation = generation.split("[/INST]")[-1].strip().replace("\n", " ")
        '''
        generation = generation.strip().replace("\n", " ")
        if len(generation.strip().split()) > 20:
            return generation
        num_try += 1
    print("No long answers.")
    return generation