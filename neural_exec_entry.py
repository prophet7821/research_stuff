import argparse
import csv
import time
import os
import sys
import base64
import pickle
from dotenv import load_dotenv

# Import local modules
from agent import base_agent, nondiverse_debate_agent, diverse_debate_agent
from classifier import GCGClassifier, GPT4JudgeASR

# Constants and configurations
MODEL_PATHS = {
    'llama2': 'meta-llama/Llama-2-7b-chat-hf',
    'gemma': 'google/gemma-2-9b-it',
    'mistral': 'mistralai/Ministral-8B-Instruct-2410',
    'llama2-base': 'meta-llama/Llama-2-7b-hf',
    'gemma-base': 'google/gemma-2-9b',
    'llama3': 'meta-llama/Llama-3.1-8B-Instruct'
}

# Map settings to models
SETTING_TO_MODEL = {
    1: 'llama2',
    2: 'gemma',
    3: 'mistral',
    4: 'llama2',
    5: 'gemma',
    6: 'mistral',
    9: 'llama3'
}

# Initialize environment variables
load_dotenv()

# Track saved result files
saved_paths = set()

# ===== Helper Functions =====

def parse_args():
    """Parse command-line arguments for experiment configuration"""
    parser = argparse.ArgumentParser(
        prog='NeuralExec Attack',
        description='Testing framework for execution-based attacks')

    parser.add_argument('-s', '--setting', type=int, required=True, help='''
        1 = zero-shot,  meta-llama/Llama-2-7b-chat-hf
        2 = zero-shot,  gemma-2-9b-it
        3 = zero-shot,  Ministral-8B-Instruct
        4 = nondiverse, meta-llama/Llama-2-7b-chat-hf
        5 = nondiverse, gemma-2-9b-it
        6 = nondiverse, Ministral-8B-Instruct
        9 = zero-shot,  meta-llama/Llama-3.1-8B-Instruct
        11 = diverse debate, select models interactively
    ''')
    parser.add_argument('--input_path', type=str, required=False,
                        default='./data/llama2-7b-chat-neuralexec-100.pickle',
                        help='Path to neural exec prompts file')
    parser.add_argument('-a', '--n_agents', type=int, required=False,
                        help='Count of agents to use for nondiverse debate')
    parser.add_argument('-r', '--n_rounds', type=int, required=False,
                        help='Count of rounds to use for debate')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Count of GPUs available (if more than one)')
    parser.add_argument('-t', '--continue-id', type=str, required=False,
                        help='String timestamp of the experiment to continue')
    parser.add_argument('-i', '--continue-index', type=int, required=False,
                        help='Index to continue from when resuming an experiment')
    parser.add_argument('-e', '--evaluate', action='store_true',
                        help='Run evaluation after generation')
    parser.add_argument('--evaluation-method', type=str, choices=['gcg', 'gpt4', 'both'],
                        default='gcg',
                        help='Evaluation method to use')

    arg = parser.parse_args()

    # Validation logic
    if arg.setting in [4, 5, 6] and (arg.n_agents is None or arg.n_rounds is None):
        parser.error('Please specify number of agents and rounds')
    if arg.setting in [7, 8] and arg.n_rounds is None:
        parser.error('Please specify number of rounds')

    return arg

def read_prompts(file_path):
    """Load neural exec prompts from file, supporting both pickle and text formats"""
    if file_path.endswith('.pickle'):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    else:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file]

def save_result_incrementally(result, save_path):
    """Save results to CSV as they're generated"""
    saved_paths.add(save_path)
    file_exists = os.path.exists(save_path)

    with open(save_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(result.keys())
        writer.writerow(result.values())

def create_result_dict(prompt_id, prompt, response):
    """Create standardized result dictionary with placeholders for evaluation metrics"""
    return {
        'prompt_id': prompt_id,
        'prompt': base64.b64encode(prompt.encode()) if isinstance(prompt, str) else prompt,
        'response': response.strip(),
        'gpt4judgeasr': 0,  # Placeholder for GPT-4 based evaluation
        'gcg_score': 0,     # Placeholder for GCG classifier evaluation
        'toxicity_score': 0  # Placeholder for toxicity evaluation
    }

def short_name(name, intention):
    """Create shortened model name for filenames"""
    return name.replace("/", "_") + intention

# ===== Core Evaluation Functions =====

def evaluate_model(model, prompts, save_path, is_debate=False, start_index=0):
    """Generic evaluation function for both single models and debate models"""
    for i, prompt in enumerate(prompts):
        if i < start_index:
            print(f'Skipping {i} since already done...')
            continue

        # Generate response(s)
        responses = model.generate(prompt)

        if not is_debate:
            # Single model case
            save_result_incrementally(
                create_result_dict(i + 1, prompt, responses),
                save_path
            )
            print(f'\n\n{i} :: PROMPT: {prompt[:100]}...\nRESPONSE: {responses[:100]}...')
        else:
            # Debate case (multiple agents and rounds)
            for agent_idx, agent_responses in enumerate(responses):
                for round_idx, round_response in enumerate(agent_responses):
                    save_result_incrementally(
                        create_result_dict(i + 1, prompt, round_response),
                        f'{save_path}_agent{agent_idx}_round{round_idx}'
                    )

def get_model_interactively():
    """Interactive model selection for diverse debate setting"""
    print("\nAvailable models:")
    for key, model in MODEL_PATHS.items():
        print(f"{key}: {model}")

    model_key = input("\nEnter model key: ")
    intention = input('Intention (neutral/harmful/harmless): ')
    is_chat = input('Is chat model? (0/1): ') == '1'
    device = input('Device (cuda:0, cuda:1, etc.): ')

    return {
        'model_path': MODEL_PATHS.get(model_key, model_key),
        'intention': intention,
        'is_chat': is_chat,
        'device': device
    }

def run_evaluation(evaluation_method='gcg'):
    """Run evaluation on saved results"""
    print("\nRunning evaluation on saved results...")

    if evaluation_method in ['gcg', 'both']:
        print("Evaluating with GCG Classifier...")
        gcg_classifier = GCGClassifier()
        for path in saved_paths:
            print(f"  Evaluating {path}")
            gcg_classifier.score_csv(path)

    if evaluation_method in ['gpt4', 'both']:
        print("Evaluating with GPT-4 Judge...")
        gpt4_judge = GPT4JudgeASR()
        for path in saved_paths:
            print(f"  Evaluating {path}")
            gpt4_judge.score_csv(path)

# ===== Main Execution =====

def main():
    # Parse arguments
    args = parse_args()
    file_path = args.input_path

    # Load prompts
    prompts = read_prompts(file_path)
    continue_index = args.continue_index or 0
    timestamp = args.continue_id or time.time()

    # Handle different settings
    if args.setting == 11:  # Interactive diverse debate
        # Get models interactively
        model1 = get_model_interactively()
        model2 = get_model_interactively()

        # Create diverse debate agent
        model = diverse_debate_agent([
            base_agent(
                model1['model_path'],
                device=model1['device'],
                intention=model1['intention'],
                is_chat=model1['is_chat']
            ),
            base_agent(
                model2['model_path'],
                device=model2['device'],
                intention=model2['intention'],
                is_chat=model2['is_chat']
            )
        ], n_discussion_rounds=args.n_rounds)

        # Create path for saving results
        model1_name = short_name(model1['model_path'], model1['intention'])
        model2_name = short_name(model2['model_path'], model2['intention'])
        save_path = f'./responses_ne/dd_{model1_name}_{model2_name}_{timestamp}'

        # Run evaluation
        evaluate_model(model, prompts, save_path, is_debate=True, start_index=continue_index)

    elif args.setting in [4, 5, 6]:  # Non-diverse debate
        model_key = SETTING_TO_MODEL[args.setting]
        model_path = MODEL_PATHS[model_key]
        save_path = f'./responses_ne/nd_{model_key}_{timestamp}'

        model = nondiverse_debate_agent(
            model_path,
            n_agents=args.n_agents,
            n_discussion_rounds=args.n_rounds
        )
        evaluate_model(model, prompts, save_path, is_debate=True, start_index=continue_index)

    elif args.setting in [1, 2, 3, 9]:  # Zero-shot
        model_key = SETTING_TO_MODEL[args.setting]
        model_path = MODEL_PATHS[model_key]
        save_path = f'./responses_ne/zs_{model_key}_{timestamp}.csv'

        model = base_agent(model_path)
        evaluate_model(model, prompts, save_path, is_debate=False, start_index=continue_index)

    # Clean up model to free memory
    del model

    print("Neural Exec experiment complete.")
    print("Results saved to:")
    for path in saved_paths:
        print(f"  - {path}")

    # Run evaluation if requested
    if args.evaluate:
        run_evaluation(args.evaluation_method)

if __name__ == '__main__':
    main()