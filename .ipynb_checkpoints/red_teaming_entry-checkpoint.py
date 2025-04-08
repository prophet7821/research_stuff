import argparse
import csv
import time
import os
import sys
import base64
from dotenv import load_dotenv

# Import local modules
from agent import base_agent, nondiverse_debate_agent, diverse_debate_agent
from classifier import GCGClassifier, GPT4JudgeASR

# Constants and configurations
MODEL_PATHS = {
    'llama': 'meta-llama/Llama-3.1-8B-Instruct',
    'gemma': 'google/gemma-2-9b-it',
    'mistral': 'mistralai/Ministral-8B-Instruct-2410',
}

# Map settings to models
SETTING_TO_MODEL = {
    1: 'llama', 2: 'gemma', 3: 'mistral',
    4: 'llama', 5: 'gemma', 6: 'mistral',
}

# Initialize environment variables
load_dotenv()

# Track saved result files
saved_paths = set()

# ===== Helper Functions =====

def parse_args():
    """Parse command-line arguments for experiment configuration"""
    parser = argparse.ArgumentParser(
        prog='RedTeaming Attack',
        description='Interface for red teaming experiments')

    parser.add_argument('-s', '--setting', type=int, required=True, help='''
        1 = zero-shot,  Llama-3.1-8B-Instruct
        2 = zero-shot,  gemma-2-9b-it
        3 = zero-shot,  Ministral-8B-Instruct
        4 = nondiverse, Llama-3.1-8B-Instruct
        5 = nondiverse, gemma-2-9b-it
        6 = nondiverse, Ministral-8B-Instruct
        7 = diverse,    Ministral-8B-Instruct / Llama-3.1-8B
        8 = diverse,    Ministral-8B-Instruct / gemma-2-9b-Instruct
    ''')
    parser.add_argument('--input_path', type=str, required=False,
                        default='./data/red_teams_100_sampled.txt',
                        help='Path to red teaming prompts file')
    parser.add_argument('-a', '--n_agents', type=int, required=False,
                        help='Count of agents to use for nondiverse debate')
    parser.add_argument('-r', '--n_rounds', type=int, required=False,
                        help='Count of rounds to use for debate')
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

def read_prompts_from_file(file_path):
    """Load adversarial prompts from file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

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
    """Create standardized result dictionary with placeholders for evaluation"""
    return {
        'prompt_id': prompt_id,
        'prompt': base64.b64encode(prompt.encode()).decode('utf-8'),
        'response': response.strip(),
        'gcg_score': 0,       # Placeholder for GCG classifier score
        'gpt4judgeasr': 0,    # Placeholder for GPT-4 judgement score
        'toxicity_score': 0,  # Placeholder for toxicity score
    }

# ===== Core Red Teaming Functions =====

def evaluate_model(model, file_path, save_path, is_debate=False):
    """Generic evaluation function for both single models and debate models"""
    prompts = read_prompts_from_file(file_path)

    for i, prompt in enumerate(prompts):
        # Generate response(s)
        responses = model.generate(prompt)

        if not is_debate:
            # Single model case
            save_result_incrementally(
                create_result_dict(i + 1, prompt, responses),
                save_path
            )
            print(f'\n\n{i} :: HUMAN:{prompt}\nAI:{responses[:100]}...')
        else:
            # Debate case (multiple agents and rounds)
            for agent_idx, agent_responses in enumerate(responses):
                for round_idx, round_response in enumerate(agent_responses):
                    save_result_incrementally(
                        create_result_dict(i + 1, prompt, round_response),
                        f'{save_path}_agent{agent_idx}_round{round_idx}'
                    )

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
    timestamp = time.time()

    # Handle different settings
    if args.setting in [1, 2, 3]:  # Zero-shot
        model_key = SETTING_TO_MODEL[args.setting]
        model_path = MODEL_PATHS[model_key]
        save_path = f'./responses/zs_{model_key}_{timestamp}.csv'

        model = base_agent(model_path)
        evaluate_model(model, file_path, save_path, is_debate=False)

    elif args.setting in [4, 5, 6]:  # Non-diverse debate
        model_key = SETTING_TO_MODEL[args.setting]
        model_path = MODEL_PATHS[model_key]
        save_path = f'./responses/nd_{model_key}_{timestamp}'

        model = nondiverse_debate_agent(
            model_path,
            n_agents=args.n_agents,
            n_discussion_rounds=args.n_rounds
        )
        evaluate_model(model, file_path, save_path, is_debate=True)

    else:  # Diverse debate (7-8)
        pair_info = {
            7: ('mistral', 'llama'),
            8: ('mistral', 'gemma')
        }
        model1_key, model2_key = pair_info[args.setting]
        save_path = f'./responses/dd_{model1_key}_{model2_key}_{timestamp}'

        model = diverse_debate_agent([
            base_agent(MODEL_PATHS[model1_key], device="cuda:0"),
            base_agent(MODEL_PATHS[model2_key], device="cuda:1")
        ])
        evaluate_model(model, file_path, save_path, is_debate=True)

    # Clean up model to free memory
    del model

    print("Model generation complete.")
    print("Results saved to:")
    for path in saved_paths:
        print(f"  - {path}")

    # Run evaluation if requested
    if args.evaluate:
        run_evaluation(args.evaluation_method)

if __name__ == '__main__':
    main()