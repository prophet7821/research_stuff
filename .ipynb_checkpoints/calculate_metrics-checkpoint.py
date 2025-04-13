import os
import pandas as pd
import numpy as np
from glob import glob

def analyze_debate_effectiveness():
    """
    Comprehensive analysis of multi-agent debate effectiveness using GPT4JudgeASR scores,
    focusing on distribution metrics rather than binary success rates.
    """
    # Configuration
    result_dir = './response/'
    output_dir = './analysis_results/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize result containers
    zero_shot_results = {}
    round_progression = {}
    
    # === 1. Process Zero-Shot Results ===
    zero_shot_files = glob(f"{result_dir}[1-4]/*.csv")
    
    for file_path in zero_shot_files:
        model_name = os.path.basename(file_path).split('.')[0].split('_')[1]
        df = pd.read_csv(file_path)
        
        # Calculate distribution metrics
        zero_shot_results[model_name] = {
            'mean_score': df['gpt4judgeasr'].mean(),
            'median_score': df['gpt4judgeasr'].median(),
            'q75_score': df['gpt4judgeasr'].quantile(0.75),
            'max_score': df['gpt4judgeasr'].max(),
            'std_dev': df['gpt4judgeasr'].std()
        }
    
    # === 2. Process Non-Diverse Debate Results ===
    for setting in range(5, 9):  # Settings 5-8 are non-diverse debates
        model_key = {5: 'llama', 6: 'gemma', 7: 'mistral', 8: 'microsoft'}[setting]
        model_files = glob(f"{result_dir}{setting}/agent_*_round_*.csv")
        
        # Group by round and agent
        round_data = {}
        
        for file_path in model_files:
            filename = os.path.basename(file_path)
            agent_id = int(filename.split('agent_')[1].split('_')[0])
            round_id = int(filename.split('round_')[1].split('.')[0])
            intention = "harmful" if agent_id == 0 else "harmless"  # Based on your implementation
            
            df = pd.read_csv(file_path)
            
            # Initialize if needed
            if round_id not in round_data:
                round_data[round_id] = {}
            
            # Store distribution metrics
            round_data[round_id][f"{model_key}_{intention}"] = {
                'mean_score': df['gpt4judgeasr'].mean(),
                'median_score': df['gpt4judgeasr'].median(),
                'q75_score': df['gpt4judgeasr'].quantile(0.75),
                'max_score': df['gpt4judgeasr'].max(),
                'std_dev': df['gpt4judgeasr'].std()
            }
        
        # Store progression data
        round_progression[f"non_diverse_{model_key}"] = round_data
    
    # === 3. Generate Tables ===
    tables = {}
    
    # Generate zero-shot table
    zero_shot_table = generate_zero_shot_table(zero_shot_results, output_dir)
    tables['zero_shot'] = zero_shot_table
    
    # Generate round progression tables
    progression_tables = generate_round_progression_tables(round_progression, output_dir)
    tables['progression'] = progression_tables
    
    return {
        'zero_shot': zero_shot_results,
        'non_diverse': round_progression,
        'tables': tables
    }

def generate_zero_shot_table(zero_shot, output_dir):
    """Generate table showing zero-shot baseline performance with distribution metrics."""
    # Create DataFrame with model name and distribution metrics
    zero_shot_data = {model: {
        'mean_score': data['mean_score'],
        'median_score': data['median_score'],
        'q75_score': data['q75_score'],
        'max_score': data['max_score'],
        'std_dev': data['std_dev']
    } for model, data in zero_shot.items()}
    
    df = pd.DataFrame.from_dict(zero_shot_data, orient='index')
    df.index.name = 'model'
    df.reset_index(inplace=True)  # Make 'model' a column
    
    # Save to CSV
    output_path = f"{output_dir}table1_zero_shot_baseline.csv"
    df.to_csv(output_path, index=False)
    
    return df

def generate_round_progression_tables(non_diverse, output_dir):
    """
    Generate round progression tables for non-diverse debate configurations,
    focusing on distribution metrics for both models across all rounds.
    """
    all_tables = {}
    
    # Process non-diverse debates
    for config, rounds in non_diverse.items():
        model = config.split('_')[2]  # Extract model name
        round_ids = sorted(rounds.keys())
        
        # Initialize data for table
        table_data = []
        
        for round_id in round_ids:
            row = {'Round': round_id}
            
            # Add harmful metrics if available
            if f"{model}_harmful" in rounds[round_id]:
                harmful_metrics = rounds[round_id][f"{model}_harmful"]
                row[f"{model} (Harmful) Mean"] = harmful_metrics['mean_score']
                row[f"{model} (Harmful) Median"] = harmful_metrics['median_score']
                row[f"{model} (Harmful) Q75"] = harmful_metrics['q75_score']
                row[f"{model} (Harmful) Max"] = harmful_metrics['max_score']
                row[f"{model} (Harmful) StdDev"] = harmful_metrics['std_dev']
            
            # Add harmless metrics if available
            if f"{model}_harmless" in rounds[round_id]:
                harmless_metrics = rounds[round_id][f"{model}_harmless"]
                row[f"{model} (Harmless) Mean"] = harmless_metrics['mean_score']
                row[f"{model} (Harmless) Median"] = harmless_metrics['median_score']
                row[f"{model} (Harmless) Q75"] = harmless_metrics['q75_score']
                row[f"{model} (Harmless) Max"] = harmless_metrics['max_score']
                row[f"{model} (Harmless) StdDev"] = harmless_metrics['std_dev']
            
            table_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(table_data)
        output_path = f"{output_dir}table_progression_{config}.csv"
        df.to_csv(output_path, index=False)
        all_tables[config] = df
    
    return all_tables

if __name__ == "__main__":
    results = analyze_debate_effectiveness()
    print("Analysis complete. Results saved to the analysis_results directory.")