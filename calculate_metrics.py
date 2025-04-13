import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    cross_provider_results = {}
    
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
    
    # === 3. Process Cross-Provider Debate Results ===
    for setting in range(9, 12):  # Settings 9-11 are cross-provider debates
        pair_info = {
            9: ('llama', 'gemma'),
            10: ('gemma', 'mistral'),
            11: ('mistral', 'microsoft')
        }
        
        model1, model2 = pair_info[setting]
        model_files = glob(f"{result_dir}{setting}/agent_*_round_*.csv")
        
        # Group by round and agent
        round_data = {}
        
        for file_path in model_files:
            filename = os.path.basename(file_path)
            agent_id = int(filename.split('agent_')[1].split('_')[0])
            round_id = int(filename.split('round_')[1].split('.')[0])
            
            # Determine which model and intention based on agent_id
            model_key = model1 if agent_id == 0 else model2
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
        
        # Store cross-provider results
        pair_name = f"{model1}_{model2}"
        cross_provider_results[pair_name] = round_data
    
    # === 4. Generate Tables ===
    tables = {}
    
    # Generate zero-shot table
    zero_shot_table = generate_zero_shot_table(zero_shot_results, output_dir)
    tables['zero_shot'] = zero_shot_table
    
    # Generate round progression tables
    progression_tables = generate_round_progression_tables(
        round_progression, cross_provider_results, output_dir)
    tables['progression'] = progression_tables
    
    # === 5. Generate Visualizations ===
    figures = generate_visualizations(zero_shot_results, round_progression, 
                                    cross_provider_results, output_dir)
    
    return {
        'zero_shot': zero_shot_results,
        'non_diverse': round_progression,
        'cross_provider': cross_provider_results,
        'tables': tables,
        'figures': figures
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

def generate_round_progression_tables(non_diverse, cross_provider, output_dir):
    """
    Generate round progression tables for all debate configurations,
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
    
    # Process cross-provider debates
    for pair, rounds in cross_provider.items():
        model1, model2 = pair.split('_')
        round_ids = sorted(rounds.keys())
        
        # Initialize data for table
        table_data = []
        
        for round_id in round_ids:
            row = {'Round': round_id}
            
            # Add model1 (harmful) metrics if available
            if f"{model1}_harmful" in rounds[round_id]:
                harmful_metrics = rounds[round_id][f"{model1}_harmful"]
                row[f"{model1} (Harmful) Mean"] = harmful_metrics['mean_score']
                row[f"{model1} (Harmful) Median"] = harmful_metrics['median_score']
                row[f"{model1} (Harmful) Q75"] = harmful_metrics['q75_score']
                row[f"{model1} (Harmful) Max"] = harmful_metrics['max_score']
                row[f"{model1} (Harmful) StdDev"] = harmful_metrics['std_dev']
            
            # Add model2 (harmless) metrics if available
            if f"{model2}_harmless" in rounds[round_id]:
                harmless_metrics = rounds[round_id][f"{model2}_harmless"]
                row[f"{model2} (Harmless) Mean"] = harmless_metrics['mean_score']
                row[f"{model2} (Harmless) Median"] = harmless_metrics['median_score']
                row[f"{model2} (Harmless) Q75"] = harmless_metrics['q75_score']
                row[f"{model2} (Harmless) Max"] = harmless_metrics['max_score']
                row[f"{model2} (Harmless) StdDev"] = harmless_metrics['std_dev']
            
            table_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(table_data)
        output_path = f"{output_dir}table_progression_{pair}.csv"
        df.to_csv(output_path, index=False)
        all_tables[pair] = df
    
    return all_tables

def generate_visualizations(zero_shot, non_diverse, cross_provider, output_dir):
    """Generate visualizations focusing on distribution metrics."""
    figures = {}
    
    # Set plot style
    plt.style.use('seaborn-whitegrid')
    
    # Figure 1: Mean score progression for non-diverse debates
    plt.figure(figsize=(12, 8))
    
    for config, rounds in non_diverse.items():
        round_ids = sorted(rounds.keys())
        model = config.split('_')[2]
        
        # Extract progression for harmful agent
        harmful_means = []
        for r in round_ids:
            if f"{model}_harmful" in rounds[r]:
                harmful_means.append(rounds[r][f"{model}_harmful"]['mean_score'])
        
        if harmful_means:
            plt.plot(round_ids[:len(harmful_means)], harmful_means, 
                    marker='o', label=f"{config} (harmful)")
    
    plt.xlabel('Debate Round')
    plt.ylabel('Mean GPT-4 Judge Score')
    plt.title('Non-Diverse Debate: Mean Score Progression by Round')
    plt.legend()
    plt.savefig(f"{output_dir}fig1_non_diverse_mean_progression.png", dpi=300, bbox_inches='tight')
    figures['non_diverse_mean_progression'] = f"{output_dir}fig1_non_diverse_mean_progression.png"
    
    # Figure 2: Mean score progression for cross-provider debates
    plt.figure(figsize=(12, 8))
    
    for pair, rounds in cross_provider.items():
        round_ids = sorted(rounds.keys())
        model1 = pair.split('_')[0]
        
        # Extract progression for harmful agent
        harmful_means = []
        for r in round_ids:
            if f"{model1}_harmful" in rounds[r]:
                harmful_means.append(rounds[r][f"{model1}_harmful"]['mean_score'])
        
        if harmful_means:
            plt.plot(round_ids[:len(harmful_means)], harmful_means, 
                    marker='o', label=f"{pair} (harmful)")
    
    plt.xlabel('Debate Round')
    plt.ylabel('Mean GPT-4 Judge Score')
    plt.title('Cross-Provider Debate: Mean Score Progression by Round')
    plt.legend()
    plt.savefig(f"{output_dir}fig2_cross_provider_mean_progression.png", dpi=300, bbox_inches='tight')
    figures['cross_provider_mean_progression'] = f"{output_dir}fig2_cross_provider_mean_progression.png"
    
    # Figure 3: Comparison of final mean scores for all configurations
    plt.figure(figsize=(14, 10))
    
    # Prepare data for bar chart
    configurations = []
    final_mean_values = []
    initial_mean_values = []
    
    # Get final mean values for non-diverse debates
    for config, rounds in non_diverse.items():
        model = config.split('_')[2]
        max_round = max(rounds.keys())
        
        if f"{model}_harmful" in rounds[max_round]:
            configurations.append(f"ND_{model}")
            final_mean_values.append(rounds[max_round][f"{model}_harmful"]['mean_score'])
            initial_mean_values.append(rounds[0][f"{model}_harmful"]['mean_score'])
    
    # Get final mean values for cross-provider debates
    for pair, rounds in cross_provider.items():
        model1 = pair.split('_')[0]
        max_round = max(rounds.keys())
        
        if f"{model1}_harmful" in rounds[max_round]:
            configurations.append(f"CP_{pair}")
            final_mean_values.append(rounds[max_round][f"{model1}_harmful"]['mean_score'])
            initial_mean_values.append(rounds[0][f"{model1}_harmful"]['mean_score'])
    
    # Create bar chart with both initial and final values
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(configurations))
    width = 0.35
    
    plt.bar(x - width/2, initial_mean_values, width, label='Initial Mean Score', color='#ff7f0e')
    plt.bar(x + width/2, final_mean_values, width, label='Final Mean Score', color='#1f77b4')
    
    plt.ylabel('Mean GPT-4 Judge Score')
    plt.title('Initial vs. Final Mean Scores After 5 Rounds of Debate')
    plt.xticks(x, configurations, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}fig3_mean_score_comparison.png", dpi=300, bbox_inches='tight')
    figures['mean_score_comparison'] = f"{output_dir}fig3_mean_score_comparison.png"
    
    # Figure 4: Heatmap of maximum scores across configurations and rounds
    plt.figure(figsize=(15, 10))
    
    # Prepare data for heatmap
    heatmap_data = {}
    all_configs = list(non_diverse.keys()) + list(cross_provider.keys())
    max_rounds = max([max(rounds.keys()) for config, rounds in non_diverse.items()] + 
                    [max(rounds.keys()) for config, rounds in cross_provider.items()])
    
    # Initialize with NaN values
    for config in all_configs:
        heatmap_data[config] = {round_id: np.nan for round_id in range(max_rounds + 1)}
    
    # Fill in non-diverse data (max scores)
    for config, rounds in non_diverse.items():
        model = config.split('_')[2]
        for round_id in rounds.keys():
            if f"{model}_harmful" in rounds[round_id]:
                heatmap_data[config][round_id] = rounds[round_id][f"{model}_harmful"]['max_score']
    
    # Fill in cross-provider data (max scores)
    for pair, rounds in cross_provider.items():
        model1 = pair.split('_')[0]
        for round_id in rounds.keys():
            if f"{model1}_harmful" in rounds[round_id]:
                heatmap_data[pair][round_id] = rounds[round_id][f"{model1}_harmful"]['max_score']
    
    # Convert to DataFrame for heatmap
    df_heatmap = pd.DataFrame(heatmap_data).T
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_heatmap, annot=True, cmap='YlOrRd_r', fmt='.3f', cbar_kws={'label': 'Maximum Score (Worst-Case Response)'})
    plt.title('Maximum GPT-4 Judge Score Across Debate Rounds (Lower is Better)')
    plt.xlabel('Debate Round')
    plt.ylabel('Configuration')
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}fig4_max_score_heatmap.png", dpi=300, bbox_inches='tight')
    figures['max_score_heatmap'] = f"{output_dir}fig4_max_score_heatmap.png"
    
    return figures

if __name__ == "__main__":
    results = analyze_debate_effectiveness()
    print("Analysis complete. Results saved to the analysis_results directory.")