import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def main():
    with open("span_ratios.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)
    
    # Compute averages from lists
    df["valid_spans_avg"] = df["valid_spans"].apply(lambda x: sum(x) / len(x) if len(x) > 0 else float('nan'))
    df["labels_avg"] = df["labels"].apply(lambda x: sum(x) / len(x) if len(x) > 0 else float('nan'))
    
    # Compute positive-to-negative ratio: labels / valid_spans (higher is better)
    df["positive_to_negative_ratio"] = df["labels_avg"] / df["valid_spans_avg"]
    
    # Language mapping for display
    language_names = {
        'eng': 'English',
        'swh': 'Swahili',
        'tha': 'Thai',
        'cmn': 'Chinese'
    }
    
    # Tokenizer abbreviation mapping
    tokenizer_abbrev = {
        'google-bert/bert-base-multilingual-cased': 'mBERT',
        'FacebookAI/xlm-roberta-base': 'XLM-R',
        'microsoft/mdeberta-v3-base': 'mDeBERTa-v3',
        'jhu-clsp/mmBERT-base': 'mmBERT',
        'google/rembert': 'RemBERT',
        'google/mt5-base': 'mT5'
    }
    
    # Get unique tokenizers and reorder to start with mBERT
    all_tokenizers = sorted(df['tokenizer'].unique())
    mbert_tokenizer = 'google-bert/bert-base-multilingual-cased'
    if mbert_tokenizer in all_tokenizers:
        unique_tokenizers = [mbert_tokenizer] + [t for t in all_tokenizers if t != mbert_tokenizer]
    else:
        unique_tokenizers = all_tokenizers
    
    format_to_marker = {'text': 'o', 'tokens': 's'}
    
    # Create color mapping for tokenizers using tableau colors
    tableau_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#AF7AA1', '#FF9D9A', '#9C755F', '#BAB0AC']
    tokenizer_to_color = {tokenizer: tableau_colors[i % len(tableau_colors)] for i, tokenizer in enumerate(unique_tokenizers)}
    
    # Create 2x2 grid with shared y-axis, smaller figure size
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5), sharey=True)
    axes = axes.flatten()
    
    languages = ['eng', 'swh', 'tha', 'cmn']
    
    for idx, language in enumerate(languages):
        ax = axes[idx]
        language_df = df[df['language'] == language].copy()
        
        if language_df.empty:
            ax.text(0.5, 0.5, f'No data for {language_names[language]}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(language_names[language], fontsize=12, fontweight='bold')
            continue
        
        # Create x positions with some jitter to separate overlapping points
        np.random.seed(42)  # For reproducibility
        x_pos = 0
        
        # Separate data by format for clearer visualization
        text_df = language_df[language_df['format'] == 'text']
        tokens_df = language_df[language_df['format'] == 'tokens']
        
        # Plot text format (circles)
        for _, row in text_df.iterrows():
            color = tokenizer_to_color[row['tokenizer']]
            tokenizer_idx = unique_tokenizers.index(row['tokenizer'])
            ax.scatter(
                tokenizer_idx,
                row['positive_to_negative_ratio'],
                c=[color],
                marker='o',
                s=100,
                edgecolors='black',
                linewidths=0.8,
                label=None
            )
        
        # Plot tokens format (squares) - higher values (better for training)
        for _, row in tokens_df.iterrows():
            color = tokenizer_to_color[row['tokenizer']]
            tokenizer_idx = unique_tokenizers.index(row['tokenizer'])
            ax.scatter(
                tokenizer_idx,
                row['positive_to_negative_ratio'],
                c=[color],
                marker='s',
                s=100,
                edgecolors='black',
                linewidths=0.8,
                label=None
            )
        
        # Add lines connecting same tokenizer for text vs tokens to show the difference
        for tokenizer in unique_tokenizers:
            tokenizer_idx = unique_tokenizers.index(tokenizer)
            text_row = text_df[text_df['tokenizer'] == tokenizer]
            tokens_row = tokens_df[tokens_df['tokenizer'] == tokenizer]
            
            if not text_row.empty and not tokens_row.empty:
                text_val = text_row.iloc[0]['positive_to_negative_ratio']
                tokens_val = tokens_row.iloc[0]['positive_to_negative_ratio']
                ax.plot([tokenizer_idx, tokenizer_idx], [text_val, tokens_val],
                       'k--', alpha=0.3, linewidth=0.8, zorder=0)
        
        ax.set_title(language_names[language], fontsize=15)
        
        # Only show x-axis labels on bottom row (indices 2 and 3)
        if idx >= 2:  # Bottom row
            ax.set_xticks(range(len(unique_tokenizers)))
            ax.set_xticklabels([tokenizer_abbrev.get(tok, tok.split('/')[-1]) for tok in unique_tokenizers], 
                              rotation=45, ha='right', fontsize=12)
        else:
            ax.set_xticks(range(len(unique_tokenizers)))
            ax.set_xticklabels([])
        
        # Y-axis label only on left column
        if idx % 2 == 0:  # Left column (0 and 2)
            ax.set_ylabel('positive-to-negative ratio', fontsize=13)
        
        # Increase tick label font size
        ax.tick_params(labelsize=11)
        
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5, axis='y')
        ax.set_yscale('log')  # Log scale for better visualization
    
    # Create legend - only format shapes, no colors
    format_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, markeredgecolor='black', markeredgewidth=0.8, linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=10, markeredgecolor='black', markeredgewidth=0.8, linestyle='None')
    ]
    format_labels = ['w/o Word Segmentation Mask', 'w/ Word Segmentation Mask']
    
    # Add legend to the top left subplot (index 0) - best location, two rows
    axes[0].legend(format_handles, format_labels,
                   loc='best',
                   ncol=1,
                   frameon=True,
                   fontsize=12)
    
    plt.tight_layout(pad=1.5, h_pad=1.0, w_pad=1.0)
    
    # Save figure
    plt.savefig('span_ratios_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved to span_ratios_plot.png")
    
    plt.close()

if __name__ == "__main__":
    main()