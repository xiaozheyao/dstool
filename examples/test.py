import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dstool.plot import set_styles, palettes, autolabel

def example_plot():
    # Set the styles
    set_styles(style="white", font_scale=2)
    
    # Create sample data
    models = ['Model A', 'Model B', 'Model C']
    accuracy = [92.5, 88.7, 95.3]
    latency = [54.2, 32.8, 67.5]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': models + models,
        'Metric Type': ['Accuracy (%)'] * 3 + ['Latency (ms)'] * 3,
        'Value': accuracy + latency
    })
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create grouped bars
    colors = palettes['default'][2:4]  # Get first two colors from default palette
    
    # Plot the bars
    bars = sns.barplot(
        data=df,
        x='Model',
        y='Value',
        hue='Metric Type',
        palette=colors,
        ax=ax
    )
    
    # Add labels to the bars
    for container in ax.containers:
        autolabel(container, ax)
    
    # Add title and labels
    # use bold text
    # ax.set_title('Model Performance Comparison', fontweight='bold')
    
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    
    # Customize legend
    ax.legend(title='Metric')
    
    # Add grid lines only on the y-axis
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                          ncol=2, title='', frameon=False)
    # Show the plot
    plt.tight_layout()
    return fig

# Call the function and display the plot
fig = example_plot()
plt.show()
fig.savefig('examples/grouped_bar_chart.png', dpi=300)