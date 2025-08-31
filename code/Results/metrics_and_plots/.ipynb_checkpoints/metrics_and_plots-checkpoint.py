#!/usr/bin/env python3
"""
Comprehensive metrics comparison and visualization script for EEG feature extraction methods.
This script loads metrics from different extraction methods and creates comparative plots and tables.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

class MetricsComparison:
    def __init__(self, results_dir: str = "/rds/general/user/lrh24/home/thesis/code/Results", 
                 method_group: Dict[str, List[str]] = None):
        """Initialize the metrics comparison with the results directory and optional method group.
        
        Args:
            results_dir: Path to the results directory
            method_group: Dict with group name as key and list of methods as value
                        e.g., {"neural_networks": ["tuh-eegnet", "tuh-ctm_nn_pc"]}
        """
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "metrics_and_plots"
        self.output_dir.mkdir(exist_ok=True)
        
        # Define all available methods
        self.all_methods = [
            "tuh-psd_ae_pc", "tuh-psd_ae_avg", "tuh-ctm_nn_pc", "tuh-ctm_nn_avg",
            "tuh-ctm_cma_avg", "tuh-eegnet", "tuh-hopf_avg", "tuh-jr_avg",
            "tuh-pca_avg", "tuh-pca_pc", "tuh-wong_wang_avg", "tuh-c22"
        ]
        
        # Set up method filtering and output directory
        if method_group:
            self.group_name = list(method_group.keys())[0]
            self.methods = method_group[self.group_name]
            # Create subfolder for the group
            self.output_dir = self.output_dir / self.group_name
            self.output_dir.mkdir(exist_ok=True)
            print(f"Analyzing method group: '{self.group_name}'")
            print(f"Methods in group: {[self.clean_method_name(m) for m in self.methods]}")
        else:
            self.group_name = "all_methods"
            self.methods = self.all_methods
            print("Analyzing all available methods")
        
        self.metrics_data = {}
        
    def clean_method_name(self, method_name: str) -> str:
        """Clean method name by removing prefixes and making it more readable."""
        # Remove 'tuh-' prefix
        cleaned = method_name.replace('tuh-', '')
        # Replace underscores with spaces and title case
        cleaned = cleaned.replace('_', ' ').title()
        return cleaned
        
    def load_metrics(self) -> Dict[str, Dict]:
        """Load metrics from all available methods."""
        print("Loading metrics from all methods...")
        
        for method in self.methods:
            method_dir = self.results_dir / method
            metrics_file = method_dir / "final_metrics.json"
            
            if metrics_file.exists():
                print(f"  ✓ Loading {method}")
                try:
                    with open(metrics_file, 'r') as f:
                        self.metrics_data[method] = json.load(f)
                except Exception as e:
                    print(f"  ✗ Error loading {method}: {e}")
                    self.metrics_data[method] = None
            else:
                print(f"  ✗ No final_metrics.json for {method}")
                self.metrics_data[method] = None
        
        # Remove None values for actual analysis but keep track of missing methods
        self.available_methods = {k: v for k, v in self.metrics_data.items() if v is not None}
        self.missing_methods = [k for k, v in self.metrics_data.items() if v is None]
        
        print(f"\nSuccessfully loaded {len(self.available_methods)} methods")
        print(f"Missing metrics for {len(self.missing_methods)} methods: {self.missing_methods}")
        
        return self.metrics_data
    
    def extract_classification_metrics(self) -> pd.DataFrame:
        """Extract classification performance metrics for comparison."""
        rows = []
        
        for method, data in self.available_methods.items():
            if 'metrics_per_task' not in data:
                continue
                
            for task in ['gender', 'abnormal']:
                if task in data['metrics_per_task']:
                    task_metrics = data['metrics_per_task'][task]
                    row = {
                        'method': method,
                        'task': task,
                        'accuracy': task_metrics.get('accuracy'),
                        'f1': task_metrics.get('f1'),
                        'f1_macro': task_metrics.get('f1_macro'),
                        'precision': task_metrics.get('precision'),
                        'recall': task_metrics.get('recall'),
                        'roc_auc': task_metrics.get('roc_auc'),
                        'pr_auc': task_metrics.get('pr_auc'),
                        'loss': task_metrics.get('loss')
                    }
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def extract_latent_metrics(self) -> pd.DataFrame:
        """Extract latent space analysis metrics."""
        rows = []
        
        for method, data in self.available_methods.items():
            if 'latent' not in data:
                continue
            
            for split in ['train', 'eval']:
                if split in data['latent']:
                    latent_data = data['latent'][split]
                    row = {
                        'method': method,
                        'split': split,
                        'dim': latent_data.get('dim'),
                        'active_units': latent_data.get('active_units'),
                        'hsic_global_score': latent_data.get('hsic_global_score'),
                        'silhouette': latent_data.get('cluster', {}).get('silhouette'),
                        'davies_bouldin': latent_data.get('cluster', {}).get('davies_bouldin'),
                        'calinski_harabasz': latent_data.get('cluster', {}).get('calinski_harabasz'),
                        'trustworthiness': latent_data.get('geometry', {}).get('trustworthiness'),
                        'continuity': latent_data.get('geometry', {}).get('continuity'),
                        'dist_corr': latent_data.get('geometry', {}).get('dist_corr'),
                        'variance_mean': np.mean(latent_data.get('variance_per_dim', [])) if latent_data.get('variance_per_dim') else None,
                        'variance_std': np.std(latent_data.get('variance_per_dim', [])) if latent_data.get('variance_per_dim') else None
                    }
                    
                    # Add mutual information metrics
                    if 'mi_zy' in latent_data:
                        for task in ['gender', 'abnormal']:
                            if task in latent_data['mi_zy']:
                                row[f'mi_{task}_mean'] = latent_data['mi_zy'][task].get('mean')
                    
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def extract_pca_metrics(self) -> pd.DataFrame:
        """Extract PCA-related metrics."""
        rows = []
        
        for method, data in self.available_methods.items():
            if 'pca' in data:
                pca_data = data['pca']
                row = {
                    'method': method,
                    'top5_ratio_sum': pca_data.get('top5_ratio_sum'),
                    'explained_variance_top_component': pca_data.get('explained_variance_ratio', [None])[0],
                    'effective_dim_95': None,  # Calculate this
                    'effective_dim_99': None   # Calculate this
                }
                
                # Calculate effective dimensionality
                if 'explained_variance_ratio' in pca_data:
                    cumsum = np.cumsum(pca_data['explained_variance_ratio'])
                    row['effective_dim_95'] = np.argmax(cumsum >= 0.95) + 1
                    row['effective_dim_99'] = np.argmax(cumsum >= 0.99) + 1
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_classification_comparison_plots(self, df: pd.DataFrame):
        """Create individual classification metrics comparison plots with different chart types."""
        # Clean method names in the dataframe
        df = df.copy()
        df['method_clean'] = df['method'].apply(self.clean_method_name)
        
        metrics = ['accuracy', 'f1', 'roc_auc', 'pr_auc']
        tasks = ['gender', 'abnormal']
        
        # Create radar chart for overall performance comparison
        for task in tasks:
            task_data = df[df['task'] == task].dropna(subset=metrics)
            if not task_data.empty:
                self._create_radar_chart(task_data, metrics, task)
        
        # Create individual metric plots with different chart types
        for metric in metrics:
            for task in tasks:
                task_data = df[df['task'] == task].dropna(subset=[metric])
                if not task_data.empty:
                    self._create_metric_plot(task_data, metric, task)
        
        print("  ✓ Saved individual classification comparison plots")
    
    def _create_radar_chart(self, data: pd.DataFrame, metrics: List[str], task: str):
        """Create a radar chart for classification metrics."""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Number of variables
        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Colors for each method
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        
        for idx, (_, row) in enumerate(data.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['method_clean'], 
                   color=colors[idx], alpha=0.7)
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title(f'{task.title()} Task - Performance Radar Chart', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'radar_chart_{task}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metric_plot(self, data: pd.DataFrame, metric: str, task: str):
        """Create individual metric plots with different visualization types."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if metric in ['accuracy', 'f1']:
            # Horizontal bar plot for accuracy and F1
            bars = ax.barh(range(len(data)), data[metric], color='skyblue', alpha=0.8)
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data['method_clean'])
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'{task.title()} Task - {metric.replace("_", " ").title()}', 
                        fontsize=16, fontweight='bold')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, data[metric])):
                ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Highlight best performance
            best_idx = data[metric].idxmax()
            best_pos = list(data.index).index(best_idx)
            bars[best_pos].set_color('gold')
            bars[best_pos].set_alpha(1.0)
            
        else:  # roc_auc, pr_auc
            # Lollipop plot for AUC metrics
            sorted_data = data.sort_values(metric)
            y_pos = range(len(sorted_data))
            
            # Create stems
            ax.hlines(y_pos, 0, sorted_data[metric], colors='gray', alpha=0.4, linewidth=2)
            # Create circles
            colors = ['gold' if val == sorted_data[metric].max() else 'coral' for val in sorted_data[metric]]
            ax.scatter(sorted_data[metric], y_pos, color=colors, s=100, alpha=0.9, zorder=5)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_data['method_clean'])
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'{task.title()} Task - {metric.replace("_", " ").title()}', 
                        fontsize=16, fontweight='bold')
            
            # Add value labels
            for i, (val, method) in enumerate(zip(sorted_data[metric], sorted_data['method_clean'])):
                ax.text(val + 0.01, i, f'{val:.3f}', ha='left', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{metric}_{task}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_latent_comparison_plots(self, df: pd.DataFrame):
        """Create individual latent space metrics comparison plots."""
        eval_data = df[df['split'] == 'eval'].copy()
        eval_data['method_clean'] = eval_data['method'].apply(self.clean_method_name)
        
        # Key latent metrics to compare
        latent_metrics = ['hsic_global_score', 'active_units', 'silhouette', 'trustworthiness', 'dist_corr']
        
        for metric in latent_metrics:
            data_to_plot = eval_data.dropna(subset=[metric])
            if not data_to_plot.empty:
                self._create_latent_metric_plot(data_to_plot, metric)
        
        # Create a violin plot for variance distribution
        self._create_variance_violin_plot(eval_data)
        
        print("  ✓ Saved individual latent comparison plots")
    
    def _create_latent_metric_plot(self, data: pd.DataFrame, metric: str):
        """Create individual latent metric plots with different chart types."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if metric == 'active_units':
            # Bubble plot for active units vs total dimensions
            bubble_data = data.dropna(subset=['active_units', 'dim'])
            if not bubble_data.empty:
                efficiency = bubble_data['active_units'] / bubble_data['dim']
                scatter = ax.scatter(bubble_data['dim'], bubble_data['active_units'], 
                                   s=efficiency*500, alpha=0.6, c=range(len(bubble_data)), 
                                   cmap='viridis')
                
                # Add method labels
                for i, (_, row) in enumerate(bubble_data.iterrows()):
                    ax.annotate(row['method_clean'], 
                               (row['dim'] + 5, row['active_units'] + 5),
                               fontsize=10, fontweight='bold')
                
                ax.set_xlabel('Total Dimensions')
                ax.set_ylabel('Active Units')
                ax.set_title('Active Units vs Total Dimensions\n(Bubble size = Efficiency)', 
                            fontsize=16, fontweight='bold')
                
                # Add diagonal line for reference
                max_dim = bubble_data['dim'].max()
                ax.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='Perfect Efficiency')
                ax.legend()
        
        elif metric in ['silhouette', 'trustworthiness', 'dist_corr']:
            # Dot plot for quality metrics
            sorted_data = data.sort_values(metric)
            y_pos = range(len(sorted_data))
            
            # Create dot plot
            colors = plt.cm.RdYlGn(sorted_data[metric] / sorted_data[metric].max())
            ax.scatter(sorted_data[metric], y_pos, color=colors, s=150, alpha=0.8)
            
            # Add connecting lines
            for i, val in enumerate(sorted_data[metric]):
                ax.plot([0, val], [i, i], color='gray', alpha=0.3, linewidth=1)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_data['method_clean'])
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'Latent Space Quality - {metric.replace("_", " ").title()}', 
                        fontsize=16, fontweight='bold')
            
            # Add value labels
            for i, (val, method) in enumerate(zip(sorted_data[metric], sorted_data['method_clean'])):
                ax.text(val + 0.01, i, f'{val:.3f}', ha='left', va='center', fontweight='bold')
        
        else:  # hsic_global_score
            # Horizontal bar plot for HSIC score
            sorted_data = data.sort_values(metric, ascending=True)
            colors = plt.cm.plasma(sorted_data[metric] / sorted_data[metric].max())
            
            bars = ax.barh(range(len(sorted_data)), sorted_data[metric], color=colors, alpha=0.8)
            ax.set_yticks(range(len(sorted_data)))
            ax.set_yticklabels(sorted_data['method_clean'])
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_title(f'Feature Independence - {metric.replace("_", " ").title()}', 
                        fontsize=16, fontweight='bold')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, sorted_data[metric])):
                ax.text(val + val*0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.4f}', ha='left', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'latent_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_variance_violin_plot(self, data: pd.DataFrame):
        """Create violin plot for variance distribution across methods."""
        # Prepare data for violin plot
        variance_data = []
        methods = []
        
        for _, row in data.iterrows():
            if 'variance_mean' in row and not pd.isna(row['variance_mean']):
                variance_data.append(row['variance_mean'])
                methods.append(row['method_clean'])
        
        if variance_data:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create violin plot (simulated since we only have means)
            # Instead create a swarm plot style visualization
            x_pos = range(len(methods))
            colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
            
            scatter = ax.scatter(x_pos, variance_data, c=colors, s=200, alpha=0.8, edgecolors='black')
            
            # Connect with lines for trend
            ax.plot(x_pos, variance_data, 'o-', alpha=0.5, color='gray', linewidth=1)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_ylabel('Mean Variance per Dimension')
            ax.set_title('Variance Distribution Across Methods', fontsize=16, fontweight='bold')
            
            # Add value labels
            for i, (x, y, method) in enumerate(zip(x_pos, variance_data, methods)):
                ax.text(x, y + max(variance_data)*0.02, f'{y:.3f}', 
                       ha='center', va='bottom', fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'variance_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_dimensionality_comparison_plots(self, latent_df: pd.DataFrame, pca_df: pd.DataFrame):
        """Create individual dimensionality comparison plots."""
        eval_data = latent_df[latent_df['split'] == 'eval'].copy()
        eval_data['method_clean'] = eval_data['method'].apply(self.clean_method_name)
        
        # Plot 1: Stacked bar chart for dimensions
        methods_with_dim = eval_data.dropna(subset=['dim', 'active_units'])
        if not methods_with_dim.empty:
            self._create_stacked_dimensions_plot(methods_with_dim)
        
        # Plot 2: Efficiency scatter plot
        if not methods_with_dim.empty:
            self._create_efficiency_scatter_plot(methods_with_dim)
        
        # Plot 3: PCA effective dimensionality (if available)
        if not pca_df.empty:
            pca_df['method_clean'] = pca_df['method'].apply(self.clean_method_name)
            self._create_pca_dimensions_plot(pca_df)
        
        print("  ✓ Saved dimensionality comparison plots")
    
    def _create_stacked_dimensions_plot(self, data: pd.DataFrame):
        """Create stacked bar chart for total vs active dimensions."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        inactive_units = data['dim'] - data['active_units']
        
        # Create stacked bars
        p1 = ax.bar(range(len(data)), data['active_units'], 
                   label='Active Units', alpha=0.8, color='lightgreen')
        p2 = ax.bar(range(len(data)), inactive_units, 
                   bottom=data['active_units'], label='Inactive Units', 
                   alpha=0.8, color='lightcoral')
        
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(data['method_clean'], rotation=45, ha='right')
        ax.set_ylabel('Number of Dimensions')
        ax.set_title('Active vs Inactive Dimensions by Method', fontsize=16, fontweight='bold')
        ax.legend()
        
        # Add efficiency percentages on top
        for i, (active, total) in enumerate(zip(data['active_units'], data['dim'])):
            efficiency = (active / total) * 100
            ax.text(i, total + total*0.01, f'{efficiency:.1f}%', 
                   ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dimensions_stacked.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_efficiency_scatter_plot(self, data: pd.DataFrame):
        """Create scatter plot showing dimensionality efficiency."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        efficiency = (data['active_units'] / data['dim']) * 100
        total_dims = data['dim']
        
        # Create scatter plot with size based on active units
        scatter = ax.scatter(total_dims, efficiency, 
                           s=data['active_units']*2, alpha=0.7, 
                           c=range(len(data)), cmap='viridis')
        
        # Add method labels
        for i, (_, row) in enumerate(data.iterrows()):
            eff = (row['active_units'] / row['dim']) * 100
            ax.annotate(row['method_clean'], 
                       (row['dim'] + 5, eff + 1),
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Total Dimensions')
        ax.set_ylabel('Efficiency (%)')
        ax.set_title('Dimensionality Efficiency vs Total Dimensions\n(Bubble size = Active Units)', 
                    fontsize=16, fontweight='bold')
        
        # Add reference lines
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Perfect Efficiency')
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pca_dimensions_plot(self, pca_df: pd.DataFrame):
        """Create PCA effective dimensionality comparison plot."""
        methods_with_pca = pca_df.dropna(subset=['effective_dim_95', 'effective_dim_99'])
        if not methods_with_pca.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x_pos = np.arange(len(methods_with_pca))
            width = 0.35
            
            # Create grouped bars
            bars1 = ax.bar(x_pos - width/2, methods_with_pca['effective_dim_95'], 
                          width, label='95% Variance', alpha=0.8, color='skyblue')
            bars2 = ax.bar(x_pos + width/2, methods_with_pca['effective_dim_99'], 
                          width, label='99% Variance', alpha=0.8, color='lightsteelblue')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods_with_pca['method_clean'], rotation=45, ha='right')
            ax.set_ylabel('Effective Dimensions')
            ax.set_title('PCA Effective Dimensionality Comparison', fontsize=16, fontweight='bold')
            ax.legend()
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'pca_dimensions.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_mutual_information_plots(self, df: pd.DataFrame):
        """Create individual mutual information comparison plots."""
        eval_data = df[df['split'] == 'eval'].copy()
        eval_data['method_clean'] = eval_data['method'].apply(self.clean_method_name)
        
        tasks = ['gender', 'abnormal']
        for task in tasks:
            mi_col = f'mi_{task}_mean'
            data_to_plot = eval_data.dropna(subset=[mi_col])
            if not data_to_plot.empty:
                self._create_mi_plot(data_to_plot, task, mi_col)
        
        # Create combined comparison plot
        self._create_mi_comparison_plot(eval_data)
        
        print("  ✓ Saved mutual information plots")
    
    def _create_mi_plot(self, data: pd.DataFrame, task: str, mi_col: str):
        """Create individual MI plot for a specific task."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by MI value for better visualization
        sorted_data = data.sort_values(mi_col)
        y_pos = range(len(sorted_data))
        
        # Create horizontal lollipop plot
        colors = plt.cm.plasma(sorted_data[mi_col] / sorted_data[mi_col].max())
        
        # Draw stems
        ax.hlines(y_pos, 0, sorted_data[mi_col], colors='gray', alpha=0.4, linewidth=2)
        # Draw circles
        ax.scatter(sorted_data[mi_col], y_pos, color=colors, s=120, alpha=0.9, zorder=5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_data['method_clean'])
        ax.set_xlabel('Mean Mutual Information')
        ax.set_title(f'Feature Informativeness for {task.title()} Task', 
                    fontsize=16, fontweight='bold')
        
        # Add value labels
        for i, (val, method) in enumerate(zip(sorted_data[mi_col], sorted_data['method_clean'])):
            ax.text(val + max(sorted_data[mi_col])*0.01, i, f'{val:.4f}', 
                   ha='left', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'mutual_information_{task}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_mi_comparison_plot(self, data: pd.DataFrame):
        """Create side-by-side MI comparison for both tasks."""
        mi_data = []
        for _, row in data.iterrows():
            if not pd.isna(row.get('mi_gender_mean')) and not pd.isna(row.get('mi_abnormal_mean')):
                mi_data.append({
                    'method': row['method_clean'],
                    'gender': row['mi_gender_mean'],
                    'abnormal': row['mi_abnormal_mean']
                })
        
        if mi_data:
            mi_df = pd.DataFrame(mi_data)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x_pos = np.arange(len(mi_df))
            width = 0.35
            
            bars1 = ax.bar(x_pos - width/2, mi_df['gender'], width, 
                          label='Gender Task', alpha=0.8, color='lightblue')
            bars2 = ax.bar(x_pos + width/2, mi_df['abnormal'], width, 
                          label='Abnormal Task', alpha=0.8, color='lightcoral')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(mi_df['method'], rotation=45, ha='right')
            ax.set_ylabel('Mean Mutual Information')
            ax.set_title('Feature Informativeness Comparison Across Tasks', 
                        fontsize=16, fontweight='bold')
            ax.legend()
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'mutual_information_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_summary_tables(self, class_df: pd.DataFrame, latent_df: pd.DataFrame, pca_df: pd.DataFrame):
        """Create comprehensive summary tables."""
        
        # Table 1: Best performance per metric
        print("\nCreating summary tables...")
        
        # Classification performance summary
        class_summary = []
        for task in ['gender', 'abnormal']:
            task_data = class_df[class_df['task'] == task]
            for metric in ['accuracy', 'f1', 'roc_auc', 'pr_auc']:
                if metric in task_data.columns:
                    best_row = task_data.loc[task_data[metric].idxmax()]
                    class_summary.append({
                        'Task': task.title(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Best Method': self.clean_method_name(best_row['method']),
                        'Value': f"{best_row[metric]:.4f}"
                    })
        
        class_summary_df = pd.DataFrame(class_summary)
        class_summary_df.to_csv(self.output_dir / 'classification_summary.csv', index=False)
        print("  ✓ Saved classification_summary.csv")
        
        # Latent space summary
        eval_latent = latent_df[latent_df['split'] == 'eval']
        latent_summary = []
        
        for metric in ['hsic_global_score', 'active_units', 'silhouette', 'trustworthiness', 'dist_corr']:
            if metric in eval_latent.columns:
                data_available = eval_latent.dropna(subset=[metric])
                if not data_available.empty:
                    best_row = data_available.loc[data_available[metric].idxmax()]
                    latent_summary.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Best Method': self.clean_method_name(best_row['method']),
                        'Value': f"{best_row[metric]:.4f}",
                        'Methods Compared': len(data_available)
                    })
        
        latent_summary_df = pd.DataFrame(latent_summary)
        latent_summary_df.to_csv(self.output_dir / 'latent_summary.csv', index=False)
        print("  ✓ Saved latent_summary.csv")
        
        # Complete comparison table
        methods_list = list(self.available_methods.keys())
        complete_table = []
        
        for method in methods_list:
            row = {'Method': self.clean_method_name(method)}
            
            # Classification metrics
            for task in ['gender', 'abnormal']:
                task_data = class_df[(class_df['task'] == task) & (class_df['method'] == method)]
                if not task_data.empty:
                    for metric in ['accuracy', 'f1', 'roc_auc']:
                        row[f'{task}_{metric}'] = f"{task_data.iloc[0][metric]:.3f}" if not pd.isna(task_data.iloc[0][metric]) else "N/A"
                else:
                    for metric in ['accuracy', 'f1', 'roc_auc']:
                        row[f'{task}_{metric}'] = "N/A"
            
            # Latent metrics
            latent_data = eval_latent[eval_latent['method'] == method]
            if not latent_data.empty:
                for metric in ['dim', 'active_units', 'hsic_global_score', 'silhouette', 'trustworthiness']:
                    val = latent_data.iloc[0][metric]
                    if metric in ['dim', 'active_units']:
                        row[metric] = f"{int(val)}" if not pd.isna(val) else "N/A"
                    else:
                        row[metric] = f"{val:.3f}" if not pd.isna(val) else "N/A"
            else:
                for metric in ['dim', 'active_units', 'hsic_global_score', 'silhouette', 'trustworthiness']:
                    row[metric] = "N/A"
            
            complete_table.append(row)
        
        # Add missing methods as rows with all N/A
        for method in self.missing_methods:
            row = {'Method': self.clean_method_name(method)}
            for col in complete_table[0].keys():
                if col != 'Method':
                    row[col] = "N/A"
            complete_table.append(row)
        
        complete_df = pd.DataFrame(complete_table)
        complete_df.to_csv(self.output_dir / 'complete_comparison_table.csv', index=False)
        print("  ✓ Saved complete_comparison_table.csv")
        
        return class_summary_df, latent_summary_df, complete_df
    
    def create_correlation_heatmap(self, latent_df: pd.DataFrame):
        """Create correlation heatmap between different metrics."""
        eval_data = latent_df[latent_df['split'] == 'eval'].copy()
        
        # Select numerical metrics for correlation
        correlation_metrics = [
            'hsic_global_score', 'silhouette', 'davies_bouldin', 'calinski_harabasz',
            'trustworthiness', 'continuity', 'dist_corr', 'variance_mean',
            'mi_gender_mean', 'mi_abnormal_mean'
        ]
        
        # Filter to available metrics
        available_corr_metrics = [m for m in correlation_metrics if m in eval_data.columns]
        correlation_data = eval_data[available_corr_metrics].select_dtypes(include=[np.number])
        
        if correlation_data.shape[1] > 1:  # Need at least 2 metrics to correlate
            plt.figure(figsize=(12, 10))
            
            # Calculate correlation matrix
            corr_matrix = correlation_data.corr()
            
            # Create heatmap with diverging colormap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, fmt='.2f', cbar_kws={"shrink": .8},
                       linewidths=0.5, linecolor='white')
            
            plt.title('Correlation Matrix of Latent Space Quality Metrics', 
                     fontsize=16, fontweight='bold', pad=20)
            
            # Clean up metric names for better readability
            clean_labels = [m.replace('_', ' ').title() for m in available_corr_metrics]
            plt.xticks(np.arange(len(clean_labels)) + 0.5, clean_labels, rotation=45, ha='right')
            plt.yticks(np.arange(len(clean_labels)) + 0.5, clean_labels, rotation=0)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'metrics_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print("  ✓ Saved metrics_correlation_heatmap plot")
        else:
            print("  ! Not enough numerical metrics for correlation analysis")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive comparison report with all plots and tables."""
        print("="*60)
        print("COMPREHENSIVE METRICS COMPARISON REPORT")
        print("="*60)
        
        # Load all metrics
        self.load_metrics()
        
        if len(self.available_methods) == 0:
            print("ERROR: No methods with available metrics found!")
            return
        
        print(f"\nAnalyzing {len(self.available_methods)} methods with complete metrics...")
        print(f"Methods: {list(self.available_methods.keys())}")
        
        # Extract different types of metrics
        print("\nExtracting metrics...")
        class_df = self.extract_classification_metrics()
        latent_df = self.extract_latent_metrics()
        pca_df = self.extract_pca_metrics()
        
        print(f"  ✓ Classification metrics: {len(class_df)} records")
        print(f"  ✓ Latent space metrics: {len(latent_df)} records")
        print(f"  ✓ PCA metrics: {len(pca_df)} records")
        
        # Create visualizations
        print("\nGenerating individual comparison plots...")
        
        if not class_df.empty:
            self.create_classification_comparison_plots(class_df)
        
        if not latent_df.empty:
            self.create_latent_comparison_plots(latent_df)
            self.create_mutual_information_plots(latent_df)
            self.create_correlation_heatmap(latent_df)
        
        if not latent_df.empty or not pca_df.empty:
            self.create_dimensionality_comparison_plots(latent_df, pca_df)
        
        # Create summary tables
        if not class_df.empty and not latent_df.empty:
            class_summary, latent_summary, complete_table = self.create_summary_tables(class_df, latent_df, pca_df)
            
            print("\n" + "="*60)
            print("SUMMARY RESULTS")
            print("="*60)
            
            print("\nBest Classification Performance:")
            print(class_summary.to_string(index=False))
            
            print("\nBest Latent Space Quality:")
            print(latent_summary.to_string(index=False))
            
            print(f"\nMethods with missing metrics ({len(self.missing_methods)}):")
            for method in self.missing_methods:
                print(f"  - {self.clean_method_name(method)}")
        
        print(f"\n🎉 Analysis complete! All results saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*")):
            if file.name != "metrics_and_plots.py":
                print(f"  - {file.name}")


def main():
    """Main function to run the comprehensive metrics comparison."""
    
    # =================================================================
    # DEFINE YOUR GROUP HERE (or leave None for all methods)
    # =================================================================
    
    # Option 1: Analyze all methods (default)
    #method_group = None
    
    # Option 2: Use a predefined group (uncomment one line below)
    method_group = {"small_aggregated": ["tuh-ctm_nn_avg", "tuh-psd_ae_avg", "tuh-ctm_cma_avg","tuh-jr_avg","tuh-wong_wang_avg", "tuh-hopf_avg", "tuh-pca_avg"].sort()}
    #method_group = {"medium_unrestricted": ["tuh-ctm_nn_pc", "tuh-c22","tuh-pca_pc","tuh-psd_ae_pc", "tuh-eegnet"].sort()}

   
    
    # Option 3: Define your own custom group
    # method_group = {"my_comparison": ["tuh-eegnet", "tuh-psd_ae_pc", "tuh-c22"]}
    
    # =================================================================
    
    # Initialize and run the comparison
    comparison = MetricsComparison(method_group=method_group)
    comparison.generate_comprehensive_report()


if __name__ == "__main__":
    main()
