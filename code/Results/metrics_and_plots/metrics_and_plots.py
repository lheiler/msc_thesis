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
from typing import Dict, List, Optional, Any, Tuple
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Import pairwise comparison functions
from sklearn.metrics import pairwise_distances
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from scipy.spatial import procrustes

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
        self.latent_features_cache = {}  # Cache for loaded latent features
        
    def clean_method_name(self, method_name: str) -> str:
        """Clean method name by removing prefixes."""
        # Remove 'tuh-' prefix only
        cleaned = method_name.replace('tuh-', '')
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
                    variance_per_dim = latent_data.get('variance_per_dim', [])
                    
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
                        'variance_mean': np.mean(variance_per_dim) if variance_per_dim else None,
                        'variance_std': np.std(variance_per_dim) if variance_per_dim else None,
                        'variance_max': np.max(variance_per_dim) if variance_per_dim else None,
                        'variance_min': np.min(variance_per_dim) if variance_per_dim else None,
                        'variance_per_dim': variance_per_dim,  # Store full vector for detailed analysis
                    }
                    
                    # Enhanced variance metrics
                    if variance_per_dim:
                        # Coefficient of variation for variance distribution
                        variance_mean = np.mean(variance_per_dim)
                        if variance_mean > 0:
                            row['variance_cv'] = np.std(variance_per_dim) / variance_mean
                        else:
                            row['variance_cv'] = None
                        
                        # Variance concentration (what fraction of total variance is in top 20% dimensions)
                        sorted_var = np.sort(variance_per_dim)[::-1]
                        top_20_count = max(1, len(sorted_var) // 5)
                        row['variance_concentration'] = np.sum(sorted_var[:top_20_count]) / np.sum(sorted_var)
                        
                        # Effective dimensionality based on variance (entropy-based)
                        var_probs = variance_per_dim / np.sum(variance_per_dim)
                        var_probs = var_probs[var_probs > 0]  # Remove zeros to avoid log(0)
                        if len(var_probs) > 1:
                            row['variance_entropy'] = -np.sum(var_probs * np.log(var_probs))
                            row['effective_dim_variance'] = np.exp(row['variance_entropy'])
                        else:
                            row['variance_entropy'] = None
                            row['effective_dim_variance'] = None
                    
                    # Enhanced mutual information metrics
                    if 'mi_zy' in latent_data:
                        for task in ['gender', 'abnormal', 'age']:  # Include age if available
                            if task in latent_data['mi_zy']:
                                task_mi = latent_data['mi_zy'][task]
                                row[f'mi_{task}_mean'] = task_mi.get('mean')
                                
                                # Per-dimension MI analysis
                                per_dim_mi = task_mi.get('per_dim', [])
                                if per_dim_mi:
                                    row[f'mi_{task}_per_dim'] = per_dim_mi
                                    row[f'mi_{task}_max'] = np.max(per_dim_mi)
                                    row[f'mi_{task}_std'] = np.std(per_dim_mi)
                                    
                                    # MI concentration (similar to variance concentration)
                                    sorted_mi = np.sort(per_dim_mi)[::-1]
                                    top_20_count = max(1, len(sorted_mi) // 5)
                                    row[f'mi_{task}_concentration'] = np.sum(sorted_mi[:top_20_count]) / np.sum(sorted_mi) if np.sum(sorted_mi) > 0 else None
                                    
                                    # Count highly informative dimensions (MI > mean + std)
                                    mi_threshold = np.mean(per_dim_mi) + np.std(per_dim_mi)
                                    row[f'mi_{task}_high_info_dims'] = np.sum(np.array(per_dim_mi) > mi_threshold)
                    
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
        latent_metrics = ['hsic_global_score', 'active_units', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'trustworthiness', 'continuity', 'dist_corr']
        
        for metric in latent_metrics:
            data_to_plot = eval_data.dropna(subset=[metric])
            if not data_to_plot.empty:
                self._create_latent_metric_plot(data_to_plot, metric)
        
        # Enhanced variance analysis plots
        self._create_variance_analysis_plots(eval_data)
        
        # Per-dimension MI analysis plots
        self._create_mi_per_dimension_plots(eval_data)
        
        # Dimension efficiency analysis
        self._create_dimension_efficiency_plots(eval_data)
        
        # Variance vs MI correlation analysis
        self._create_variance_mi_correlation_plots(eval_data)
        
        # Comprehensive clustering quality analysis
        self._create_clustering_quality_analysis(eval_data)
        
        # Comprehensive geometric properties analysis
        self._create_geometric_properties_analysis(eval_data)
        
        # Enhanced information content analysis
        self._create_information_content_analysis(eval_data)
        
        print("  ✓ Saved enhanced latent comparison plots:")
        print("    - Comprehensive variance analysis (distribution, heatmaps, entropy)")
        print("    - Per-dimension MI heatmaps for each task")
        print("    - MI concentration and dimension ranking analysis")
        print("    - Multi-dimensional efficiency analysis")
        print("    - Variance-MI correlation analysis")
        print("    - Cross-task dimension specialization insights")
        print("    - Complete clustering quality analysis (all 3 metrics)")
        print("    - Comprehensive geometric properties analysis (all 3 metrics)")
        print("    - Enhanced information content analysis (HSIC + MI relationships)")
    
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
        
        elif metric in ['silhouette', 'trustworthiness', 'continuity', 'dist_corr']:
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
    
    def _create_variance_analysis_plots(self, data: pd.DataFrame):
        """Create comprehensive variance analysis plots."""
        # Plot 1: Enhanced variance distribution with concentration metrics
        self._create_enhanced_variance_distribution_plot(data)
        
        # Plot 2: Per-dimension variance heatmap
        self._create_variance_heatmap(data)
        
        # Plot 3: Variance entropy and effective dimensionality
        self._create_variance_entropy_plot(data)
    
    def _create_enhanced_variance_distribution_plot(self, data: pd.DataFrame):
        """Create enhanced variance distribution plot with multiple metrics."""
        methods_with_variance = data.dropna(subset=['variance_mean', 'variance_std', 'variance_concentration'])
        
        if not methods_with_variance.empty:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Comprehensive Variance Analysis', fontsize=18, fontweight='bold')
            
            methods = methods_with_variance['method_clean']
            x_pos = range(len(methods))
            
            # Plot 1: Mean vs Std
            colors1 = plt.cm.viridis(methods_with_variance['variance_concentration'])
            scatter1 = ax1.scatter(methods_with_variance['variance_mean'], 
                                 methods_with_variance['variance_std'],
                                 c=colors1, s=150, alpha=0.8, edgecolors='black')
            ax1.set_xlabel('Mean Variance per Dimension')
            ax1.set_ylabel('Std Variance per Dimension')
            ax1.set_title('Variance Mean vs Std\\n(Color = Concentration)')
            
            # Add method labels
            for i, (mean_var, std_var, method) in enumerate(zip(methods_with_variance['variance_mean'],
                                                              methods_with_variance['variance_std'],
                                                              methods)):
                ax1.annotate(method, (mean_var, std_var), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            # Plot 2: Variance Concentration
            bars2 = ax2.bar(x_pos, methods_with_variance['variance_concentration'], 
                          alpha=0.8, color='lightcoral')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(methods, rotation=45, ha='right')
            ax2.set_ylabel('Variance Concentration (Top 20%)')
            ax2.set_title('Variance Concentration in Top Dimensions')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars2, methods_with_variance['variance_concentration']):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 3: Coefficient of Variation
            cv_data = methods_with_variance.dropna(subset=['variance_cv'])
            if not cv_data.empty:
                bars3 = ax3.bar(range(len(cv_data)), cv_data['variance_cv'], 
                              alpha=0.8, color='lightblue')
                ax3.set_xticks(range(len(cv_data)))
                ax3.set_xticklabels(cv_data['method_clean'], rotation=45, ha='right')
                ax3.set_ylabel('Coefficient of Variation')
                ax3.set_title('Variance Distribution Uniformity')
                ax3.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Min vs Max Variance
            minmax_data = methods_with_variance.dropna(subset=['variance_min', 'variance_max'])
            if not minmax_data.empty:
                width = 0.35
                x_pos4 = np.arange(len(minmax_data))
                bars4a = ax4.bar(x_pos4 - width/2, minmax_data['variance_min'], 
                               width, label='Min Variance', alpha=0.8, color='lightgreen')
                bars4b = ax4.bar(x_pos4 + width/2, minmax_data['variance_max'], 
                               width, label='Max Variance', alpha=0.8, color='lightsteelblue')
                ax4.set_xticks(x_pos4)
                ax4.set_xticklabels(minmax_data['method_clean'], rotation=45, ha='right')
                ax4.set_ylabel('Variance Value')
                ax4.set_title('Min vs Max Dimension Variance')
                ax4.legend()
                ax4.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'variance_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_variance_heatmap(self, data: pd.DataFrame):
        """Create heatmap showing per-dimension variance across methods."""
        # Collect variance per dimension data
        variance_matrix = []
        method_names = []
        max_dims = 0
        
        for _, row in data.iterrows():
            if 'variance_per_dim' in row and row['variance_per_dim'] and len(row['variance_per_dim']) > 0:
                variance_matrix.append(row['variance_per_dim'])
                method_names.append(row['method_clean'])
                max_dims = max(max_dims, len(row['variance_per_dim']))
        
        if variance_matrix and max_dims > 1:
            # Pad shorter variance vectors with NaN
            variance_matrix_padded = []
            for var_vec in variance_matrix:
                padded = var_vec + [np.nan] * (max_dims - len(var_vec))
                variance_matrix_padded.append(padded)
            
            variance_array = np.array(variance_matrix_padded)
            
            fig, ax = plt.subplots(figsize=(max(8, max_dims), max(6, len(method_names)*0.8)))
            
            # Create heatmap
            im = ax.imshow(variance_array, cmap='YlOrRd', aspect='auto', 
                          interpolation='nearest')
            
            # Set ticks and labels
            ax.set_xticks(range(max_dims))
            ax.set_xticklabels([f'Dim {i+1}' for i in range(max_dims)])
            ax.set_yticks(range(len(method_names)))
            ax.set_yticklabels(method_names)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Variance', rotation=270, labelpad=20)
            
            # Add text annotations
            for i in range(len(method_names)):
                for j in range(max_dims):
                    if not np.isnan(variance_array[i, j]):
                        text = ax.text(j, i, f'{variance_array[i, j]:.2f}',
                                     ha='center', va='center', fontsize=8,
                                     color='white' if variance_array[i, j] > np.nanmean(variance_array) else 'black')
            
            ax.set_title('Per-Dimension Variance Heatmap', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Latent Dimensions')
            ax.set_ylabel('Methods')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'variance_per_dimension_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_variance_entropy_plot(self, data: pd.DataFrame):
        """Create plot showing variance entropy and effective dimensionality."""
        entropy_data = data.dropna(subset=['variance_entropy', 'effective_dim_variance'])
        
        if not entropy_data.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            methods = entropy_data['method_clean']
            x_pos = range(len(methods))
            
            # Plot 1: Variance Entropy
            bars1 = ax1.bar(x_pos, entropy_data['variance_entropy'], 
                           alpha=0.8, color='skyblue')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(methods, rotation=45, ha='right')
            ax1.set_ylabel('Variance Entropy')
            ax1.set_title('Variance Distribution Entropy\\n(Higher = More Uniform)')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars1, entropy_data['variance_entropy']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Effective Dimensionality vs Actual Dimensionality
            actual_dims = entropy_data['dim']
            effective_dims = entropy_data['effective_dim_variance']
            
            # Scatter plot with efficiency line
            scatter = ax2.scatter(actual_dims, effective_dims, 
                                s=150, alpha=0.7, c=range(len(entropy_data)), 
                                cmap='viridis', edgecolors='black')
            
            # Add diagonal line for reference (perfect efficiency)
            max_dim = max(actual_dims.max(), effective_dims.max())
            ax2.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, 
                   label='Perfect Efficiency')
            
            # Add method labels
            for i, (actual, effective, method) in enumerate(zip(actual_dims, effective_dims, methods)):
                ax2.annotate(method, (actual, effective), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)
            
            ax2.set_xlabel('Actual Dimensions')
            ax2.set_ylabel('Effective Dimensions (Variance-based)')
            ax2.set_title('Dimensional Efficiency\\n(Closer to diagonal = More efficient)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'variance_entropy_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_mi_per_dimension_plots(self, data: pd.DataFrame):
        """Create per-dimension mutual information analysis plots."""
        # Create MI heatmaps for each task
        for task in ['gender', 'abnormal', 'age']:
            self._create_mi_heatmap(data, task)
        
        # Create MI concentration analysis
        self._create_mi_concentration_plot(data)
        
        # Create dimension ranking by informativeness
        self._create_dimension_ranking_plot(data)
    
    def _create_mi_heatmap(self, data: pd.DataFrame, task: str):
        """Create heatmap showing per-dimension MI for a specific task."""
        mi_col = f'mi_{task}_per_dim'
        
        # Collect MI per dimension data
        mi_matrix = []
        method_names = []
        max_dims = 0
        
        for _, row in data.iterrows():
            if mi_col in row and row[mi_col] and len(row[mi_col]) > 0:
                mi_matrix.append(row[mi_col])
                method_names.append(row['method_clean'])
                max_dims = max(max_dims, len(row[mi_col]))
        
        if mi_matrix and max_dims > 1:
            # Pad shorter MI vectors with NaN
            mi_matrix_padded = []
            for mi_vec in mi_matrix:
                padded = mi_vec + [np.nan] * (max_dims - len(mi_vec))
                mi_matrix_padded.append(padded)
            
            mi_array = np.array(mi_matrix_padded)
            
            fig, ax = plt.subplots(figsize=(max(8, max_dims), max(6, len(method_names)*0.8)))
            
            # Create heatmap with diverging colormap
            im = ax.imshow(mi_array, cmap='Reds', aspect='auto', 
                          interpolation='nearest')
            
            # Set ticks and labels
            ax.set_xticks(range(max_dims))
            ax.set_xticklabels([f'Dim {i+1}' for i in range(max_dims)])
            ax.set_yticks(range(len(method_names)))
            ax.set_yticklabels(method_names)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Mutual Information', rotation=270, labelpad=20)
            
            # Add text annotations for non-NaN values
            for i in range(len(method_names)):
                for j in range(max_dims):
                    if not np.isnan(mi_array[i, j]):
                        text = ax.text(j, i, f'{mi_array[i, j]:.3f}',
                                     ha='center', va='center', fontsize=8,
                                     color='white' if mi_array[i, j] > np.nanmean(mi_array) else 'black')
            
            ax.set_title(f'Per-Dimension Mutual Information: {task.title()} Task', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Latent Dimensions')
            ax.set_ylabel('Methods')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'mi_per_dimension_{task}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_mi_concentration_plot(self, data: pd.DataFrame):
        """Create plot showing MI concentration across tasks."""
        tasks = ['gender', 'abnormal']
        concentration_data = []
        
        for _, row in data.iterrows():
            row_data = {'method': row['method_clean']}
            for task in tasks:
                conc_col = f'mi_{task}_concentration'
                if conc_col in row and not pd.isna(row[conc_col]):
                    row_data[f'{task}_concentration'] = row[conc_col]
            
            if len(row_data) > 1:  # Has at least one concentration value
                concentration_data.append(row_data)
        
        if concentration_data:
            conc_df = pd.DataFrame(concentration_data)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x_pos = np.arange(len(conc_df))
            width = 0.35
            
            # Plot concentrations for each task
            colors = ['lightblue', 'lightcoral']
            bars = []
            for i, task in enumerate(tasks):
                conc_col = f'{task}_concentration'
                if conc_col in conc_df.columns:
                    task_data = conc_df.dropna(subset=[conc_col])
                    if not task_data.empty:
                        task_bars = ax.bar(x_pos[task_data.index] + i*width, 
                                         task_data[conc_col], width, 
                                         label=f'{task.title()} Task', 
                                         alpha=0.8, color=colors[i])
                        bars.extend(task_bars)
            
            ax.set_xticks(x_pos + width/2)
            ax.set_xticklabels(conc_df['method'], rotation=45, ha='right')
            ax.set_ylabel('MI Concentration (Top 20% Dimensions)')
            ax.set_title('Mutual Information Concentration Analysis', 
                        fontsize=16, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'mi_concentration_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_dimension_ranking_plot(self, data: pd.DataFrame):
        """Create dimension ranking plot by informativeness."""
        # Find method with highest average MI for detailed analysis
        best_method_data = None
        best_avg_mi = 0
        
        for _, row in data.iterrows():
            avg_mi = 0
            count = 0
            for task in ['gender', 'abnormal']:
                mi_mean_col = f'mi_{task}_mean'
                if mi_mean_col in row and not pd.isna(row[mi_mean_col]):
                    avg_mi += row[mi_mean_col]
                    count += 1
            
            if count > 0:
                avg_mi /= count
                if avg_mi > best_avg_mi:
                    best_avg_mi = avg_mi
                    best_method_data = row
        
        if best_method_data is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: Dimension ranking for best method
            method_name = best_method_data['method_clean']
            
            dimensions = []
            mi_values = []
            variance_values = []
            
            # Collect per-dimension data
            for task in ['gender', 'abnormal']:
                mi_per_dim_col = f'mi_{task}_per_dim'
                if (mi_per_dim_col in best_method_data and 
                    best_method_data[mi_per_dim_col] and
                    'variance_per_dim' in best_method_data and
                    best_method_data['variance_per_dim']):
                    
                    mi_per_dim = best_method_data[mi_per_dim_col]
                    var_per_dim = best_method_data['variance_per_dim']
                    
                    for i, (mi_val, var_val) in enumerate(zip(mi_per_dim, var_per_dim)):
                        dimensions.append(f'Dim {i+1}')
                        mi_values.append(mi_val)
                        variance_values.append(var_val)
                    break  # Use first available task data
            
            if dimensions:
                # Sort by MI values
                sorted_data = sorted(zip(dimensions, mi_values, variance_values), 
                                   key=lambda x: x[1], reverse=True)
                dims_sorted, mi_sorted, var_sorted = zip(*sorted_data)
                
                # Create ranking plot
                y_pos = range(len(dims_sorted))
                bars1 = ax1.barh(y_pos, mi_sorted, alpha=0.8, color='skyblue')
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(dims_sorted)
                ax1.set_xlabel('Mutual Information')
                ax1.set_title(f'Dimension Ranking by MI\\n({method_name})')
                ax1.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
                for i, (bar, val) in enumerate(zip(bars1, mi_sorted)):
                    ax1.text(val + max(mi_sorted)*0.01, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', ha='left', va='center', fontsize=8)
            
            # Plot 2: High information dimensions across all methods
            high_info_counts = []
            method_names = []
            
            for _, row in data.iterrows():
                total_high_info = 0
                for task in ['gender', 'abnormal']:
                    high_info_col = f'mi_{task}_high_info_dims'
                    if high_info_col in row and not pd.isna(row[high_info_col]):
                        total_high_info += row[high_info_col]
                
                if total_high_info > 0:
                    high_info_counts.append(total_high_info)
                    method_names.append(row['method_clean'])
            
            if high_info_counts:
                bars2 = ax2.bar(range(len(method_names)), high_info_counts, 
                              alpha=0.8, color='lightcoral')
                ax2.set_xticks(range(len(method_names)))
                ax2.set_xticklabels(method_names, rotation=45, ha='right')
                ax2.set_ylabel('High Information Dimensions')
                ax2.set_title('Count of Highly Informative Dimensions')
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, val in zip(bars2, high_info_counts):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{int(val)}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'dimension_ranking_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_dimension_efficiency_plots(self, data: pd.DataFrame):
        """Create comprehensive dimension efficiency analysis plots."""
        # Plot 1: Efficiency vs Performance
        self._create_efficiency_performance_plot(data)
        
        # Plot 2: Multi-dimensional efficiency analysis
        self._create_multidim_efficiency_plot(data)
    
    def _create_efficiency_performance_plot(self, data: pd.DataFrame):
        """Create scatter plot showing efficiency vs classification performance."""
        # Collect efficiency and performance data
        efficiency_data = []
        
        for _, row in data.iterrows():
            if ('active_units' in row and 'dim' in row and 
                not pd.isna(row['active_units']) and not pd.isna(row['dim']) and
                row['dim'] > 0):
                
                row_data = {
                    'method': row['method_clean'],
                    'efficiency': row['active_units'] / row['dim'],
                    'active_units': row['active_units'],
                    'total_dims': row['dim']
                }
                
                # Add MI data
                for task in ['gender', 'abnormal']:
                    mi_col = f'mi_{task}_mean'
                    if mi_col in row and not pd.isna(row[mi_col]):
                        row_data[f'mi_{task}'] = row[mi_col]
                
                efficiency_data.append(row_data)
        
        if efficiency_data:
            eff_df = pd.DataFrame(efficiency_data)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Dimension Efficiency Analysis', fontsize=18, fontweight='bold')
            
            # Plot 1: Efficiency vs Gender MI
            if 'mi_gender' in eff_df.columns:
                gender_data = eff_df.dropna(subset=['efficiency', 'mi_gender'])
                if not gender_data.empty:
                    scatter1 = ax1.scatter(gender_data['efficiency'], gender_data['mi_gender'],
                                         s=gender_data['active_units']*10, alpha=0.7,
                                         c=range(len(gender_data)), cmap='viridis',
                                         edgecolors='black')
                    
                    for i, (eff, mi, method) in enumerate(zip(gender_data['efficiency'], 
                                                            gender_data['mi_gender'],
                                                            gender_data['method'])):
                        ax1.annotate(method, (eff, mi), xytext=(5, 5),
                                   textcoords='offset points', fontsize=8)
                    
                    ax1.set_xlabel('Dimensional Efficiency')
                    ax1.set_ylabel('Gender MI')
                    ax1.set_title('Efficiency vs Gender Informativeness\\n(Bubble size = Active Units)')
                    ax1.grid(True, alpha=0.3)
            
            # Plot 2: Efficiency vs Abnormal MI
            if 'mi_abnormal' in eff_df.columns:
                abnormal_data = eff_df.dropna(subset=['efficiency', 'mi_abnormal'])
                if not abnormal_data.empty:
                    scatter2 = ax2.scatter(abnormal_data['efficiency'], abnormal_data['mi_abnormal'],
                                         s=abnormal_data['active_units']*10, alpha=0.7,
                                         c=range(len(abnormal_data)), cmap='plasma',
                                         edgecolors='black')
                    
                    for i, (eff, mi, method) in enumerate(zip(abnormal_data['efficiency'], 
                                                            abnormal_data['mi_abnormal'],
                                                            abnormal_data['method'])):
                        ax2.annotate(method, (eff, mi), xytext=(5, 5),
                                   textcoords='offset points', fontsize=8)
                    
                    ax2.set_xlabel('Dimensional Efficiency')
                    ax2.set_ylabel('Abnormal MI')
                    ax2.set_title('Efficiency vs Abnormal Informativeness\\n(Bubble size = Active Units)')
                    ax2.grid(True, alpha=0.3)
            
            # Plot 3: Active units vs Total dimensions
            scatter3 = ax3.scatter(eff_df['total_dims'], eff_df['active_units'],
                                 s=150, alpha=0.7, c=eff_df['efficiency'],
                                 cmap='RdYlGn', edgecolors='black')
            
            # Add diagonal line
            max_dim = max(eff_df['total_dims'].max(), eff_df['active_units'].max())
            ax3.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='Perfect Efficiency')
            
            for i, (total, active, method) in enumerate(zip(eff_df['total_dims'], 
                                                          eff_df['active_units'],
                                                          eff_df['method'])):
                ax3.annotate(method, (total, active), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)
            
            ax3.set_xlabel('Total Dimensions')
            ax3.set_ylabel('Active Units')
            ax3.set_title('Active vs Total Dimensions\\n(Color = Efficiency)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add colorbar for efficiency
            cbar3 = plt.colorbar(scatter3, ax=ax3)
            cbar3.set_label('Efficiency', rotation=270, labelpad=20)
            
            # Plot 4: Efficiency distribution
            bars4 = ax4.bar(range(len(eff_df)), eff_df['efficiency'], 
                          alpha=0.8, color='lightsteelblue')
            ax4.set_xticks(range(len(eff_df)))
            ax4.set_xticklabels(eff_df['method'], rotation=45, ha='right')
            ax4.set_ylabel('Dimensional Efficiency')
            ax4.set_title('Efficiency Comparison Across Methods')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars4, eff_df['efficiency']):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'dimension_efficiency_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_variance_mi_correlation_plots(self, data: pd.DataFrame):
        """Create plots showing correlation between variance and mutual information."""
        # Analyze correlation between variance and MI per dimension
        correlation_data = []
        
        for _, row in data.iterrows():
            if ('variance_per_dim' in row and row['variance_per_dim'] and
                len(row['variance_per_dim']) > 0):
                
                variance_vec = np.array(row['variance_per_dim'])
                method_name = row['method_clean']
                
                for task in ['gender', 'abnormal']:
                    mi_col = f'mi_{task}_per_dim'
                    if (mi_col in row and row[mi_col] and 
                        len(row[mi_col]) == len(variance_vec)):
                        
                        mi_vec = np.array(row[mi_col])
                        
                        # Calculate correlation
                        if len(variance_vec) > 1 and np.std(variance_vec) > 0 and np.std(mi_vec) > 0:
                            correlation = np.corrcoef(variance_vec, mi_vec)[0, 1]
                            
                            correlation_data.append({
                                'method': method_name,
                                'task': task,
                                'correlation': correlation,
                                'variance_vec': variance_vec,
                                'mi_vec': mi_vec
                            })
        
        if correlation_data:
            # Plot 1: Correlation heatmap
            corr_df = pd.DataFrame(correlation_data)
            
            # Create pivot table for heatmap
            pivot_corr = corr_df.pivot(index='method', columns='task', values='correlation')
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Correlation heatmap
            if not pivot_corr.empty:
                sns.heatmap(pivot_corr, annot=True, cmap='RdBu_r', center=0,
                           square=True, fmt='.2f', cbar_kws={"shrink": .8},
                           linewidths=0.5, ax=ax1)
                ax1.set_title('Variance-MI Correlation by Method and Task', 
                            fontsize=14, fontweight='bold')
                ax1.set_xlabel('Task')
                ax1.set_ylabel('Method')
            
            # Scatter plot example (best correlated method)
            if 'correlation' in corr_df.columns:
                best_corr_idx = corr_df['correlation'].abs().idxmax()
                best_data = corr_df.loc[best_corr_idx]
                
                scatter = ax2.scatter(best_data['variance_vec'], best_data['mi_vec'],
                                    alpha=0.7, s=100, color='darkblue')
                
                # Add trend line
                z = np.polyfit(best_data['variance_vec'], best_data['mi_vec'], 1)
                p = np.poly1d(z)
                ax2.plot(best_data['variance_vec'], p(best_data['variance_vec']), 
                        "r--", alpha=0.8, linewidth=2)
                
                ax2.set_xlabel('Variance per Dimension')
                ax2.set_ylabel('MI per Dimension')
                ax2.set_title(f'Variance vs MI Example\\n{best_data["method"]} - {best_data["task"].title()} Task\\n(r = {best_data["correlation"]:.3f})')
                ax2.grid(True, alpha=0.3)
                
                # Annotate dimensions
                for i, (var, mi) in enumerate(zip(best_data['variance_vec'], best_data['mi_vec'])):
                    ax2.annotate(f'D{i+1}', (var, mi), xytext=(3, 3),
                               textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'variance_mi_correlation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_multidim_efficiency_plot(self, data: pd.DataFrame):
        """Create multi-dimensional efficiency analysis plot."""
        # Comprehensive efficiency metrics
        efficiency_metrics = []
        
        for _, row in data.iterrows():
            if ('active_units' in row and 'dim' in row and 'variance_entropy' in row and
                not pd.isna(row['active_units']) and not pd.isna(row['dim']) and 
                not pd.isna(row['variance_entropy']) and row['dim'] > 0):
                
                metrics = {
                    'method': row['method_clean'],
                    'dimensional_efficiency': row['active_units'] / row['dim'],
                    'variance_entropy': row['variance_entropy'],
                    'active_units': row['active_units'],
                    'total_dims': row['dim']
                }
                
                # Add variance concentration if available
                if 'variance_concentration' in row and not pd.isna(row['variance_concentration']):
                    metrics['variance_concentration'] = row['variance_concentration']
                
                # Add MI concentration if available
                mi_concentration_sum = 0
                mi_count = 0
                for task in ['gender', 'abnormal']:
                    conc_col = f'mi_{task}_concentration'
                    if conc_col in row and not pd.isna(row[conc_col]):
                        mi_concentration_sum += row[conc_col]
                        mi_count += 1
                
                if mi_count > 0:
                    metrics['mi_concentration'] = mi_concentration_sum / mi_count
                
                efficiency_metrics.append(metrics)
        
        if efficiency_metrics:
            eff_df = pd.DataFrame(efficiency_metrics)
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Create bubble plot with multiple dimensions
            x = eff_df['dimensional_efficiency']
            y = eff_df['variance_entropy']
            
            # Size based on total dimensions
            sizes = eff_df['total_dims'] * 20
            
            # Color based on MI concentration if available
            if 'mi_concentration' in eff_df.columns:
                colors = eff_df['mi_concentration']
                colormap = 'viridis'
                color_label = 'MI Concentration'
            else:
                colors = range(len(eff_df))
                colormap = 'Set3'
                color_label = 'Method Index'
            
            scatter = ax.scatter(x, y, s=sizes, c=colors, alpha=0.7, 
                               cmap=colormap, edgecolors='black', linewidth=1)
            
            # Add method labels
            for i, (eff, entropy, method) in enumerate(zip(x, y, eff_df['method'])):
                ax.annotate(method, (eff, entropy), xytext=(5, 5),
                           textcoords='offset points', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Dimensional Efficiency (Active/Total)', fontsize=12)
            ax.set_ylabel('Variance Entropy (Uniformity)', fontsize=12)
            ax.set_title('Multi-Dimensional Efficiency Analysis\\n(Bubble size = Total Dimensions)', 
                        fontsize=16, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_label, rotation=270, labelpad=20)
            
            # Add grid and reference lines
            ax.grid(True, alpha=0.3)
            
            # Add quadrant labels for interpretation
            ax.axhline(y=np.median(y), color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=np.median(x), color='red', linestyle='--', alpha=0.5)
            
            # Add quadrant annotations
            ax.text(0.95, 0.95, 'High Efficiency\\nHigh Uniformity', transform=ax.transAxes,
                   ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            ax.text(0.05, 0.05, 'Low Efficiency\\nLow Uniformity', transform=ax.transAxes,
                   ha='left', va='bottom', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'multidimensional_efficiency_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_clustering_quality_analysis(self, data: pd.DataFrame):
        """Create comprehensive clustering quality analysis with all three metrics."""
        # Collect clustering quality data
        clustering_data = []
        
        for _, row in data.iterrows():
            if ('silhouette' in row and 'davies_bouldin' in row and 'calinski_harabasz' in row and
                not pd.isna(row['silhouette']) and not pd.isna(row['davies_bouldin']) and 
                not pd.isna(row['calinski_harabasz'])):
                
                clustering_data.append({
                    'method': row['method_clean'],
                    'silhouette': row['silhouette'],
                    'davies_bouldin': row['davies_bouldin'],
                    'calinski_harabasz': row['calinski_harabasz']
                })
        
        if clustering_data:
            cluster_df = pd.DataFrame(clustering_data)
            
            # Create comprehensive clustering analysis plot
            fig = plt.figure(figsize=(20, 12))
            
            # Create subplot layout: 2x3 grid
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # Plot 1: Silhouette Score (Higher Better)
            ax1 = fig.add_subplot(gs[0, 0])
            sorted_silhouette = cluster_df.sort_values('silhouette', ascending=True)
            colors1 = plt.cm.RdYlGn(sorted_silhouette['silhouette'] / sorted_silhouette['silhouette'].max())
            bars1 = ax1.barh(range(len(sorted_silhouette)), sorted_silhouette['silhouette'], 
                           color=colors1, alpha=0.8)
            ax1.set_yticks(range(len(sorted_silhouette)))
            ax1.set_yticklabels(sorted_silhouette['method'])
            ax1.set_xlabel('Silhouette Score')
            ax1.set_title('Clustering Quality: Silhouette Score\\n(Higher = Better Separated Clusters)', 
                         fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars1, sorted_silhouette['silhouette'])):
                ax1.text(val + max(sorted_silhouette['silhouette'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Plot 2: Davies-Bouldin Index (Lower Better)
            ax2 = fig.add_subplot(gs[0, 1])
            sorted_davies = cluster_df.sort_values('davies_bouldin', ascending=False)  # Sort desc for visual consistency
            colors2 = plt.cm.RdYlGn_r(sorted_davies['davies_bouldin'] / sorted_davies['davies_bouldin'].max())
            bars2 = ax2.barh(range(len(sorted_davies)), sorted_davies['davies_bouldin'], 
                           color=colors2, alpha=0.8)
            ax2.set_yticks(range(len(sorted_davies)))
            ax2.set_yticklabels(sorted_davies['method'])
            ax2.set_xlabel('Davies-Bouldin Index')
            ax2.set_title('Clustering Quality: Davies-Bouldin Index\\n(Lower = Better Separated Clusters)', 
                         fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars2, sorted_davies['davies_bouldin'])):
                ax2.text(val + max(sorted_davies['davies_bouldin'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Plot 3: Calinski-Harabasz Index (Higher Better)
            ax3 = fig.add_subplot(gs[0, 2])
            sorted_calinski = cluster_df.sort_values('calinski_harabasz', ascending=True)
            colors3 = plt.cm.RdYlGn(sorted_calinski['calinski_harabasz'] / sorted_calinski['calinski_harabasz'].max())
            bars3 = ax3.barh(range(len(sorted_calinski)), sorted_calinski['calinski_harabasz'], 
                           color=colors3, alpha=0.8)
            ax3.set_yticks(range(len(sorted_calinski)))
            ax3.set_yticklabels(sorted_calinski['method'])
            ax3.set_xlabel('Calinski-Harabasz Index')
            ax3.set_title('Clustering Quality: Calinski-Harabasz Index\\n(Higher = Better Separated Clusters)', 
                         fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars3, sorted_calinski['calinski_harabasz'])):
                ax3.text(val + max(sorted_calinski['calinski_harabasz'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.0f}', ha='left', va='center', fontweight='bold')
            
            # Plot 4: Clustering Quality Radar Chart
            ax4 = fig.add_subplot(gs[1, :2], projection='polar')
            
            # Normalize metrics for radar chart (0-1 scale)
            normalized_data = cluster_df.copy()
            normalized_data['silhouette_norm'] = normalized_data['silhouette'] / normalized_data['silhouette'].max()
            normalized_data['davies_bouldin_norm'] = 1 - (normalized_data['davies_bouldin'] / normalized_data['davies_bouldin'].max())  # Invert since lower is better
            normalized_data['calinski_harabasz_norm'] = normalized_data['calinski_harabasz'] / normalized_data['calinski_harabasz'].max()
            
            # Set up radar chart
            metrics_radar = ['Silhouette\\n(norm)', 'Davies-Bouldin\\n(inverted)', 'Calinski-Harabasz\\n(norm)']
            N = len(metrics_radar)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Plot each method
            colors_radar = plt.cm.Set3(np.linspace(0, 1, len(normalized_data)))
            for idx, (_, row) in enumerate(normalized_data.iterrows()):
                values = [row['silhouette_norm'], row['davies_bouldin_norm'], row['calinski_harabasz_norm']]
                values += values[:1]  # Complete the circle
                
                ax4.plot(angles, values, 'o-', linewidth=2, label=row['method'], 
                        color=colors_radar[idx], alpha=0.7)
                ax4.fill(angles, values, alpha=0.1, color=colors_radar[idx])
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics_radar)
            ax4.set_ylim(0, 1)
            ax4.set_title('Clustering Quality Radar Chart\\n(All metrics normalized 0-1, higher = better)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Overall Clustering Quality Score
            ax5 = fig.add_subplot(gs[1, 2])
            
            # Calculate composite clustering score (average of normalized metrics)
            cluster_df['composite_score'] = (
                normalized_data['silhouette_norm'] + 
                normalized_data['davies_bouldin_norm'] + 
                normalized_data['calinski_harabasz_norm']
            ) / 3
            
            sorted_composite = cluster_df.sort_values('composite_score', ascending=True)
            colors5 = plt.cm.viridis(sorted_composite['composite_score'])
            bars5 = ax5.barh(range(len(sorted_composite)), sorted_composite['composite_score'], 
                           color=colors5, alpha=0.8)
            ax5.set_yticks(range(len(sorted_composite)))
            ax5.set_yticklabels(sorted_composite['method'])
            ax5.set_xlabel('Composite Clustering Score')
            ax5.set_title('Overall Clustering Quality\\n(Average of Normalized Metrics)', 
                         fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars5, sorted_composite['composite_score'])):
                ax5.text(val + max(sorted_composite['composite_score'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            plt.suptitle('Comprehensive Clustering Quality Analysis', fontsize=20, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'clustering_quality_comprehensive.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create clustering correlation plot
            self._create_clustering_correlation_plot(cluster_df)
    
    def _create_clustering_correlation_plot(self, cluster_df: pd.DataFrame):
        """Create correlation analysis between different clustering metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Clustering Metrics Correlation Analysis', fontsize=18, fontweight='bold')
        
        # Plot 1: Silhouette vs Davies-Bouldin (should be negatively correlated)
        scatter1 = ax1.scatter(cluster_df['silhouette'], cluster_df['davies_bouldin'], 
                             alpha=0.7, s=150, c=range(len(cluster_df)), cmap='viridis',
                             edgecolors='black')
        
        # Add trend line
        z1 = np.polyfit(cluster_df['silhouette'], cluster_df['davies_bouldin'], 1)
        p1 = np.poly1d(z1)
        ax1.plot(cluster_df['silhouette'], p1(cluster_df['silhouette']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr1 = np.corrcoef(cluster_df['silhouette'], cluster_df['davies_bouldin'])[0, 1]
        ax1.set_xlabel('Silhouette Score (Higher Better)')
        ax1.set_ylabel('Davies-Bouldin Index (Lower Better)')
        ax1.set_title(f'Silhouette vs Davies-Bouldin\\nr = {corr1:.3f}')
        ax1.grid(True, alpha=0.3)
        
        # Add method labels
        for i, (sil, db, method) in enumerate(zip(cluster_df['silhouette'], 
                                                cluster_df['davies_bouldin'], 
                                                cluster_df['method'])):
            ax1.annotate(method, (sil, db), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        # Plot 2: Silhouette vs Calinski-Harabasz (should be positively correlated)
        scatter2 = ax2.scatter(cluster_df['silhouette'], cluster_df['calinski_harabasz'], 
                             alpha=0.7, s=150, c=range(len(cluster_df)), cmap='plasma',
                             edgecolors='black')
        
        # Add trend line
        z2 = np.polyfit(cluster_df['silhouette'], cluster_df['calinski_harabasz'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(cluster_df['silhouette'], p2(cluster_df['silhouette']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr2 = np.corrcoef(cluster_df['silhouette'], cluster_df['calinski_harabasz'])[0, 1]
        ax2.set_xlabel('Silhouette Score (Higher Better)')
        ax2.set_ylabel('Calinski-Harabasz Index (Higher Better)')
        ax2.set_title(f'Silhouette vs Calinski-Harabasz\\nr = {corr2:.3f}')
        ax2.grid(True, alpha=0.3)
        
        # Add method labels
        for i, (sil, ch, method) in enumerate(zip(cluster_df['silhouette'], 
                                                cluster_df['calinski_harabasz'], 
                                                cluster_df['method'])):
            ax2.annotate(method, (sil, ch), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        # Plot 3: Davies-Bouldin vs Calinski-Harabasz (should be negatively correlated)
        scatter3 = ax3.scatter(cluster_df['davies_bouldin'], cluster_df['calinski_harabasz'], 
                             alpha=0.7, s=150, c=range(len(cluster_df)), cmap='coolwarm',
                             edgecolors='black')
        
        # Add trend line
        z3 = np.polyfit(cluster_df['davies_bouldin'], cluster_df['calinski_harabasz'], 1)
        p3 = np.poly1d(z3)
        ax3.plot(cluster_df['davies_bouldin'], p3(cluster_df['davies_bouldin']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr3 = np.corrcoef(cluster_df['davies_bouldin'], cluster_df['calinski_harabasz'])[0, 1]
        ax3.set_xlabel('Davies-Bouldin Index (Lower Better)')
        ax3.set_ylabel('Calinski-Harabasz Index (Higher Better)')
        ax3.set_title(f'Davies-Bouldin vs Calinski-Harabasz\\nr = {corr3:.3f}')
        ax3.grid(True, alpha=0.3)
        
        # Add method labels
        for i, (db, ch, method) in enumerate(zip(cluster_df['davies_bouldin'], 
                                                cluster_df['calinski_harabasz'], 
                                                cluster_df['method'])):
            ax3.annotate(method, (db, ch), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        # Plot 4: Correlation matrix heatmap
        corr_matrix = cluster_df[['silhouette', 'davies_bouldin', 'calinski_harabasz']].corr()
        
        im = ax4.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        metrics_labels = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
        ax4.set_xticks(range(len(metrics_labels)))
        ax4.set_xticklabels(metrics_labels, rotation=45, ha='right')
        ax4.set_yticks(range(len(metrics_labels)))
        ax4.set_yticklabels(metrics_labels)
        
        # Add correlation values
        for i in range(len(metrics_labels)):
            for j in range(len(metrics_labels)):
                text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center', fontweight='bold',
                               color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        ax4.set_title('Clustering Metrics\\nCorrelation Matrix')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'clustering_metrics_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_geometric_properties_analysis(self, data: pd.DataFrame):
        """Create comprehensive geometric properties analysis with all three metrics."""
        # Collect geometric properties data
        geometric_data = []
        
        for _, row in data.iterrows():
            if ('trustworthiness' in row and 'continuity' in row and 'dist_corr' in row and
                not pd.isna(row['trustworthiness']) and not pd.isna(row['continuity']) and 
                not pd.isna(row['dist_corr'])):
                
                geometric_data.append({
                    'method': row['method_clean'],
                    'trustworthiness': row['trustworthiness'],
                    'continuity': row['continuity'],
                    'dist_corr': row['dist_corr']
                })
        
        if geometric_data:
            geom_df = pd.DataFrame(geometric_data)
            
            # Create comprehensive geometric analysis plot
            fig = plt.figure(figsize=(20, 12))
            
            # Create subplot layout: 2x3 grid
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # Plot 1: Trustworthiness (Higher Better)
            ax1 = fig.add_subplot(gs[0, 0])
            sorted_trust = geom_df.sort_values('trustworthiness', ascending=True)
            colors1 = plt.cm.RdYlGn(sorted_trust['trustworthiness'])
            bars1 = ax1.barh(range(len(sorted_trust)), sorted_trust['trustworthiness'], 
                           color=colors1, alpha=0.8)
            ax1.set_yticks(range(len(sorted_trust)))
            ax1.set_yticklabels(sorted_trust['method'])
            ax1.set_xlabel('Trustworthiness Score')
            ax1.set_title('Geometric Quality: Trustworthiness\\n(Higher = Better Neighborhood Preservation)', 
                         fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars1, sorted_trust['trustworthiness'])):
                ax1.text(val + max(sorted_trust['trustworthiness'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Plot 2: Continuity (Higher Better)
            ax2 = fig.add_subplot(gs[0, 1])
            sorted_cont = geom_df.sort_values('continuity', ascending=True)
            colors2 = plt.cm.RdYlGn(sorted_cont['continuity'])
            bars2 = ax2.barh(range(len(sorted_cont)), sorted_cont['continuity'], 
                           color=colors2, alpha=0.8)
            ax2.set_yticks(range(len(sorted_cont)))
            ax2.set_yticklabels(sorted_cont['method'])
            ax2.set_xlabel('Continuity Score')
            ax2.set_title('Geometric Quality: Continuity\\n(Higher = Better Smoothness)', 
                         fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars2, sorted_cont['continuity'])):
                ax2.text(val + max(sorted_cont['continuity'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Plot 3: Distance Correlation (Higher Better)
            ax3 = fig.add_subplot(gs[0, 2])
            sorted_dist = geom_df.sort_values('dist_corr', ascending=True)
            colors3 = plt.cm.RdYlGn(sorted_dist['dist_corr'])
            bars3 = ax3.barh(range(len(sorted_dist)), sorted_dist['dist_corr'], 
                           color=colors3, alpha=0.8)
            ax3.set_yticks(range(len(sorted_dist)))
            ax3.set_yticklabels(sorted_dist['method'])
            ax3.set_xlabel('Distance Correlation')
            ax3.set_title('Geometric Quality: Distance Correlation\\n(Higher = Better Distance Preservation)', 
                         fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars3, sorted_dist['dist_corr'])):
                ax3.text(val + max(sorted_dist['dist_corr'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Plot 4: Geometric Quality Radar Chart
            ax4 = fig.add_subplot(gs[1, :2], projection='polar')
            
            # Set up radar chart
            metrics_radar = ['Trustworthiness', 'Continuity', 'Distance Correlation']
            N = len(metrics_radar)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Plot each method
            colors_radar = plt.cm.Set3(np.linspace(0, 1, len(geom_df)))
            for idx, (_, row) in enumerate(geom_df.iterrows()):
                values = [row['trustworthiness'], row['continuity'], row['dist_corr']]
                values += values[:1]  # Complete the circle
                
                ax4.plot(angles, values, 'o-', linewidth=2, label=row['method'], 
                        color=colors_radar[idx], alpha=0.7)
                ax4.fill(angles, values, alpha=0.1, color=colors_radar[idx])
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics_radar)
            ax4.set_ylim(0, 1)
            ax4.set_title('Geometric Properties Radar Chart\\n(All metrics 0-1, higher = better)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Overall Geometric Quality Score
            ax5 = fig.add_subplot(gs[1, 2])
            
            # Calculate composite geometric score (average of all metrics)
            geom_df['composite_geometric_score'] = (
                geom_df['trustworthiness'] + 
                geom_df['continuity'] + 
                geom_df['dist_corr']
            ) / 3
            
            sorted_composite = geom_df.sort_values('composite_geometric_score', ascending=True)
            colors5 = plt.cm.viridis(sorted_composite['composite_geometric_score'])
            bars5 = ax5.barh(range(len(sorted_composite)), sorted_composite['composite_geometric_score'], 
                           color=colors5, alpha=0.8)
            ax5.set_yticks(range(len(sorted_composite)))
            ax5.set_yticklabels(sorted_composite['method'])
            ax5.set_xlabel('Composite Geometric Score')
            ax5.set_title('Overall Geometric Quality\\n(Average of All Metrics)', 
                         fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars5, sorted_composite['composite_geometric_score'])):
                ax5.text(val + max(sorted_composite['composite_geometric_score'])*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            plt.suptitle('Comprehensive Geometric Properties Analysis', fontsize=20, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'geometric_properties_comprehensive.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create geometric correlation plot
            self._create_geometric_correlation_plot(geom_df)
    
    def _create_geometric_correlation_plot(self, geom_df: pd.DataFrame):
        """Create correlation analysis between different geometric metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Geometric Properties Correlation Analysis', fontsize=18, fontweight='bold')
        
        # Plot 1: Trustworthiness vs Continuity (should be positively correlated)
        scatter1 = ax1.scatter(geom_df['trustworthiness'], geom_df['continuity'], 
                             alpha=0.7, s=150, c=range(len(geom_df)), cmap='viridis',
                             edgecolors='black')
        
        # Add trend line
        z1 = np.polyfit(geom_df['trustworthiness'], geom_df['continuity'], 1)
        p1 = np.poly1d(z1)
        ax1.plot(geom_df['trustworthiness'], p1(geom_df['trustworthiness']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr1 = np.corrcoef(geom_df['trustworthiness'], geom_df['continuity'])[0, 1]
        ax1.set_xlabel('Trustworthiness (Neighborhood Preservation)')
        ax1.set_ylabel('Continuity (Smoothness)')
        ax1.set_title(f'Trustworthiness vs Continuity\\nr = {corr1:.3f}')
        ax1.grid(True, alpha=0.3)
        
        # Add method labels
        for i, (trust, cont, method) in enumerate(zip(geom_df['trustworthiness'], 
                                                    geom_df['continuity'], 
                                                    geom_df['method'])):
            ax1.annotate(method, (trust, cont), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        # Plot 2: Trustworthiness vs Distance Correlation
        scatter2 = ax2.scatter(geom_df['trustworthiness'], geom_df['dist_corr'], 
                             alpha=0.7, s=150, c=range(len(geom_df)), cmap='plasma',
                             edgecolors='black')
        
        # Add trend line
        z2 = np.polyfit(geom_df['trustworthiness'], geom_df['dist_corr'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(geom_df['trustworthiness'], p2(geom_df['trustworthiness']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr2 = np.corrcoef(geom_df['trustworthiness'], geom_df['dist_corr'])[0, 1]
        ax2.set_xlabel('Trustworthiness (Neighborhood Preservation)')
        ax2.set_ylabel('Distance Correlation')
        ax2.set_title(f'Trustworthiness vs Distance Correlation\\nr = {corr2:.3f}')
        ax2.grid(True, alpha=0.3)
        
        # Add method labels
        for i, (trust, dist, method) in enumerate(zip(geom_df['trustworthiness'], 
                                                    geom_df['dist_corr'], 
                                                    geom_df['method'])):
            ax2.annotate(method, (trust, dist), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        # Plot 3: Continuity vs Distance Correlation
        scatter3 = ax3.scatter(geom_df['continuity'], geom_df['dist_corr'], 
                             alpha=0.7, s=150, c=range(len(geom_df)), cmap='coolwarm',
                             edgecolors='black')
        
        # Add trend line
        z3 = np.polyfit(geom_df['continuity'], geom_df['dist_corr'], 1)
        p3 = np.poly1d(z3)
        ax3.plot(geom_df['continuity'], p3(geom_df['continuity']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr3 = np.corrcoef(geom_df['continuity'], geom_df['dist_corr'])[0, 1]
        ax3.set_xlabel('Continuity (Smoothness)')
        ax3.set_ylabel('Distance Correlation')
        ax3.set_title(f'Continuity vs Distance Correlation\\nr = {corr3:.3f}')
        ax3.grid(True, alpha=0.3)
        
        # Add method labels
        for i, (cont, dist, method) in enumerate(zip(geom_df['continuity'], 
                                                   geom_df['dist_corr'], 
                                                   geom_df['method'])):
            ax3.annotate(method, (cont, dist), xytext=(5, 5),
                       textcoords='offset points', fontsize=8)
        
        # Plot 4: Correlation matrix heatmap
        corr_matrix = geom_df[['trustworthiness', 'continuity', 'dist_corr']].corr()
        
        im = ax4.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        metrics_labels = ['Trustworthiness', 'Continuity', 'Distance Corr']
        ax4.set_xticks(range(len(metrics_labels)))
        ax4.set_xticklabels(metrics_labels, rotation=45, ha='right')
        ax4.set_yticks(range(len(metrics_labels)))
        ax4.set_yticklabels(metrics_labels)
        
        # Add correlation values
        for i in range(len(metrics_labels)):
            for j in range(len(metrics_labels)):
                text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center', fontweight='bold',
                               color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        ax4.set_title('Geometric Metrics\\nCorrelation Matrix')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'geometric_properties_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_information_content_analysis(self, data: pd.DataFrame):
        """Create enhanced information content analysis combining HSIC and MI metrics."""
        # Collect information content data
        info_data = []
        
        for _, row in data.iterrows():
            row_data = {'method': row['method_clean']}
            
            # HSIC (independence measure - lower is better)
            if 'hsic_global_score' in row and not pd.isna(row['hsic_global_score']):
                row_data['hsic_global_score'] = row['hsic_global_score']
                row_data['independence_score'] = 1 - row['hsic_global_score']  # Convert to higher-is-better
            
            # MI measures (informativeness - higher is better)
            mi_scores = []
            for task in ['gender', 'abnormal', 'age']:
                mi_col = f'mi_{task}_mean'
                if mi_col in row and not pd.isna(row[mi_col]):
                    row_data[f'mi_{task}'] = row[mi_col]
                    mi_scores.append(row[mi_col])
            
            if mi_scores:
                row_data['mi_average'] = np.mean(mi_scores)
                row_data['mi_total'] = np.sum(mi_scores)
            
            # Information efficiency (high MI with high independence)
            if 'independence_score' in row_data and 'mi_average' in row_data:
                row_data['info_efficiency'] = row_data['independence_score'] * row_data['mi_average']
            
            if len(row_data) > 1:  # Has at least one metric
                info_data.append(row_data)
        
        if info_data:
            info_df = pd.DataFrame(info_data)
            
            # Create comprehensive information analysis plot
            fig = plt.figure(figsize=(20, 14))
            
            # Create subplot layout: 3x3 grid
            gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
            
            # Plot 1: HSIC Independence Score
            if 'independence_score' in info_df.columns:
                ax1 = fig.add_subplot(gs[0, 0])
                indep_data = info_df.dropna(subset=['independence_score'])
                if not indep_data.empty:
                    sorted_indep = indep_data.sort_values('independence_score', ascending=True)
                    colors1 = plt.cm.RdYlGn(sorted_indep['independence_score'])
                    bars1 = ax1.barh(range(len(sorted_indep)), sorted_indep['independence_score'], 
                                   color=colors1, alpha=0.8)
                    ax1.set_yticks(range(len(sorted_indep)))
                    ax1.set_yticklabels(sorted_indep['method'])
                    ax1.set_xlabel('Independence Score (1 - HSIC)')
                    ax1.set_title('Feature Independence\\n(Higher = More Independent)', fontweight='bold')
                    ax1.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels
                    for i, (bar, val) in enumerate(zip(bars1, sorted_indep['independence_score'])):
                        ax1.text(val + max(sorted_indep['independence_score'])*0.01, 
                                bar.get_y() + bar.get_height()/2,
                                f'{val:.3f}', ha='left', va='center', fontweight='bold')
            
            # Plot 2: Average MI Informativeness
            if 'mi_average' in info_df.columns:
                ax2 = fig.add_subplot(gs[0, 1])
                mi_data = info_df.dropna(subset=['mi_average'])
                if not mi_data.empty:
                    sorted_mi = mi_data.sort_values('mi_average', ascending=True)
                    colors2 = plt.cm.plasma(sorted_mi['mi_average'] / sorted_mi['mi_average'].max())
                    bars2 = ax2.barh(range(len(sorted_mi)), sorted_mi['mi_average'], 
                                   color=colors2, alpha=0.8)
                    ax2.set_yticks(range(len(sorted_mi)))
                    ax2.set_yticklabels(sorted_mi['method'])
                    ax2.set_xlabel('Average MI Score')
                    ax2.set_title('Feature Informativeness\\n(Higher = More Informative)', fontweight='bold')
                    ax2.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels
                    for i, (bar, val) in enumerate(zip(bars2, sorted_mi['mi_average'])):
                        ax2.text(val + max(sorted_mi['mi_average'])*0.01, 
                                bar.get_y() + bar.get_height()/2,
                                f'{val:.4f}', ha='left', va='center', fontweight='bold')
            
            # Plot 3: Information Efficiency
            if 'info_efficiency' in info_df.columns:
                ax3 = fig.add_subplot(gs[0, 2])
                eff_data = info_df.dropna(subset=['info_efficiency'])
                if not eff_data.empty:
                    sorted_eff = eff_data.sort_values('info_efficiency', ascending=True)
                    colors3 = plt.cm.viridis(sorted_eff['info_efficiency'] / sorted_eff['info_efficiency'].max())
                    bars3 = ax3.barh(range(len(sorted_eff)), sorted_eff['info_efficiency'], 
                                   color=colors3, alpha=0.8)
                    ax3.set_yticks(range(len(sorted_eff)))
                    ax3.set_yticklabels(sorted_eff['method'])
                    ax3.set_xlabel('Information Efficiency')
                    ax3.set_title('Information Efficiency\\n(Independence × Informativeness)', fontweight='bold')
                    ax3.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels
                    for i, (bar, val) in enumerate(zip(bars3, sorted_eff['info_efficiency'])):
                        ax3.text(val + max(sorted_eff['info_efficiency'])*0.01, 
                                bar.get_y() + bar.get_height()/2,
                                f'{val:.4f}', ha='left', va='center', fontweight='bold')
            
            # Plot 4: Independence vs Informativeness Scatter
            if 'independence_score' in info_df.columns and 'mi_average' in info_df.columns:
                ax4 = fig.add_subplot(gs[1, :2])
                scatter_data = info_df.dropna(subset=['independence_score', 'mi_average'])
                if not scatter_data.empty:
                    # Color by efficiency if available
                    if 'info_efficiency' in scatter_data.columns:
                        colors_scatter = scatter_data['info_efficiency']
                        colormap = 'viridis'
                        color_label = 'Information Efficiency'
                    else:
                        colors_scatter = range(len(scatter_data))
                        colormap = 'Set3'
                        color_label = 'Method Index'
                    
                    scatter = ax4.scatter(scatter_data['independence_score'], scatter_data['mi_average'],
                                        s=200, alpha=0.8, c=colors_scatter, cmap=colormap,
                                        edgecolors='black', linewidth=1)
                    
                    # Add method labels
                    for i, (indep, mi, method) in enumerate(zip(scatter_data['independence_score'],
                                                              scatter_data['mi_average'],
                                                              scatter_data['method'])):
                        ax4.annotate(method, (indep, mi), xytext=(5, 5),
                                   textcoords='offset points', fontsize=10, fontweight='bold')
                    
                    ax4.set_xlabel('Independence Score (1 - HSIC)')
                    ax4.set_ylabel('Average MI Score')
                    ax4.set_title('Independence vs Informativeness Trade-off', fontweight='bold')
                    ax4.grid(True, alpha=0.3)
                    
                    # Add quadrant labels
                    ax4.axhline(y=np.median(scatter_data['mi_average']), color='red', linestyle='--', alpha=0.5)
                    ax4.axvline(x=np.median(scatter_data['independence_score']), color='red', linestyle='--', alpha=0.5)
                    
                    ax4.text(0.95, 0.95, 'High Independence\\nHigh Informativeness', transform=ax4.transAxes,
                           ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
                    ax4.text(0.05, 0.05, 'Low Independence\\nLow Informativeness', transform=ax4.transAxes,
                           ha='left', va='bottom', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax4)
                    cbar.set_label(color_label, rotation=270, labelpad=20)
            
            # Plot 5: Task-specific MI comparison
            ax5 = fig.add_subplot(gs[1, 2])
            tasks_with_data = []
            for task in ['gender', 'abnormal', 'age']:
                mi_col = f'mi_{task}'
                if mi_col in info_df.columns:
                    tasks_with_data.append(task)
            
            if len(tasks_with_data) >= 2:
                task_data = info_df.dropna(subset=[f'mi_{task}' for task in tasks_with_data])
                if not task_data.empty:
                    x_pos = np.arange(len(task_data))
                    width = 0.8 / len(tasks_with_data)
                    
                    colors_tasks = ['lightblue', 'lightcoral', 'lightgreen']
                    for i, task in enumerate(tasks_with_data):
                        mi_col = f'mi_{task}'
                        bars = ax5.bar(x_pos + i*width, task_data[mi_col], width,
                                     label=f'{task.title()} Task', alpha=0.8, color=colors_tasks[i])
                    
                    ax5.set_xticks(x_pos + width * (len(tasks_with_data) - 1) / 2)
                    ax5.set_xticklabels(task_data['method'], rotation=45, ha='right')
                    ax5.set_ylabel('Mutual Information')
                    ax5.set_title('Task-Specific Informativeness', fontweight='bold')
                    ax5.legend()
                    ax5.grid(True, alpha=0.3, axis='y')
            
            # Plot 6: Information content distribution
            if 'mi_total' in info_df.columns:
                ax6 = fig.add_subplot(gs[2, :])
                total_data = info_df.dropna(subset=['mi_total'])
                if not total_data.empty:
                    # Create distribution plot
                    positions = range(len(total_data))
                    
                    # Main bars
                    bars = ax6.bar(positions, total_data['mi_total'], alpha=0.7, color='steelblue')
                    
                    # Add individual task contributions if available
                    bottom_values = np.zeros(len(total_data))
                    colors_stack = ['lightblue', 'lightcoral', 'lightgreen']
                    
                    for i, task in enumerate(['gender', 'abnormal', 'age']):
                        mi_col = f'mi_{task}'
                        if mi_col in total_data.columns:
                            task_values = total_data[mi_col].fillna(0)
                            ax6.bar(positions, task_values, bottom=bottom_values,
                                  alpha=0.8, label=f'{task.title()} Task', color=colors_stack[i])
                            bottom_values += task_values
                    
                    ax6.set_xticks(positions)
                    ax6.set_xticklabels(total_data['method'], rotation=45, ha='right')
                    ax6.set_ylabel('Total Mutual Information')
                    ax6.set_title('Total Information Content by Task Contribution', fontweight='bold')
                    ax6.legend()
                    ax6.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('Comprehensive Information Content Analysis', fontsize=20, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'information_content_comprehensive.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # ========== PAIRWISE COMPARISON FUNCTIONS ==========
    
    def _center_rows(self, X: np.ndarray) -> np.ndarray:
        """Center rows for HSIC computation."""
        X = np.asarray(X, dtype=np.float64)
        return X - X.mean(axis=0, keepdims=True)

    def _hsic_linear(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute linear HSIC between two matrices."""
        Xc = self._center_rows(X)
        Yc = self._center_rows(Y)
        K = Xc @ Xc.T  # (n,n)
        L = Yc @ Yc.T  # (n,n)
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        KH = H @ K @ H
        LH = H @ L @ H
        return float(np.sum(KH * LH))

    def linear_cka(self, Z1: np.ndarray, Z2: np.ndarray) -> float:
        """Linear CKA similarity between two embeddings (rows = samples)."""
        hsic_xy = self._hsic_linear(Z1, Z2)
        hsic_xx = self._hsic_linear(Z1, Z1)
        hsic_yy = self._hsic_linear(Z2, Z2)
        denom = np.sqrt(hsic_xx * hsic_yy) + 1e-12
        val = hsic_xy / denom
        return float(np.clip(val, -1.0, 1.0))

    def rbf_cka(self, Z1: np.ndarray, Z2: np.ndarray, gamma1: Optional[float] = None, gamma2: Optional[float] = None) -> float:
        """RBF-CKA using median heuristic for bandwidth if not provided."""
        def _rbf_kernel(Z: np.ndarray, gamma: Optional[float]) -> np.ndarray:
            D2 = pairwise_distances(Z, metric="sqeuclidean")
            if gamma is None:
                nz = D2[D2 > 0]
                if nz.size == 0:
                    gamma_eff = 1.0
                else:
                    gamma_eff = 1.0 / (2.0 * np.median(nz))
            else:
                gamma_eff = gamma
            return np.exp(-gamma_eff * D2)

        K = _rbf_kernel(Z1, gamma1)
        L = _rbf_kernel(Z2, gamma2)
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        KH = H @ K @ H
        LH = H @ L @ H
        hsic_xy = float(np.sum(KH * LH))
        hsic_xx = float(np.sum(KH * KH))
        hsic_yy = float(np.sum(LH * LH))
        val = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12)
        return float(np.clip(val, -1.0, 1.0))

    def cca_maxcorr(self, Z1: np.ndarray, Z2: np.ndarray, n_components: Optional[int] = None) -> float:
        """Maximum canonical correlation between two embeddings."""
        Z1 = np.asarray(Z1, dtype=np.float64)
        Z2 = np.asarray(Z2, dtype=np.float64)
        n, d1 = Z1.shape
        _, d2 = Z2.shape
        if n < 2 or d1 == 0 or d2 == 0:
            return 0.0
        k = min(d1, d2, n - 1)
        if n_components is not None:
            k = min(k, int(n_components))
        if k < 1:
            return 0.0
        cca = CCA(n_components=k, max_iter=500)
        Xc = self._center_rows(Z1)
        Yc = self._center_rows(Z2)
        try:
            Xc, Yc = cca.fit_transform(Xc, Yc)
            corrs = []
            for i in range(Xc.shape[1]):
                x = Xc[:, i]
                y = Yc[:, i]
                if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                    corrs.append(0.0)
                else:
                    corrs.append(float(np.corrcoef(x, y)[0, 1]))
            return float(np.max(np.abs(corrs))) if corrs else 0.0
        except:
            return 0.0

    def distance_geometry_corr(self, Z1: np.ndarray, Z2: np.ndarray) -> float:
        """Correlation of pairwise distances between two spaces."""
        D1 = pairwise_distances(Z1)
        D2 = pairwise_distances(Z2)
        iu = np.triu_indices_from(D1, k=1)
        a, b = D1[iu], D2[iu]
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def knn_jaccard_overlap(self, Z1: np.ndarray, Z2: np.ndarray, k: int = 10) -> float:
        """Average Jaccard overlap of k-NN sets between two spaces."""
        n = Z1.shape[0]
        if n <= 1 or k < 1:
            return 0.0
        k = min(k, n - 1)
        D1 = pairwise_distances(Z1)
        D2 = pairwise_distances(Z2)
        np.fill_diagonal(D1, np.inf)
        np.fill_diagonal(D2, np.inf)
        idx1 = np.argsort(D1, axis=1)[:, :k]
        idx2 = np.argsort(D2, axis=1)[:, :k]
        scores = []
        for i in range(n):
            s1 = set(idx1[i].tolist())
            s2 = set(idx2[i].tolist())
            inter = len(s1 & s2)
            union = len(s1 | s2)
            scores.append(inter / union if union > 0 else 0.0)
        return float(np.mean(scores)) if scores else 0.0

    def procrustes_disparity(self, Z1: np.ndarray, Z2: np.ndarray) -> float:
        """Procrustes disparity (lower = more similar) after optimal alignment."""
        try:
            A, B = Z1, Z2
            if A.shape[1] != B.shape[1]:
                k = min(A.shape[1], B.shape[1])
                if k < 1:
                    return 0.0
                p1 = PCA(n_components=k, random_state=42)
                p2 = PCA(n_components=k, random_state=42)
                A = p1.fit_transform(A)
                B = p2.fit_transform(B)
            _, _, disparity = procrustes(A, B)
            return float(disparity)
        except Exception:
            return 0.0

    def compute_pairwise_summary(self, Z1: np.ndarray, Z2: np.ndarray, k: int = 10) -> Dict[str, float]:
        """Compute comprehensive pairwise similarity metrics."""
        return {
            "cka_linear": self.linear_cka(Z1, Z2),
            "cka_rbf": self.rbf_cka(Z1, Z2),
            "cca_maxcorr": self.cca_maxcorr(Z1, Z2),
            "dist_geom_corr": self.distance_geometry_corr(Z1, Z2),
            f"knn_jaccard_k{k}": self.knn_jaccard_overlap(Z1, Z2, k=k),
            "procrustes_disparity": self.procrustes_disparity(Z1, Z2),
        }
    
    def _load_latent_features(self, method: str, split: str = 'eval') -> Optional[np.ndarray]:
        """Load latent features for a method."""
        cache_key = f"{method}_{split}"
        if cache_key in self.latent_features_cache:
            return self.latent_features_cache[cache_key]
        
        method_dir = self.results_dir / method
        latent_file = method_dir / f"temp_latent_features_{split}.json"
        
        if not latent_file.exists():
            print(f"  ✗ No latent features file for {method} ({split})")
            return None
        
        try:
            # Simple JSON loading - assume JSONL format with latent vectors
            latent_vectors = []
            with open(latent_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if 'latent' in data:
                        latent_vectors.append(data['latent'])
                    elif 'features' in data:  # Alternative key name
                        latent_vectors.append(data['features'])
            
            if latent_vectors:
                Z = np.array(latent_vectors, dtype=np.float32)
                self.latent_features_cache[cache_key] = Z
                print(f"  ✓ Loaded {method} ({split}): {Z.shape}")
                return Z
            else:
                print(f"  ✗ No latent vectors found in {method} ({split})")
                return None
                
        except Exception as e:
            print(f"  ✗ Error loading {method} ({split}): {e}")
            return None
    
    def compute_pairwise_similarities(self, split: str = 'eval', k: int = 10) -> pd.DataFrame:
        """Compute pairwise similarities between all methods in the group."""
        print(f"\nComputing pairwise similarities for {split} split...")
        
        # Load latent features for all available methods
        method_features = {}
        for method in self.methods:
            Z = self._load_latent_features(method, split)
            if Z is not None:
                method_features[method] = Z
        
        if len(method_features) < 2:
            print(f"  ✗ Need at least 2 methods with latent features, found {len(method_features)}")
            return pd.DataFrame()
        
        print(f"  ✓ Computing similarities for {len(method_features)} methods")
        
        # Compute all pairwise comparisons
        similarity_data = []
        methods_list = list(method_features.keys())
        
        for i, method1 in enumerate(methods_list):
            for j, method2 in enumerate(methods_list):
                if i <= j:  # Include diagonal and upper triangle
                    Z1 = method_features[method1]
                    Z2 = method_features[method2]
                    
                    # Align samples (use minimum overlapping samples)
                    min_samples = min(Z1.shape[0], Z2.shape[0])
                    Z1_aligned = Z1[:min_samples]
                    Z2_aligned = Z2[:min_samples]
                    
                    if i == j:
                        # Diagonal - perfect similarity for most metrics
                        similarities = {
                            "cka_linear": 1.0,
                            "cka_rbf": 1.0, 
                            "cca_maxcorr": 1.0,
                            "dist_geom_corr": 1.0,
                            f"knn_jaccard_k{k}": 1.0,
                            "procrustes_disparity": 0.0,  # Lower is better
                        }
                    else:
                        # Compute actual similarities
                        similarities = self.compute_pairwise_summary(Z1_aligned, Z2_aligned, k=k)
                    
                    # Store results
                    for metric, value in similarities.items():
                        similarity_data.append({
                            'method1': method1,
                            'method2': method2,
                            'method1_clean': self.clean_method_name(method1),
                            'method2_clean': self.clean_method_name(method2),
                            'metric': metric,
                            'value': value,
                            'samples_used': min_samples
                        })
        
        return pd.DataFrame(similarity_data)
    
    def create_pairwise_similarity_analysis(self, split: str = 'eval'):
        """Create comprehensive pairwise similarity analysis and visualizations."""
        print("\n" + "="*60)
        print("PAIRWISE SIMILARITY ANALYSIS")
        print("="*60)
        
        similarity_df = self.compute_pairwise_similarities(split)
        if similarity_df.empty:
            print("No pairwise similarities computed - skipping analysis")
            return
        
        # Create visualizations for each metric
        metrics = similarity_df['metric'].unique()
        
        # Create comprehensive similarity matrices plot
        self._create_similarity_matrices_plot(similarity_df, metrics)
        
        # Create detailed analysis plots
        self._create_similarity_analysis_plots(similarity_df, metrics)
        
        # Save similarity matrices as CSV
        self._save_similarity_matrices(similarity_df, metrics)
        
        print("  ✓ Saved pairwise similarity analysis")
    
    def _create_similarity_matrices_plot(self, similarity_df: pd.DataFrame, metrics: List[str]):
        """Create comprehensive similarity matrices visualization."""
        # Determine grid size
        n_metrics = len(metrics)
        cols = 3
        rows = (n_metrics + cols - 1) // cols
        
        fig = plt.figure(figsize=(5*cols, 4*rows))
        
        for idx, metric in enumerate(metrics):
            ax = plt.subplot(rows, cols, idx + 1)
            
            # Create pivot table for this metric
            metric_data = similarity_df[similarity_df['metric'] == metric]
            pivot_matrix = metric_data.pivot(index='method1_clean', columns='method2_clean', values='value')
            
            # Make symmetric by filling lower triangle
            pivot_matrix = pivot_matrix.fillna(pivot_matrix.T)
            
            # Choose appropriate colormap based on metric
            if 'disparity' in metric:
                # Lower is better - use reversed colormap
                cmap = 'Reds_r'
                vmin, vmax = 0, pivot_matrix.max().max()
            else:
                # Higher is better
                cmap = 'RdYlGn'
                vmin, vmax = 0, 1
            
            # Create heatmap
            sns.heatmap(pivot_matrix, annot=True, cmap=cmap, fmt='.3f',
                       square=True, cbar_kws={"shrink": .8}, ax=ax,
                       vmin=vmin, vmax=vmax)
            
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            # Rotate labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.suptitle(f'Pairwise Similarity Matrices - {self.group_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pairwise_similarity_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_similarity_analysis_plots(self, similarity_df: pd.DataFrame, metrics: List[str]):
        """Create detailed similarity analysis plots."""
        # Plot 1: Average similarity rankings
        self._create_similarity_rankings_plot(similarity_df, metrics)
        
        # Plot 2: Similarity correlations between metrics
        self._create_similarity_correlations_plot(similarity_df, metrics)
        
        # Plot 3: Similarity distribution analysis
        self._create_similarity_distributions_plot(similarity_df, metrics)
    
    def _create_similarity_rankings_plot(self, similarity_df: pd.DataFrame, metrics: List[str]):
        """Create rankings based on average similarity to other methods."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            metric_data = similarity_df[similarity_df['metric'] == metric]
            
            # Calculate average similarity for each method (excluding self-comparison)
            avg_similarities = []
            methods = metric_data['method1_clean'].unique()
            
            for method in methods:
                # Get similarities where this method is involved (excluding self)
                method_sims = metric_data[
                    ((metric_data['method1_clean'] == method) | (metric_data['method2_clean'] == method)) &
                    (metric_data['method1_clean'] != metric_data['method2_clean'])
                ]['value'].values
                
                if len(method_sims) > 0:
                    if 'disparity' in metric:
                        # Lower is better - use negative for consistent ranking
                        avg_sim = -np.mean(method_sims)
                    else:
                        avg_sim = np.mean(method_sims)
                    avg_similarities.append((method, avg_sim))
            
            # Sort and plot
            avg_similarities.sort(key=lambda x: x[1], reverse=True)
            methods_sorted, scores_sorted = zip(*avg_similarities)
            
            colors = plt.cm.RdYlGn(np.linspace(0.3, 1.0, len(methods_sorted)))
            bars = ax.barh(range(len(methods_sorted)), scores_sorted, color=colors, alpha=0.8)
            
            ax.set_yticks(range(len(methods_sorted)))
            ax.set_yticklabels(methods_sorted)
            ax.set_xlabel(f'Average {metric.replace("_", " ").title()}')
            ax.set_title(f'Method Rankings by {metric.replace("_", " ").title()}', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, scores_sorted)):
                ax.text(score + max(scores_sorted)*0.01, bar.get_y() + bar.get_height()/2,
                       f'{abs(score):.3f}', ha='left', va='center', fontweight='bold')
        
        # Hide unused subplots
        for idx in range(len(metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Method Rankings by Similarity Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pairwise_similarity_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_similarity_correlations_plot(self, similarity_df: pd.DataFrame, metrics: List[str]):
        """Create correlation analysis between different similarity metrics."""
        # Create correlation matrix between metrics
        correlation_data = []
        
        # Get all unique method pairs (excluding self-comparisons)
        pairs_data = similarity_df[similarity_df['method1_clean'] != similarity_df['method2_clean']]
        
        if pairs_data.empty:
            return
        
        # Pivot to get metrics as columns
        pivot_data = pairs_data.pivot_table(
            index=['method1_clean', 'method2_clean'], 
            columns='metric', 
            values='value'
        ).reset_index()
        
        # Calculate correlation matrix
        metric_cols = [col for col in pivot_data.columns if col in metrics]
        if len(metric_cols) < 2:
            return
            
        corr_matrix = pivot_data[metric_cols].corr()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Correlation heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax1)
        ax1.set_title('Similarity Metrics Correlation Matrix', fontweight='bold')
        
        # Plot 2: Pairwise scatter plot (best correlated pair)
        if len(metric_cols) >= 2:
            # Find best correlated pair (excluding diagonal)
            corr_abs = np.abs(corr_matrix.values)
            np.fill_diagonal(corr_abs, 0)
            i, j = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
            
            metric1, metric2 = metric_cols[i], metric_cols[j]
            corr_val = corr_matrix.iloc[i, j]
            
            scatter_data = pivot_data.dropna(subset=[metric1, metric2])
            if not scatter_data.empty:
                ax2.scatter(scatter_data[metric1], scatter_data[metric2], 
                           alpha=0.7, s=100, color='darkblue')
                
                # Add trend line
                z = np.polyfit(scatter_data[metric1], scatter_data[metric2], 1)
                p = np.poly1d(z)
                ax2.plot(scatter_data[metric1], p(scatter_data[metric1]), 
                        "r--", alpha=0.8, linewidth=2)
                
                ax2.set_xlabel(metric1.replace('_', ' ').title())
                ax2.set_ylabel(metric2.replace('_', ' ').title())
                ax2.set_title(f'{metric1.replace("_", " ").title()} vs {metric2.replace("_", " ").title()}\\nr = {corr_val:.3f}', 
                             fontweight='bold')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pairwise_similarity_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_similarity_distributions_plot(self, similarity_df: pd.DataFrame, metrics: List[str]):
        """Create distribution analysis of similarity values."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            metric_data = similarity_df[
                (similarity_df['metric'] == metric) & 
                (similarity_df['method1_clean'] != similarity_df['method2_clean'])
            ]
            
            if not metric_data.empty:
                values = metric_data['value'].values
                
                # Create histogram
                ax.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(values):.3f}')
                ax.axvline(np.median(values), color='orange', linestyle='--', linewidth=2,
                          label=f'Median: {np.median(values):.3f}')
                
                ax.set_xlabel(f'{metric.replace("_", " ").title()} Value')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{metric.replace("_", " ").title()} Distribution', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Similarity Metrics Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pairwise_similarity_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_similarity_matrices(self, similarity_df: pd.DataFrame, metrics: List[str]):
        """Save similarity matrices as CSV files."""
        for metric in metrics:
            metric_data = similarity_df[similarity_df['metric'] == metric]
            pivot_matrix = metric_data.pivot(index='method1_clean', columns='method2_clean', values='value')
            # Make symmetric
            pivot_matrix = pivot_matrix.fillna(pivot_matrix.T)
            
            filename = f'pairwise_{metric}_matrix.csv'
            pivot_matrix.to_csv(self.output_dir / filename)
        
        print(f"  ✓ Saved {len(metrics)} similarity matrices as CSV files")

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
        
        # Core latent metrics
        core_metrics = ['hsic_global_score', 'active_units', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'trustworthiness', 'continuity', 'dist_corr']
        
        # Enhanced metrics
        enhanced_metrics = [
            'variance_entropy', 'effective_dim_variance', 'variance_concentration',
            'mi_gender_mean', 'mi_abnormal_mean', 'mi_gender_concentration', 'mi_abnormal_concentration'
        ]
        
        all_metrics = core_metrics + enhanced_metrics
        
        for metric in all_metrics:
            if metric in eval_latent.columns:
                data_available = eval_latent.dropna(subset=[metric])
                if not data_available.empty:
                    # Handle metrics with different optimization directions
                    if metric in ['hsic_global_score', 'davies_bouldin']:  # Lower is better
                        best_row = data_available.loc[data_available[metric].idxmin()]
                        direction = "(Lower Better)"
                    else:  # Higher is better
                        best_row = data_available.loc[data_available[metric].idxmax()]
                        if 'concentration' in metric:
                            direction = "(Higher = More Concentrated)"
                        else:
                            direction = "(Higher Better)"
                    
                    latent_summary.append({
                        'Metric': metric.replace('_', ' ').title() + f" {direction}",
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
            
            # Latent metrics (core + key enhanced metrics)
            latent_data = eval_latent[eval_latent['method'] == method]
            if not latent_data.empty:
                # Core metrics
                core_latent_metrics = ['dim', 'active_units', 'hsic_global_score', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'trustworthiness', 'continuity', 'dist_corr']
                # Enhanced metrics
                enhanced_latent_metrics = ['variance_entropy', 'mi_gender_mean', 'mi_abnormal_mean']
                
                all_latent_metrics = core_latent_metrics + enhanced_latent_metrics
                
                for metric in all_latent_metrics:
                    val = latent_data.iloc[0][metric] if metric in latent_data.iloc[0] else None
                    if metric in ['dim', 'active_units']:
                        row[metric] = f"{int(val)}" if not pd.isna(val) else "N/A"
                    else:
                        row[metric] = f"{val:.3f}" if not pd.isna(val) else "N/A"
                
                # Add efficiency calculation
                if not pd.isna(latent_data.iloc[0].get('active_units')) and not pd.isna(latent_data.iloc[0].get('dim')):
                    efficiency = latent_data.iloc[0]['active_units'] / latent_data.iloc[0]['dim']
                    row['efficiency'] = f"{efficiency:.3f}"
                else:
                    row['efficiency'] = "N/A"
            else:
                for metric in ['dim', 'active_units', 'hsic_global_score', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'trustworthiness', 'continuity', 'dist_corr',
                              'variance_entropy', 'mi_gender_mean', 'mi_abnormal_mean', 'efficiency']:
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
            'trustworthiness', 'continuity', 'dist_corr', 'variance_mean', 'variance_entropy',
            'mi_gender_mean', 'mi_abnormal_mean', 'variance_concentration'
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
        
        # Create pairwise similarity analysis
        self.create_pairwise_similarity_analysis()
        
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
    
  
    small_aggregated = ["tuh-ctm_nn_avg", "tuh-psd_ae_avg", "tuh-ctm_cma_avg","tuh-jr_avg","tuh-wong_wang_avg", "tuh-hopf_avg", "tuh-pca_avg"]
    small_aggregated.sort()
    small_aggregated.reverse()

   
    medium_unrestricted = ["tuh-ctm_nn_pc", "tuh-psd_ae_pc", "tuh-pca_pc", "tuh-c22", "tuh-eegnet"]
    medium_unrestricted.sort()
    medium_unrestricted.reverse()


    #method_group = {"small_aggregated": small_aggregated}
    method_group = {"medium_unrestricted": medium_unrestricted}

    # =================================================================
    
    # Initialize and run the comparison
    comparison = MetricsComparison(method_group=method_group)
    comparison.generate_comprehensive_report()


if __name__ == "__main__":
    main()
