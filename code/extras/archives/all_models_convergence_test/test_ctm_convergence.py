#!/usr/bin/env python3
"""
Test CTM model fitting with different convergence tolerances.
Compare speed vs accuracy trade-offs for different tolfun values.
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path

# Add the code directory to the Python path
sys.path.insert(0, '/rds/general/user/lrh24/home/thesis/code')
from latent_extraction.cortico_thalamic import fit_parameters, _P_omega
from utils.util import compute_psd_from_raw

def test_convergence_tolerance(edf_path, tolfun_values=None, output_dir=None):
    """
    Test CTM fitting with different convergence tolerances.
    
    Args:
        edf_path: Path to EDF file
        tolfun_values: List of tolfun values to test
        output_dir: Directory to save plots
    """
    
    if tolfun_values is None:
        # Test range from very strict to very loose
        tolfun_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    if output_dir is None:
        output_dir = Path('/rds/general/user/lrh24/home/thesis/ctm_convergence_test')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    print("🔬 CTM Convergence Tolerance Analysis")
    print("=" * 50)
    
    # Load and preprocess EEG data
    print("📁 Loading EEG data...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick_channels(raw.ch_names[:4])  # Use first 4 channels
    raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
    print(f"✅ Loaded: {raw.info['sfreq']} Hz, {len(raw.ch_names)} channels")
    
    # Compute real PSD
    print("📊 Computing PSD...")
    real_psd, freqs = compute_psd_from_raw(raw, calculate_average=True, normalize=False, return_freqs=True)
    print(f"✅ PSD computed: {len(freqs)} frequency points")
    
    # Storage for results
    results = {
        'tolfun': [],
        'fit_time': [],
        'final_loss': [],
        'iterations': [],
        'parameters': [],
        'model_psd': []
    }
    
    print(f"\n🧪 Testing {len(tolfun_values)} convergence tolerances (extended range: 1e-9 to 1e-1)...")
    
    # Test each tolfun value
    for i, tolfun in enumerate(tolfun_values):
        print(f"\n--- Test {i+1}/{len(tolfun_values)}: tolfun = {tolfun:.0e} ---")
        
        # Fit model with current tolfun
        start_time = time.perf_counter()
        
        best_params, theta_best, final_loss = fit_parameters(
            freqs, real_psd,
            cma_opts={'tolfun': tolfun},
            return_full=True
        )
        
        end_time = time.perf_counter()
        fit_time = end_time - start_time
        
        # Generate model PSD with fitted parameters
        model_psd = _P_omega(best_params, freqs)
        
        # Store results
        results['tolfun'].append(tolfun)
        results['fit_time'].append(fit_time)
        results['final_loss'].append(final_loss)
        results['parameters'].append(best_params.copy())
        results['model_psd'].append(model_psd.copy())
        
        # Compute quality metrics
        mse = np.mean((model_psd - real_psd) ** 2)
        r2 = 1 - np.sum((real_psd - model_psd)**2) / np.sum((real_psd - np.mean(real_psd))**2)
        
        print(f"  ⚡ Time: {fit_time:.3f}s")
        print(f"  📉 Final loss: {final_loss:.6f}")
        print(f"  📈 MSE: {mse:.6f}")
        print(f"  🎯 R²: {r2:.4f}")
        print(f"  🔧 Key params: G_ee={best_params['G_ee']:.2f}, alpha={best_params['alpha']:.1f}")
    
    # Convert to numpy arrays for easier handling
    for key in ['tolfun', 'fit_time', 'final_loss']:
        results[key] = np.array(results[key])
    
    # Create comprehensive plots
    print(f"\n📊 Creating plots...")
    create_convergence_plots(results, freqs, real_psd, output_dir)
    
    # Print summary table
    print_summary_table(results)
    
    print(f"\n✅ Analysis complete! Plots saved to: {output_dir}")
    
    return results

def create_convergence_plots(results, freqs, real_psd, output_dir):
    """Create comprehensive plots showing convergence analysis."""
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 100})
    
    # 1. Speed vs Accuracy Trade-off
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Fitting time vs tolfun
    ax1.loglog(results['tolfun'], results['fit_time'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Convergence Tolerance (tolfun)')
    ax1.set_ylabel('Fitting Time (seconds)')
    ax1.set_title('⚡ Fitting Speed vs Convergence Tolerance')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Smaller tolfun (stricter) on the right
    
    # Add annotations for time savings
    fastest_idx = np.argmin(results['fit_time'])
    slowest_idx = np.argmax(results['fit_time'])
    ax1.annotate(f'Fastest: {results["fit_time"][fastest_idx]:.2f}s', 
                xy=(results['tolfun'][fastest_idx], results['fit_time'][fastest_idx]),
                xytext=(10, 10), textcoords='offset points', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'),
                arrowprops=dict(arrowstyle='->', color='green'))
    
    # Plot 2: Final loss vs tolfun
    ax2.loglog(results['tolfun'], results['final_loss'], 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Convergence Tolerance (tolfun)')
    ax2.set_ylabel('Final Loss')
    ax2.set_title('🎯 Fitting Quality vs Convergence Tolerance')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    # Plot 3: Speed vs Quality scatter
    colors = plt.cm.viridis(np.linspace(0, 1, len(results['tolfun'])))
    scatter = ax3.scatter(results['fit_time'], results['final_loss'], 
                         c=np.log10(results['tolfun']), s=100, cmap='viridis')
    ax3.set_xlabel('Fitting Time (seconds)')
    ax3.set_ylabel('Final Loss')
    ax3.set_title('⚖️ Speed vs Quality Trade-off')
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('log₁₀(tolfun)')
    
    # Annotate points with tolfun values
    for i, tolfun in enumerate(results['tolfun']):
        ax3.annotate(f'{tolfun:.0e}', 
                    (results['fit_time'][i], results['final_loss'][i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 4: Parameter stability
    param_names = ['G_ee', 'G_ei', 'alpha', 'beta']
    for i, param_name in enumerate(param_names):
        values = [params[param_name] for params in results['parameters']]
        ax4.semilogx(results['tolfun'], values, 'o-', label=param_name, linewidth=2)
    
    ax4.set_xlabel('Convergence Tolerance (tolfun)')
    ax4.set_ylabel('Parameter Value')
    ax4.set_title('🔧 Parameter Stability vs Convergence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PSD Comparison Plot
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    # Show 12 different tolfun values (spread across the range)
    indices_to_show = np.linspace(0, len(results['tolfun'])-1, 12, dtype=int)
    
    for i, idx in enumerate(indices_to_show):
        ax = axes[i]
        tolfun = results['tolfun'][idx]
        model_psd = results['model_psd'][idx]
        
        # Plot real vs model PSD
        ax.loglog(freqs, real_psd, 'k-', linewidth=2, label='Real EEG', alpha=0.8)
        ax.loglog(freqs, model_psd, 'r--', linewidth=2, label='CTM Model', alpha=0.8)
        
        # Compute R²
        r2 = 1 - np.sum((real_psd - model_psd)**2) / np.sum((real_psd - np.mean(real_psd))**2)
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title(f'tolfun = {tolfun:.0e}\nR² = {r2:.3f}, Time = {results["fit_time"][idx]:.2f}s')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, 45])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'psd_comparisons.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Recommended tolerance plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Normalize metrics for comparison
    norm_time = (results['fit_time'] - results['fit_time'].min()) / (results['fit_time'].max() - results['fit_time'].min())
    norm_loss = (results['final_loss'] - results['final_loss'].min()) / (results['final_loss'].max() - results['final_loss'].min())
    
    # Combined score (lower is better)
    combined_score = norm_time + norm_loss
    best_idx = np.argmin(combined_score)
    
    ax.semilogx(results['tolfun'], norm_time, 'b-o', label='Normalized Time', linewidth=2)
    ax.semilogx(results['tolfun'], norm_loss, 'r-o', label='Normalized Loss', linewidth=2)
    ax.semilogx(results['tolfun'], combined_score, 'g-o', label='Combined Score', linewidth=3)
    
    # Mark the optimal point
    ax.axvline(results['tolfun'][best_idx], color='orange', linestyle='--', linewidth=2, 
               label=f'Optimal: {results["tolfun"][best_idx]:.0e}')
    
    ax.set_xlabel('Convergence Tolerance (tolfun)')
    ax.set_ylabel('Normalized Score')
    ax.set_title('🎯 Optimal Convergence Tolerance Selection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimal_tolerance.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_summary_table(results):
    """Print a summary table of results."""
    
    print("\n📋 CONVERGENCE TOLERANCE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"{'tolfun':<12} {'Time (s)':<10} {'Loss':<12} {'Speedup':<10} {'Quality':<10}")
    print("-" * 80)
    
    base_time = results['fit_time'][0]  # Most strict tolerance time
    base_loss = results['final_loss'][0]
    
    for i in range(len(results['tolfun'])):
        tolfun = results['tolfun'][i]
        fit_time = results['fit_time'][i]
        final_loss = results['final_loss'][i]
        speedup = base_time / fit_time
        quality_ratio = final_loss / base_loss
        
        print(f"{tolfun:<12.0e} {fit_time:<10.3f} {final_loss:<12.6f} {speedup:<10.2f}x {quality_ratio:<10.3f}")
    
    # Find the sweet spot
    norm_time = (results['fit_time'] - results['fit_time'].min()) / (results['fit_time'].max() - results['fit_time'].min())
    norm_loss = (results['final_loss'] - results['final_loss'].min()) / (results['final_loss'].max() - results['final_loss'].min())
    combined_score = norm_time + norm_loss
    best_idx = np.argmin(combined_score)
    
    print("\n🎯 RECOMMENDATION:")
    print(f"   Optimal tolfun: {results['tolfun'][best_idx]:.0e}")
    print(f"   Speedup: {base_time / results['fit_time'][best_idx]:.2f}x")
    print(f"   Quality loss: {(results['final_loss'][best_idx] / base_loss - 1) * 100:.1f}%")

def main():
    """Main function to run the convergence analysis."""
    
    # Configuration
    edf_path = "/rds/general/user/lrh24/ephemeral/edf/train/normal/01_tcp_ar/aaaaambs_s003_t000.edf"
    
    # Extended test range - from very strict to very loose
    tolfun_values = [1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    
    try:
        results = test_convergence_tolerance(edf_path, tolfun_values)
        
        print("\n🎉 Analysis completed successfully!")
        print("Check the generated plots to see the speed vs accuracy trade-offs.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
