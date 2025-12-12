"""
Visualization & Analysis for Communication Optimization Experiments
Generate 5 separate graphs from unified experimental results

Output Graphs:
1. perbandingan_strong_scaling.png - Fase 1: Standard vs Hybrid vs Optimized strong scaling (512x512)
2. perbandingan_weak_scaling.png - Fase 1: Standard vs Hybrid vs Optimized weak scaling  
3. strong_scaling.png - Fase 2: Hybrid vs Optimized strong scaling performance (4096x4096)
4. weak_scaling.png - Fase 2: Hybrid vs Optimized weak scaling efficiency trend
5. communication_overhead_analysis.png - Detailed communication impact analysis
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob
import warnings

warnings.filterwarnings('ignore')

# Professional color scheme
COLORS = {
    'standard': '#e74c3c',      # Red - inefficient baseline
    'hybrid': '#f39c12',         # Orange - intermediate optimization
    'optimized': '#27ae60',      # Green - efficient with full optimization
    'ideal': '#34495e'           # Dark gray - ideal theoretical
}

FONT_CONFIG = {
    'title': 14,
    'subtitle': 12,
    'label': 11,
    'legend': 10,
    'value': 9
}


def load_results(csv_file):
    """Load experimental results from CSV"""
    try:
        df = pd.read_csv(csv_file)
        print(f"[√] Loaded {len(df)} records from {os.path.basename(csv_file)}")
        return df
    except Exception as e:
        print(f"[✗] Error loading CSV: {e}")
        return None


def calculate_efficiency(speedup, num_procs):
    """Calculate parallel efficiency percentage"""
    return (speedup / num_procs) * 100 if num_procs > 0 else 0


def find_latest_csv():
    """Find latest CSV file in results directory"""
    csv_files = glob.glob('results/experiment_results_*.csv')
    if csv_files:
        latest = max(csv_files, key=os.path.getctime)
        print(f"[→] Found latest results: {os.path.basename(latest)}")
        return latest
    return None


# ============================================================================
# GRAPH 1: Perbandingan Strong Scaling (Phase 1) - Standard vs Hybrid vs Optimized
# ============================================================================

def plot_1_perbandingan_strong_scaling(df, output_dir):
    """
    Graph 1: Phase 1 Strong Scaling Comparison
    Menunjukkan PROGRESSION dari Standard → Hybrid → Optimized pada matrix 512x512
    Demonstrasi komunikasi overhead reduction spectrum
    Updated to support 1, 2, 3, 4 processes
    """
    phase1_strong = df[(df['Phase'] == 'phase1_strong')]
    
    if phase1_strong.empty:
        print("[⚠] No phase1_strong data found")
        return False
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Graph 1: Communication Overhead - Strong Scaling (512×512)\nStandard vs Hybrid vs Optimized Comparison (1-4 Processes)', 
                     fontsize=FONT_CONFIG['title'], fontweight='bold', y=0.98)
        
        x_pos = sorted(phase1_strong['Processes'].unique())
        width = 0.25
        
        # Subplot 1: Execution Time
        ax1 = axes[0]
        
        for idx, version in enumerate(['standard', 'hybrid', 'optimized']):
            times = []
            for procs in x_pos:
                data = phase1_strong[(phase1_strong['Processes'] == procs) & 
                                    (phase1_strong['Version'] == version)]
                times.append(data['Time(s)'].values[0] if not data.empty else 0)
            
            offset = width * (idx - 1)
            bars = ax1.bar([p + offset for p in x_pos], times, width,
                          label=version.capitalize(), color=COLORS[version], alpha=0.8)
            
            # Value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}s', ha='center', va='bottom',
                            fontsize=FONT_CONFIG['value'])
        
        ax1.set_xlabel('Number of Processes', fontsize=FONT_CONFIG['label'], fontweight='bold')
        ax1.set_ylabel('Execution Time (seconds)', fontsize=FONT_CONFIG['label'], fontweight='bold')
        ax1.set_title('Execution Time: Optimization Strategy Comparison', fontsize=FONT_CONFIG['subtitle'], fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.legend(fontsize=FONT_CONFIG['legend'])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Subplot 2: Speedup Progression
        ax2 = axes[1]
        
        for version in ['standard', 'hybrid', 'optimized']:
            speedups = []
            procs_list = []
            
            for procs in x_pos:
                data = phase1_strong[(phase1_strong['Processes'] == procs) & 
                                    (phase1_strong['Version'] == version)]
                if not data.empty:
                    speedup = data['Speedup'].values[0]
                    speedups.append(speedup)
                    procs_list.append(procs)
            
            if speedups:
                marker = 'o' if version == 'standard' else 's' if version == 'hybrid' else '^'
                ax2.plot(procs_list, speedups, marker=marker,
                        linewidth=2.5, markersize=10, label=version.capitalize(),
                        color=COLORS[version], alpha=0.8)
        
        # Ideal speedup line
        ax2.plot(x_pos, x_pos, 'k--', linewidth=2, label='Ideal Speedup', alpha=0.6)
        
        ax2.set_xlabel('Number of Processes', fontsize=FONT_CONFIG['label'], fontweight='bold')
        ax2.set_ylabel('Speedup Factor', fontsize=FONT_CONFIG['label'], fontweight='bold')
        ax2.set_title('Speedup: Communication Efficiency Spectrum', fontsize=FONT_CONFIG['subtitle'], fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_ylim([0, max(x_pos) + 1])
        ax2.legend(fontsize=FONT_CONFIG['legend'])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'perbandingan_strong_scaling.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[✓] Graph 1 saved: {output_path}")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"[✗] Error in graph 1: {e}")
        plt.close('all')
        return False


# ============================================================================
# GRAPH 2: Perbandingan Weak Scaling (Phase 1) - Standard vs Hybrid vs Optimized
# ============================================================================

def plot_2_perbandingan_weak_scaling(df, output_dir):
    """
    Graph 2: Phase 1 Weak Scaling Comparison
    Menunjukkan komunikasi efficiency trend untuk 3 strategi
    Updated to support 1, 2, 3, 4 processes
    """
    phase1_weak = df[(df['Phase'] == 'phase1_weak')]
    
    if phase1_weak.empty:
        print("[⚠] No phase1_weak data found")
        return False
    
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle('Graph 2: Weak Scaling Efficiency - Phase 1 Comparison\n(Problem Size Scaled with Process Count - 3 Strategies, 1-4 Processes)', 
                     fontsize=FONT_CONFIG['title'], fontweight='bold', y=0.98)
        
        for version in ['standard', 'hybrid', 'optimized']:
            version_data = phase1_weak[phase1_weak['Version'] == version].sort_values('Processes')
            
            if not version_data.empty:
                times = version_data['Time(s)'].values
                procs = version_data['Processes'].values
                
                # Calculate efficiency relative to baseline
                baseline_time = times[0]
                efficiencies = [(baseline_time / t * 100) for t in times]
                
                marker = 'o' if version == 'standard' else 's' if version == 'hybrid' else '^'
                ax.plot(procs, efficiencies, marker=marker,
                       linewidth=2.5, markersize=10, label=version.capitalize(),
                       color=COLORS[version], alpha=0.8)
                
                # Add value labels
                for p, eff in zip(procs, efficiencies):
                    ax.text(p, eff, f'{eff:.1f}%', ha='center', va='bottom',
                           fontsize=FONT_CONFIG['value'])
        
        # Ideal line (100% efficiency)
        ax.axhline(y=100, color='k', linestyle='--', linewidth=2, 
                  label='Ideal (100%)', alpha=0.6)
        
        ax.set_xlabel('Number of Processes', fontsize=FONT_CONFIG['label'], fontweight='bold')
        ax.set_ylabel('Weak Scaling Efficiency (%)', fontsize=FONT_CONFIG['label'], fontweight='bold')
        ax.set_title('Communication Efficiency Spectrum Across Strategies (1-4 Processes)', fontsize=FONT_CONFIG['subtitle'], fontweight='bold')
        ax.set_ylim([0, 120])
        ax.legend(fontsize=FONT_CONFIG['legend'], loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'perbandingan_weak_scaling.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[✓] Graph 2 saved: {output_path}")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"[✗] Error in graph 2: {e}")
        plt.close('all')
        return False


# ============================================================================
# GRAPH 3: Strong Scaling (Phase 2) - Hybrid + Optimized
# ============================================================================

def plot_3_strong_scaling_phase2(df, output_dir):
    """
    Graph 3: Phase 2 Strong Scaling Performance
    Hybrid + Optimized comparison (was Optimized only)
    Updated to support 1, 2, 3, 4 processes
    """
    phase2_strong = df[(df['Phase'] == 'phase2_strong')]
    
    if phase2_strong.empty:
        print("[⚠] No phase2_strong data found")
        return False
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Graph 3: Strong Scaling Analysis - Production Scale (4096×4096)\nHybrid vs Optimized Implementation (1-4 Processes)',
                     fontsize=FONT_CONFIG['title'], fontweight='bold', y=0.98)
        
        x_pos = sorted(phase2_strong['Processes'].unique())
        width = 0.35
        
        # Subplot 1: Execution Time Comparison
        ax1_idx = 0
        for version in ['hybrid', 'optimized']:
            times = []
            for procs in x_pos:
                data = phase2_strong[(phase2_strong['Processes'] == procs) & 
                                    (phase2_strong['Version'] == version)]
                times.append(data['Time(s)'].values[0] if not data.empty else 0)
            
            offset = width * (ax1_idx - 0.5)
            bars = ax1.bar([p + offset for p in x_pos], times, width,
                          label=version.capitalize(), color=COLORS[version], alpha=0.8)
            
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}s', ha='center', va='bottom',
                            fontsize=FONT_CONFIG['value'])
            
            ax1_idx += 1
        
        ax1.set_xlabel('Number of Processes', fontsize=FONT_CONFIG['label'], fontweight='bold')
        ax1.set_ylabel('Execution Time (seconds)', fontsize=FONT_CONFIG['label'], fontweight='bold')
        ax1.set_title('Execution Time: Hybrid vs Optimized Comparison', fontsize=FONT_CONFIG['subtitle'], fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.legend(fontsize=FONT_CONFIG['legend'])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Subplot 2: Speedup Comparison
        ax2_idx = 0
        for version in ['hybrid', 'optimized']:
            speedups = []
            procs_list = []
            
            for procs in x_pos:
                data = phase2_strong[(phase2_strong['Processes'] == procs) & 
                                    (phase2_strong['Version'] == version)]
                if not data.empty:
                    speedup = data['Speedup'].values[0]
                    speedups.append(speedup)
                    procs_list.append(procs)
            
            if speedups:
                marker = 's' if version == 'hybrid' else '^'
                ax2.plot(procs_list, speedups, marker=marker,
                        linewidth=2.5, markersize=10, label=version.capitalize(),
                        color=COLORS[version], alpha=0.8)
        
        # Ideal speedup line
        ax2.plot(x_pos, x_pos, 'k--', linewidth=2, label='Ideal Speedup', alpha=0.6)
        
        ax2.set_xlabel('Number of Processes', fontsize=FONT_CONFIG['label'], fontweight='bold')
        ax2.set_ylabel('Speedup Factor', fontsize=FONT_CONFIG['label'], fontweight='bold')
        ax2.set_title('Speedup: Production Scale Efficiency', fontsize=FONT_CONFIG['subtitle'], fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_ylim([0, max(x_pos) + 1])
        ax2.legend(fontsize=FONT_CONFIG['legend'])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'strong_scaling.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[✓] Graph 3 saved: {output_path}")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"[✗] Error in graph 3: {e}")
        plt.close('all')
        return False


# ============================================================================
# GRAPH 4: Weak Scaling (Phase 2) - Hybrid + Optimized  
# ============================================================================

def plot_4_weak_scaling_phase2(df, output_dir):
    """
    Graph 4: Phase 2 Weak Scaling Performance
    Hybrid + Optimized comparison (was Optimized only)
    Updated to support 1, 2, 3, 4 processes
    """
    phase2_weak = df[(df['Phase'] == 'phase2_weak')]
    
    if phase2_weak.empty:
        print("[⚠] No phase2_weak data found")
        return False
    
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle('Graph 4: Weak Scaling Analysis - Large Workloads\nHybrid vs Optimized Implementation (1-4 Processes, Problem Size Scales)',
                     fontsize=FONT_CONFIG['title'], fontweight='bold', y=0.98)
        
        for version in ['hybrid', 'optimized']:
            version_data = phase2_weak[phase2_weak['Version'] == version].sort_values('Processes')
            
            if not version_data.empty:
                times = version_data['Time(s)'].values
                procs = version_data['Processes'].values
                
                baseline_time = times[0]
                efficiencies = [(baseline_time / t * 100) for t in times]
                
                marker = 's' if version == 'hybrid' else '^'
                ax.plot(procs, efficiencies, marker=marker,
                       linewidth=2.5, markersize=10, label=version.capitalize(),
                       color=COLORS[version], alpha=0.8)
                
                for p, eff in zip(procs, efficiencies):
                    ax.text(p, eff, f'{eff:.1f}%', ha='center', va='bottom',
                           fontsize=FONT_CONFIG['value'])
        
        # Ideal line (100% efficiency)
        ax.axhline(y=100, color='k', linestyle='--', linewidth=2, 
                  label='Ideal (100%)', alpha=0.6)
        
        ax.set_xlabel('Number of Processes', fontsize=FONT_CONFIG['label'], fontweight='bold')
        ax.set_ylabel('Weak Scaling Efficiency (%)', fontsize=FONT_CONFIG['label'], fontweight='bold')
        ax.set_title('Weak Scaling Efficiency: Hybrid vs Optimized (1-4 Processes)', fontsize=FONT_CONFIG['subtitle'], fontweight='bold')
        ax.set_ylim([0, 120])
        ax.legend(fontsize=FONT_CONFIG['legend'], loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'weak_scaling.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[✓] Graph 4 saved: {output_path}")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"[✗] Error in graph 4: {e}")
        plt.close('all')
        return False


# ============================================================================
# GRAPH 5: Communication Overhead Analysis (Detailed Insight)
# ============================================================================

def plot_5_communication_overhead_analysis(df, output_dir):
    """
    Graph 5: Detailed Communication Overhead Analysis
    Comprehensive view of how communication impacts performance
    Kompare all phases untuk menunjukkan optimization effect
    """
    try:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        # ===== Subplot 1: Phase 1 Strong Scaling Overhead =====
        ax1 = fig.add_subplot(gs[0, 0])
        
        phase1_strong = df[df['Phase'] == 'phase1_strong']
        if not phase1_strong.empty:
            for version in ['standard', 'hybrid', 'optimized']:
                version_data = phase1_strong[phase1_strong['Version'] == version].sort_values('Processes')
                
                if not version_data.empty:
                    speedups = version_data['Speedup'].values
                    procs = version_data['Processes'].values
                    efficiencies = [calculate_efficiency(s, p) for s, p in zip(speedups, procs)]
                    
                    ax1.plot(procs, efficiencies, marker='o' if version == 'standard' else 's' if version == 'hybrid' else '^',
                            linewidth=2.5, markersize=8, label=version.capitalize(),
                            color=COLORS[version], alpha=0.8)
            
            ax1.axhline(y=100, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
            ax1.set_xlabel('Processes', fontweight='bold')
            ax1.set_ylabel('Efficiency (%)', fontweight='bold')
            ax1.set_title('Phase 1: Strong Scaling (512²)\nCommunication Overhead Impact', fontweight='bold')
            ax1.set_xticks([1, 2, 3, 4])
            ax1.set_ylim([0, 120])
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
        
        # ===== Subplot 2: Phase 1 Weak Scaling Overhead =====
        ax2 = fig.add_subplot(gs[0, 1])
        
        phase1_weak = df[df['Phase'] == 'phase1_weak']
        if not phase1_weak.empty:
            for version in ['standard', 'hybrid', 'optimized']:
                version_data = phase1_weak[phase1_weak['Version'] == version].sort_values('Processes')
                
                if not version_data.empty:
                    times = version_data['Time(s)'].values
                    procs = version_data['Processes'].values
                    baseline = times[0]
                    efficiencies = [(baseline / t * 100) for t in times]
                    
                    ax2.plot(procs, efficiencies, marker='o' if version == 'standard' else 's' if version == 'hybrid' else '^',
                            linewidth=2.5, markersize=8, label=version.capitalize(),
                            color=COLORS[version], alpha=0.8)
            
            ax2.axhline(y=100, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
            ax2.set_xlabel('Processes', fontweight='bold')
            ax2.set_ylabel('Efficiency (%)', fontweight='bold')
            ax2.set_title('Phase 1: Weak Scaling\nCommunication Efficiency Trend', fontweight='bold')
            ax2.set_ylim([0, 120])
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
        
        # ===== Subplot 3: Phase 2 Strong Scaling Performance =====
        ax3 = fig.add_subplot(gs[1, 0])
        
        phase2_strong = df[df['Phase'] == 'phase2_strong']
        if not phase2_strong.empty:
            data_sorted = phase2_strong.sort_values('Processes')
            procs = data_sorted['Processes'].values
            times = data_sorted['Time(s)'].values
            speedups = data_sorted['Speedup'].values
            efficiencies = [calculate_efficiency(s, p) for s, p in zip(speedups, procs)]
            
            # Time bars
            bars = ax3.bar(procs - 0.15, times, 0.3, label='Execution Time',
                          color=COLORS['optimized'], alpha=0.8)
            
            # Efficiency on secondary axis
            ax3_twin = ax3.twinx()
            ax3_twin.plot(procs, efficiencies, 'o-', color='#e67e22',
                         linewidth=2.5, markersize=8, label='Efficiency', alpha=0.8)
            ax3_twin.set_ylabel('Efficiency (%)', fontweight='bold', color='#e67e22')
            ax3_twin.set_ylim([0, 120])
            
            ax3.set_xlabel('Processes', fontweight='bold')
            ax3.set_ylabel('Time (s)', fontweight='bold')
            ax3.set_title('Phase 2: Strong Scaling (4096²)\nOptimized Performance', fontweight='bold')
            ax3.set_xticks(procs)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # ===== Subplot 4: Communication Overhead Percentage =====
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Calculate overhead from all phases
        all_data = []
        labels = []
        
        for phase_key in ['phase1_strong', 'phase1_weak', 'phase2_strong', 'phase2_weak']:
            phase_data = df[df['Phase'] == phase_key]
            if not phase_data.empty and 'optimized' in phase_data['Version'].values:
                opt_data = phase_data[phase_data['Version'] == 'optimized'].sort_values('Processes')
                if len(opt_data) > 1:
                    times = opt_data['Time(s)'].values
                    baseline = times[0]
                    speedups = opt_data['Speedup'].values
                    efficiencies = [calculate_efficiency(s, p) for s, p in 
                                   zip(speedups, opt_data['Processes'].values)]
                    avg_overhead = 100 - np.mean(efficiencies)
                    all_data.append(avg_overhead)
                    
                    phase_labels = {
                        'phase1_strong': 'P1 Strong\n(512²)',
                        'phase1_weak': 'P1 Weak',
                        'phase2_strong': 'P2 Strong\n(4096²)',
                        'phase2_weak': 'P2 Weak'
                    }
                    labels.append(phase_labels.get(phase_key, phase_key))
        
        if all_data:
            colors_overhead = ['#e74c3c' if x > 20 else '#f39c12' if x > 10 else '#27ae60' 
                             for x in all_data]
            bars = ax4.bar(range(len(all_data)), all_data, color=colors_overhead, alpha=0.8)
            
            # Value labels
            for i, (bar, val) in enumerate(zip(bars, all_data)):
                ax4.text(i, val, f'{val:.1f}%', ha='center', va='bottom',
                        fontsize=FONT_CONFIG['value'], fontweight='bold')
            
            ax4.set_xticks(range(len(all_data)))
            ax4.set_xticklabels(labels, fontsize=9)
            ax4.set_ylabel('Communication Overhead (%)', fontweight='bold')
            ax4.set_title('Average Communication Overhead by Phase\n(Lower = Better)', fontweight='bold')
            ax4.set_ylim([0, max(all_data) * 1.2 if all_data else 30])
            ax4.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Graph 5: Comprehensive Communication Overhead Analysis\nOptimization Impact Across All Phases',
                    fontsize=FONT_CONFIG['title'], fontweight='bold', y=0.995)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'communication_overhead_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[✓] Graph 5 saved: {output_path}")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"[✗] Error in graph 5: {e}")
        plt.close('all')
        return False


def main():
    print(f"\n{'='*80}")
    print("VISUALIZATION: Communication Optimization Analysis".center(80))
    print(f"{'='*80}\n")
    
    # Find or use provided CSV
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = find_latest_csv()
    
    if not csv_file or not os.path.exists(csv_file):
        print("[✗] CSV file not found!")
        print("   Usage: python scripts/plot_results.py [csv_file]")
        print("   Or run: python scripts/run_experiment.py first")
        sys.exit(1)
    
    # Load data
    df = load_results(csv_file)
    if df is None:
        sys.exit(1)
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Generate 5 graphs
    print("\n[→] Generating 5 visualization graphs...\n")
    
    success_count = 0
    success_count += plot_1_perbandingan_strong_scaling(df, 'results')
    success_count += plot_2_perbandingan_weak_scaling(df, 'results')
    success_count += plot_3_strong_scaling_phase2(df, 'results')
    success_count += plot_4_weak_scaling_phase2(df, 'results')
    success_count += plot_5_communication_overhead_analysis(df, 'results')
    
    print(f"\n{'='*80}")
    print(f"[✓] Visualization Complete: {success_count}/5 graphs generated".center(80))
    print(f"{'='*80}")
    print(f"\nGraphs saved in: {os.path.abspath('results/')}")
    print("\nGenerated files:")
    print("  1. perbandingan_strong_scaling.png - Phase 1 Standard vs Hybrid vs Optimized (512²)")
    print("  2. perbandingan_weak_scaling.png - Phase 1 Weak scaling comparison")
    print("  3. strong_scaling.png - Phase 2 Hybrid vs Optimized strong scaling (4096²)")
    print("  4. weak_scaling.png - Phase 2 Hybrid vs Optimized weak scaling")
    print("  5. communication_overhead_analysis.png - Detailed overhead analysis")
    print()


if __name__ == "__main__":
    main()
