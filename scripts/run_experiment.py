"""
OPTIMIZED EXPERIMENT SUITE FOR MULTI-CORE SYSTEMS
Tests with 1, 2, 3, and 4 processes (Intel i3-7020U + dual-core CPUs)

SKENARIO EKSPERIMEN:

FASE 1: GAP ANALYSIS - Small Matrices (3 Strategies: Standard, Hybrid, Optimized)
  - Strong Scaling: Matrix 512x512, dengan proses 1, 2, 3, 4
  - Weak Scaling: Proses 1 (matrix 256), Proses 2 (matrix 362), Proses 3 (matrix 443), Proses 4 (matrix 512)
  - Tujuan: Identifikasi GAP komunikasi overhead antara Standard vs Hybrid vs Optimized
  
FASE 2: LARGE-SCALE ANALYSIS - Full Load Test (Hybrid + Optimized)
  - Strong Scaling: Matrix 4096x4096, proses 1, 2, 3, 4 (Both Hybrid & Optimized)
  - Weak Scaling: Proses 1 (matrix 2048), Proses 2 (matrix 2896), Proses 3 (matrix 3544), Proses 4 (matrix 4096) (Both Hybrid & Optimized)
  - Tujuan: Analisis performa dan komunikasi overhead pada skala produksi dengan multiple processes (Hybrid vs Optimized comparison)

OUTPUT: Satu CSV file berisi semua hasil fase 1 & 2 untuk 1-4 processes dengan 2 strategi di fase 2
PLOTTING: 5 grafik terpisah dari data tersebut (mendukung 1-4 processes, dengan hybrid di phase 2 graphs)
"""
import subprocess
import csv
import os
import shutil
import sys
from datetime import datetime

class TerminalFormatter:
    """Format terminal output cleanly and readable"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    
    @staticmethod
    def section(title):
        return f"\n{'='*80}\n{title:^80}\n{'='*80}"
    
    @staticmethod
    def subsection(title):
        return f"\n{title}\n{'-'*80}"
    
    @staticmethod
    def success(msg):
        return f"{TerminalFormatter.GREEN}✓{TerminalFormatter.END} {msg}"
    
    @staticmethod
    def info(msg):
        return f"{TerminalFormatter.BLUE}→{TerminalFormatter.END} {msg}"
    
    @staticmethod
    def warning(msg):
        return f"{TerminalFormatter.YELLOW}⚠{TerminalFormatter.END} {msg}"
    
    @staticmethod
    def error(msg):
        return f"{TerminalFormatter.RED}✗{TerminalFormatter.END} {msg}"


def get_mpi_launcher():
    """Detect available MPI launcher"""
    if shutil.which('mpirun'):
        return 'mpirun', '-np'
    elif shutil.which('mpiexec'):
        return 'mpiexec', '-n'
    else:
        return None, None


def run_experiment(program_path, num_procs, matrix_size, timeout_sec=120):
    """
    Run single experiment and collect timing data
    Returns: (execution_time, output_text) or (None, None) on failure
    """
    try:
        prog = os.path.normpath(program_path)
        mpi_cmd, mpi_flag = get_mpi_launcher()
        
        # Build command
        if mpi_cmd and int(num_procs) > 1:
            cmd = [mpi_cmd, mpi_flag, str(num_procs), sys.executable, prog]
        else:
            cmd = [sys.executable, prog]
        
        cmd += ["--test-n", str(matrix_size)]
        
        # Run with timeout
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout_sec
        )
        
        # Parse output
        output = result.stdout + result.stderr
        if "Time:" in output:
            time_str = output.split("Time: ")[1].split("s")[0]
            exec_time = float(time_str)
            return exec_time, output
        else:
            return None, output
            
    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT after {timeout_sec}s"
    except (OSError, subprocess.SubprocessError) as e:
        # Log specific system/subprocess errors to stderr
        print(TerminalFormatter.error(f"System error: {e}"), file=sys.stderr)
        return None, "System error during execution"
    except Exception as e:
        # Securely log unexpected errors and return a generic message
        print(TerminalFormatter.error(f"Unexpected error: {e}"), file=sys.stderr)
        return None, "Internal execution error"


def phase1_gap_analysis():
    """
    FASE 1: GAP ANALYSIS - Identifikasi komunikasi overhead pada small matrices
    
    Strong Scaling: Matrix 512x512
    - 1, 2, 3, 4 Proses: Standard + Hybrid + Optimized
    
    Weak Scaling: Scale matrix dengan proses
    - 1 Proses: 256x256 matrix
    - 2 Proses: 362x362 matrix
    - 3 Proses: 443x443 matrix
    - 4 Proses: 512x512 matrix
    
    Tujuan: Lihat bagaimana strategi komunikasi berbeda mengurangi overhead
    """
    results = []
    
    print(TerminalFormatter.section("FASE 1: GAP ANALYSIS - Communication Overhead (1-4 Processes)"))
    print(f"Tujuan: PROGRESSION dari Standard → Hybrid → Optimized")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    
    # ===== STRONG SCALING PHASE 1 =====
    print(TerminalFormatter.subsection("PHASE 1A: STRONG SCALING - Matrix 512x512 (Fixed Size, Increasing Procs)"))
    
    matrix_512 = 512
    baselines = {'standard': None, 'hybrid': None, 'optimized': None}
    
    for num_procs in [1, 2, 3, 4]:
        print(f"\n{num_procs} Process(es):")
        
        for version in ['standard', 'hybrid', 'optimized']:
            exec_time, _ = run_experiment(f"mpi_programs/matrix_{version}.py", num_procs, matrix_512, timeout_sec=180)
            
            if exec_time:
                # Calculate speedup
                baseline = baselines[version]
                if baseline is None:
                    baselines[version] = exec_time
                    baseline = exec_time
                
                speedup = baseline / exec_time if exec_time > 0 else 0
                results.append(['phase1_strong', version, num_procs, matrix_512, exec_time, speedup])
                
                improvement = ((baseline - exec_time) / baseline * 100) if baseline > 0 else 0
                print(TerminalFormatter.success(f"{version.capitalize()}: {exec_time:.6f}s (Speedup: {speedup:.2f}x)"))
    
    # ===== WEAK SCALING PHASE 1 =====
    print(TerminalFormatter.subsection("PHASE 1B: WEAK SCALING - Scale Matrix Size with Processes"))
    
    weak_configs = [
        {'procs': 1, 'matrix_size': 256, 'label': 'Matrix 256x256'},
        {'procs': 2, 'matrix_size': 362, 'label': 'Matrix 362x362'},
        {'procs': 3, 'matrix_size': 443, 'label': 'Matrix 443x443'},
        {'procs': 4, 'matrix_size': 512, 'label': 'Matrix 512x512'}
    ]
    
    weak_baselines = {'standard': None, 'hybrid': None, 'optimized': None}
    
    for config in weak_configs:
        procs = config['procs']
        matrix_size = config['matrix_size']
        label = config['label']
        
        print(f"\n{procs} Process(es) - {label}:")
        
        for version in ['standard', 'hybrid', 'optimized']:
            exec_time, _ = run_experiment(f"mpi_programs/matrix_{version}.py", procs, matrix_size, timeout_sec=180)
            
            if exec_time:
                if weak_baselines[version] is None:
                    weak_baselines[version] = exec_time
                
                efficiency = (weak_baselines[version] / exec_time * 100) if exec_time > 0 else 0
                results.append(['phase1_weak', version, procs, matrix_size, exec_time, 1.0])
                
                print(TerminalFormatter.success(f"{version.capitalize()}: {exec_time:.6f}s (Efficiency: {efficiency:.1f}%)"))
    
    return results


def phase2_large_scale_analysis():
    """
    FASE 2: LARGE-SCALE ANALYSIS - Hybrid + Optimized pada beban produksi
    
    Strong Scaling: Matrix 4096x4096 (Hybrid + Optimized)
    - 1, 2, 3, 4 Proses: Comparison antara Hybrid dan Optimized
    
    Weak Scaling: Scale matrix (Hybrid + Optimized)
    - 1 Proses: 2048x2048 matrix
    - 2 Proses: 2896x2896 matrix
    - 3 Proses: 3544x3544 matrix
    - 4 Proses: 4096x4096 matrix
    
    Tujuan: Performa riil pada production scale untuk kedua strategi optimized
    """
    results = []
    
    print(TerminalFormatter.section("FASE 2: LARGE-SCALE ANALYSIS - Production Scale (1-4 Processes)"))
    print(f"Tujuan: Analisis performa Hybrid vs Optimized pada beban besar")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    
    # ===== STRONG SCALING PHASE 2 =====
    print(TerminalFormatter.subsection("PHASE 2A: STRONG SCALING - Matrix 4096x4096 (Hybrid vs Optimized)"))
    
    matrix_4096 = 4096
    baselines_2 = {'hybrid': None, 'optimized': None}
    
    for num_procs in [1, 2, 3, 4]:
        print(f"\n{num_procs} Process(es):")
        
        for version in ['hybrid', 'optimized']:
            exec_time, _ = run_experiment(f"mpi_programs/matrix_{version}.py", num_procs, matrix_4096, timeout_sec=600)
            
            if exec_time:
                baseline = baselines_2[version]
                if baseline is None:
                    baselines_2[version] = exec_time
                    baseline = exec_time
                
                speedup = baseline / exec_time if exec_time > 0 else 0
                efficiency = (speedup / num_procs * 100) if num_procs > 0 else 0
                results.append(['phase2_strong', version, num_procs, matrix_4096, exec_time, speedup])
                
                print(TerminalFormatter.success(f"{version.capitalize()}: {exec_time:.6f}s (Speedup: {speedup:.2f}x, Efficiency: {efficiency:.1f}%)"))
    
    # ===== WEAK SCALING PHASE 2 =====
    print(TerminalFormatter.subsection("PHASE 2B: WEAK SCALING - Large Matrices Scaled with Processes (Hybrid vs Optimized)"))
    
    weak_configs_2 = [
        {'procs': 1, 'matrix_size': 2048, 'label': 'Matrix 2048x2048'},
        {'procs': 2, 'matrix_size': 2896, 'label': 'Matrix 2896x2896'},
        {'procs': 3, 'matrix_size': 3544, 'label': 'Matrix 3544x3544'},
        {'procs': 4, 'matrix_size': 4096, 'label': 'Matrix 4096x4096'}
    ]
    
    weak_baselines_2 = {'hybrid': None, 'optimized': None}
    
    for config in weak_configs_2:
        procs = config['procs']
        matrix_size = config['matrix_size']
        label = config['label']
        
        print(f"\n{procs} Process(es) - {label}:")
        
        for version in ['hybrid', 'optimized']:
            exec_time, _ = run_experiment(f"mpi_programs/matrix_{version}.py", procs, matrix_size, timeout_sec=600)
            
            if exec_time:
                if weak_baselines_2[version] is None:
                    weak_baselines_2[version] = exec_time
                
                efficiency = (weak_baselines_2[version] / exec_time * 100) if exec_time > 0 else 0
                results.append(['phase2_weak', version, procs, matrix_size, exec_time, 1.0])
                
                print(TerminalFormatter.success(f"{version.capitalize()}: {exec_time:.6f}s (Efficiency: {efficiency:.1f}%)"))
    
    return results


def save_comprehensive_results(phase1_results, phase2_results):
    """
    Save semua hasil ke satu CSV file untuk plotting
    Format: Phase | Version | Processes | MatrixSize | Time(s) | Speedup
    """
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = phase1_results + phase2_results
    
    csv_path = f'results/experiment_results_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Phase', 'Version', 'Processes', 'MatrixSize', 'Time(s)', 'Speedup'])
        
        for row in all_results:
            writer.writerow(row)
    
    print(TerminalFormatter.section("EXPERIMENT RESULTS SAVED"))
    print(TerminalFormatter.success(f"CSV File: {csv_path}"))
    print(f"\nNext: python scripts/plot_results.py")
    
    return csv_path


def main():
    print(TerminalFormatter.section("MPI MATRIX MULTIPLICATION - COMMUNICATION OPTIMIZATION ANALYSIS"))
    print("Project: Analisis Strong & Weak Scaling dengan Communication Optimization")
    print("Supported Processes: 1, 2, 3, 4")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📌 FASE 1: Standard vs Hybrid vs Optimized (Small Matrices)")
    print("📌 FASE 2: Hybrid vs Optimized (Large Matrices - Production Scale)")
    
    try:
        # Phase 1: Gap Analysis
        phase1_results = phase1_gap_analysis()
        
        # Phase 2: Large-Scale Analysis (with both Hybrid and Optimized)
        phase2_results = phase2_large_scale_analysis()
        
        # Save all results
        csv_file = save_comprehensive_results(phase1_results, phase2_results)
        
        print(TerminalFormatter.section("ALL EXPERIMENTS COMPLETED"))
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nResults: {csv_file}")
        print(f"To visualize: python scripts/plot_results.py {csv_file}")
        
    except KeyboardInterrupt:
        print(TerminalFormatter.warning("Experiments interrupted by user"))
        sys.exit(0)
    except Exception as e:
        print(TerminalFormatter.error(f"Error: {str(e)}"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
