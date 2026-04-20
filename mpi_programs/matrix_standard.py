"""
Standard Matrix Multiplication with MPI (Vectorized Baseline)
Uses np.dot() for computation while maintaining standard communication
Significantly faster than triple nested loop
"""
import os
import numpy as np
import sys
import argparse
import time

try:
    from mpi4py import MPI
except Exception as e:
    print("ERROR: mpi4py is not installed or failed to import.")
    print("Install mpi4py and an MPI runtime (e.g., MS-MPI/OpenMPI), then re-run.")
    print("Import error:", e)
    sys.exit(1)

def matrix_multiply_standard(A, B, rank, size, comm):
    """
    STANDARD approach with Vectorized Computation:
    - Uses NumPy vectorization (np.dot)
    - Communication: Full B matrix broadcast to all processes
    """
    n = A.shape[0]
    
    # Distribute rows of A to each process
    rows_per_process = n // size
    start_row = rank * rows_per_process
    end_row = start_row + rows_per_process
    
    local_A = A[start_row:end_row, :]
    
    broadcast_B = comm.bcast(B, root=0)
    
    # Optimized with NumPy vectorization (np.dot)
    local_C = np.dot(local_A, broadcast_B)
    
    # Gather results
    C = None
    if rank == 0:
        C = np.zeros((n, n), dtype=np.float32)
    
    comm.Gatherv(
        [local_C, MPI.FLOAT],
        [C, ([rows_per_process * n] * size, [i * rows_per_process * n for i in range(size)]), MPI.FLOAT] if rank == 0 else None,
        root=0
    )
    
    return C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-n', type=int, default=None, help='Override matrix size for testing')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.test_n is not None:
        n = args.test_n
    else:
        n = int(os.environ.get('TEST_N', '4096'))
    
    if rank == 0:
        A = np.random.rand(n, n).astype(np.float32)
        B = np.random.rand(n, n).astype(np.float32)
    else:
        A = None
        B = None
    
    # Broadcast matrices from rank 0 to all processes
    A = comm.bcast(A, root=0)
    B = comm.bcast(B, root=0)
    
    comm.Barrier()
    start_time = MPI.Wtime()
    C = matrix_multiply_standard(A, B, rank, size, comm)
    comm.Barrier()
    end_time = MPI.Wtime()
    
    if rank == 0:
        elapsed_time = end_time - start_time
        print(f"Standard (Vectorized) - Processes: {size}, Matrix: {n}x{n}, Time: {elapsed_time:.4f}s")
        
        # Optional: Verify correctness (expensive for large matrices)
        if n <= 512:
            C_reference = A @ B
            error = np.linalg.norm(C - C_reference) / np.linalg.norm(C_reference)
            print(f"Relative Error: {error:.2e}")


if __name__ == "__main__":
    main()
