[SO_accessibility_simulation.py](https://github.com/user-attachments/files/26913970/SO_accessibility_simulation.py)
# SO-accessibility-simulation-2
Minimal simulation for Structural Observability accessibility results
SO_accessibility_demo.py
========================
Reproducible implementation of the accessibility simulation from:
  Dukes, M. "Structural Observability: Threshold Conditions for Quantum 
  Outcomes and an Accessibility-Based Test" (2026), Section XI / Table I.

Circuit: n=6 qubits, brick-wall Clifford+CNOT, depth d=6, random-pairs
States: |psiL> = U|000000>, |psiR> = U|100000>
A(f)   = max over k-subsets of T(rhoL^X, rhoR^X)
Seeds:  20 independent realisations (seeds 0..19 used here; 
        paper's exact seeds not specified — this is a reproducibility gap)

REPRODUCIBILITY STATUS:
  Qualitative structure: reproduced (threshold at f=0.5, bimodal at f=0.333,
                          global TD=1, Hayden-Preskill structure)
  Exact Table I numbers: seed-dependent; paper did not record seeds.
  Seeds 0-19 give 11/9 bimodal split; paper reports 13/7.
  Seeds 100-119 give 13/7 — matching paper exactly.

Requirements: numpy
Python 3.9+
"""

import numpy as np
from itertools import combinations

# ─────────────────────────────────────────────────
# Circuit construction
# ─────────────────────────────────────────────────

# 9 distinct single-qubit Clifford gates used in simulation
CLIFFORD_1Q = [
    np.eye(2, dtype=complex),                                        # I
    np.array([[0,1],[1,0]], dtype=complex),                          # X
    np.array([[1,0],[0,-1]], dtype=complex),                         # Z
    np.array([[0,-1j],[1j,0]], dtype=complex),                       # Y
    np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2),              # H
    np.array([[1,0],[0,1j]], dtype=complex),                         # S
    np.array([[1,0],[0,-1j]], dtype=complex),                        # Sdg
    (np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)) @ np.array([[1,0],[0,1j]],dtype=complex),  # HS
    np.array([[1,0],[0,-1]],dtype=complex) @ np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2),    # ZH
]

def apply_1q(gate, qubit, psi, n):
    """Apply 1-qubit gate to state vector psi."""
    psi = psi.reshape([2]*n)
    psi = np.tensordot(gate, psi, axes=([1],[qubit]))
    psi = np.moveaxis(psi, 0, qubit)
    return psi.reshape(-1)

def apply_cnot(ctrl, tgt, psi, n):
    """Apply CNOT in-place on state vector."""
    dim = 2**n
    psi_out = psi.copy()
    for i in range(dim):
        bits = [(i >> (n-1-q)) & 1 for q in range(n)]
        if bits[ctrl] == 1:
            bits[tgt] ^= 1
            j = sum(b << (n-1-q) for q, b in enumerate(bits))
            psi_out[j], psi_out[i] = psi[i], psi[j]
    return psi_out

def scramble(n, depth, rng):
    """Build brick-wall scrambling circuit state vector (starting from |0...0>)."""
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    for _ in range(depth):
        # Random 1Q Clifford on each qubit
        for q in range(n):
            g = CLIFFORD_1Q[rng.integers(len(CLIFFORD_1Q))]
            psi = apply_1q(g, q, psi, n)
        # Random CNOT pairs
        qubits = rng.permutation(n)
        for i in range(0, n-1, 2):
            psi = apply_cnot(qubits[i], qubits[i+1], psi, n)
    return psi

# ─────────────────────────────────────────────────
# Reduced state and trace distance
# ─────────────────────────────────────────────────

def partial_trace(rho, keep, n):
    """Trace out qubits not in keep list."""
    trace_out = [q for q in range(n) if q not in keep]
    k = len(keep)
    dim_k = 2**k
    dim_t = 2**(n-k)
    rt = rho.reshape([2]*(2*n))
    keep_ket = sorted(keep)
    trace_ket = sorted(trace_out)
    new_order = keep_ket + trace_ket + [q+n for q in keep_ket] + [q+n for q in trace_ket]
    rt = np.transpose(rt, new_order).reshape(dim_k, dim_t, dim_k, dim_t)
    return np.einsum('iaja->ij', rt.reshape(dim_k, dim_t, dim_k, dim_t))

def trace_distance(a, b):
    """0.5 * ||a-b||_1"""
    return 0.5 * np.sum(np.abs(np.linalg.eigvalsh(a - b)))

# ─────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────

def run_simulation(n=6, depth=6, n_seeds=20, seed_start=0):
    print(f"n={n}, depth={depth}, seeds {seed_start}..{seed_start+n_seeds-1}")
    print(f"Circuit: brick-wall, {len(CLIFFORD_1Q)}-element 1Q Clifford set + random-pair CNOTs")
    print()
    
    k_values = list(range(1, n+1))
    all_Amax = {k: [] for k in k_values}
    all_Aavg = {k: [] for k in k_values}
    bimodal = {0:0, 1:0, "other":0}
    global_overlaps, global_tds = [], []

    for seed in range(seed_start, seed_start + n_seeds):
        rng = np.random.default_rng(seed)
        
        # Same scrambling unitary applied to two orthogonal inputs
        # psiL = U|000000>, psiR = U|100000>
        # We achieve this by scrambling |000000> and storing the gates,
        # then applying same gates to |100000>.
        # Simpler: seed the same rng twice with same sequence.
        # We store the gate sequence.
        
        gate_seq = []
        rng2 = np.random.default_rng(seed)
        psiL = np.zeros(2**n, dtype=complex); psiL[0] = 1.0
        psiR = np.zeros(2**n, dtype=complex); psiR[2**(n-1)] = 1.0
        
        for _ in range(depth):
            choices = rng2.integers(len(CLIFFORD_1Q), size=n)
            for q in range(n):
                g = CLIFFORD_1Q[choices[q]]
                psiL = apply_1q(g, q, psiL, n)
                psiR = apply_1q(g, q, psiR, n)
            qubits = rng2.permutation(n)
            for i in range(0, n-1, 2):
                psiL = apply_cnot(qubits[i], qubits[i+1], psiL, n)
                psiR = apply_cnot(qubits[i], qubits[i+1], psiR, n)
        
        overlap = abs(np.dot(psiL.conj(), psiR))
        global_overlaps.append(overlap)
        
        rhoL = np.outer(psiL, psiL.conj())
        rhoR = np.outer(psiR, psiR.conj())
        td_global = trace_distance(rhoL, rhoR)
        global_tds.append(td_global)
        
        for k in k_values:
            vals = []
            for subset in combinations(range(n), k):
                rL = partial_trace(rhoL, list(subset), n)
                rR = partial_trace(rhoR, list(subset), n)
                vals.append(trace_distance(rL, rR))
            all_Amax[k].append(max(vals))
            all_Aavg[k].append(np.mean(vals))
            
            if k == 2:
                am = max(vals)
                if am > 0.9: bimodal[1] += 1
                elif am < 0.1: bimodal[0] += 1
                else: bimodal["other"] += 1
    
    # Report
    print(f"Global overlap:   mean={np.mean(global_overlaps):.2e} ± {np.std(global_overlaps):.2e}")
    print(f"Global TD:        mean={np.mean(global_tds):.6f} ± {np.std(global_tds):.2e}")
    print(f"Paper:            overlap≈1.47×10⁻⁵², TD=1.000000±7.76×10⁻¹⁶")
    print()
    print(f"{'f':>6} {'k':>3} {'Amax':>8} {'±σ':>8} {'Aavg':>8} {'gap':>8}  | Paper")
    print("-"*75)
    paper = {1:(0.100,0.300,0.025,0.075), 2:(0.650,0.477,0.117,0.533),
             3:(1.000,0.000,0.398,0.603), 4:(1.000,0.000,0.763,0.237),
             5:(1.000,0.000,0.933,0.067), 6:(1.000,0.000,1.000,0.000)}
    for k in k_values:
        f = k/n
        Am = np.mean(all_Amax[k]); As = np.std(all_Amax[k])
        Av = np.mean(all_Aavg[k])
        gap = Am - Av
        pA,pS,pAv,pG = paper[k]
        print(f"{f:>6.3f} {k:>3} {Am:>8.3f} {As:>8.3f} {Av:>8.3f} {gap:>8.3f}  | "
              f"{pA:.3f}/{pS:.3f}/{pAv:.3f}/{pG:.3f}")
    print()
    print(f"Bimodal at k=2: Amax≈1:{bimodal[1]}/20, Amax≈0:{bimodal[0]}/20, "
          f"intermediate:{bimodal['other']}/20")
    print(f"Paper claims:   13/20, 7/20, 0/20")

if __name__ == "__main__":
    print("Seeds 0-19 (arbitrary default):")
    run_simulation(seed_start=0)
    print()
    print("Seeds 100-119 (reproduce paper's 13/7 bimodal split):")
    run_simulation(seed_start=100)

