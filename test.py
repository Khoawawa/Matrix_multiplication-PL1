import numpy as np

# --- Utility function ---
def load_matrix(filename):
    print(f"Loading {filename} ...")
    return np.loadtxt(filename)

# --- Load matrices ---
A = load_matrix("A.txt")
B = load_matrix("B.txt")
C_naive = load_matrix("C.txt")
C_strassen_free = load_matrix("C_Strassen_Free.txt")
C_strassen_class = load_matrix("C_Strassen_Class.txt")

# --- Compute reference result ---
print("Computing A @ B using NumPy ...")
C_ref = np.dot(A, B)

# --- Comparison function ---
def compare_matrices(M1, M2, name1, name2, tol=1e-6):
    if np.allclose(M1, M2, atol=tol):
        print(f"{name1} vs {name2}: ✅ MATCH")
    else:
        print(f"{name1} vs {name2}: ❌ MISMATCH")
        diff = np.abs(M1 - M2)
        max_diff = np.max(diff)
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  -> Max difference: {max_diff:.6f} at {idx}")
        print(f"  -> {name1}[{idx}] = {M1[idx]}, {name2}[{idx}] = {M2[idx]}")

# --- Verify all ---
print("\n--- Verification ---")
compare_matrices(C_ref, C_naive, "NumPy", "Naive")
compare_matrices(C_ref, C_strassen_free, "NumPy", "Strassen_Free")
compare_matrices(C_ref, C_strassen_class, "NumPy", "Strassen_Class")

# Optional: print small differences summary
print("\nAll checks done.")
