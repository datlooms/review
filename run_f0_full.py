import triple_convergence_and_d2ddir as f0

# Operator param for this Stage-8 run: F0 internal pre-gate MIN_PF = 2.0 (trim
# only — keeps more candidates for wf re-scoring; NOT the selection threshold).
# Set as a module-global override so the ratified triple_convergence_and_d2ddir.py
# stays BYTE-IDENTICAL (no source edit). F0's functions read MIN_PF as a module
# global at call time, so this takes effect for the full run.
# f0.MIN_PF override NEUTERED: the scanner's own default is now 2.0, so this assignment
# would restate a value that already holds. Left as a comment rather than deleted so the
# file's history is legible.
# f0.MIN_PF = 2.0

if __name__ == '__main__':
    print(f"F0 full-scope run | MIN_PF override = {f0.MIN_PF} (trim only)")
    f0.main()
