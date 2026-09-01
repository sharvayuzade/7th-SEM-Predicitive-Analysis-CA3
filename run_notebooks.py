"""
Execute Forest Fire AI notebooks sequentially by extracting and running code cells.
This is equivalent to running them in Jupyter but scriptable.
"""

import sys
import os
import json
import traceback
from pathlib import Path

NB_DIR = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/Forest Fire task/Implementation/notebooks")

# Notebooks to run in order
NOTEBOOKS = [
    "01_dataset_audit.ipynb",
    "02_image_data_preprocessing.ipynb",
    "03_image_model_benchmarking.ipynb",
    "04_final_image_model_training.ipynb",
    "05_image_model_evaluation_explainability.ipynb",
    "06_tabular_data_analysis.ipynb",
    "07_tabular_model_benchmarking.ipynb",
    "08_final_tabular_model_training.ipynb",
    "09_tabular_model_evaluation.ipynb",
    "10_final_inference_testing.ipynb",
]

def run_notebook(nb_path):
    """Execute all code cells in a notebook."""
    import nbformat
    nb_path = Path(nb_path)
    print(f"\n{'='*70}")
    print(f"EXECUTING: {nb_path.name}")
    print(f"{'='*70}")
    
    with open(nb_path, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Set up a shared namespace for the whole notebook
    ns = {}
    
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        src = cell.source.strip()
        if not src:
            continue
        
        print(f"\n  --- Cell {i+1} ---")
        try:
            exec(compile(src, f"<cell {i+1}>", 'exec'), ns)
        except Exception as e:
            print(f"  [ERROR in cell {i+1}]: {type(e).__name__}: {e}")
            traceback.print_exc()
            # Continue to next cell unless it's a critical failure
            if any(kw in src for kw in ['torch.load', 'pickle.load', 'open(MODEL_DIR']):
                print("  [CRITICAL] Model loading failed — stopping notebook")
                return False
    
    print(f"\n  [OK] {nb_path.name} complete.")
    return True

# Which notebooks to run (can set start index)
START_NB = int(sys.argv[1]) if len(sys.argv) > 1 else 0
END_NB   = int(sys.argv[2]) if len(sys.argv) > 2 else len(NOTEBOOKS)

for nb_name in NOTEBOOKS[START_NB:END_NB]:
    nb_path = NB_DIR / nb_name
    if not nb_path.exists():
        print(f"[SKIP] {nb_name} not found")
        continue
    success = run_notebook(nb_path)
    if not success:
        print(f"[ABORT] {nb_name} failed critically")
        break

print("\n\nAll specified notebooks executed.")
