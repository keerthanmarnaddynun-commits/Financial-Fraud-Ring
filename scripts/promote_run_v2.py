import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path

RUN_DIR = Path('runs/20261117_162714_c1-full-v2')
OUTPUTS_DIR = Path('outputs')
ARCHIVE_DIR = Path('archive/outputs_prev_20261117_162714')
RESULTS_SNAPSHOT_DIR = Path('results_snapshot')

def on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"Failed to clear {path}: {e}")

# 1. Archive current outputs (if not already done)
if OUTPUTS_DIR.exists():
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for item in OUTPUTS_DIR.iterdir():
        dest = ARCHIVE_DIR / item.name
        if not dest.exists():
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
    print(f"Archived current outputs to {ARCHIVE_DIR}")

# 2. Make all files in outputs writable before cleaning
if OUTPUTS_DIR.exists():
    for root, dirs, files in os.walk(OUTPUTS_DIR):
        for f in files:
            p = Path(root) / f
            try:
                os.chmod(p, stat.S_IWRITE)
            except Exception:
                pass
        for d in dirs:
            p = Path(root) / d
            try:
                os.chmod(p, stat.S_IWRITE)
            except Exception:
                pass
    shutil.rmtree(OUTPUTS_DIR, onerror=on_rm_error)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# 3. Copy from run folder to outputs
def copy_and_make_writable(src, dst):
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            copy_and_make_writable(item, dst / item.name)
    else:
        shutil.copyfile(src, dst)
        os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)

for item in RUN_DIR.iterdir():
    copy_and_make_writable(item, OUTPUTS_DIR / item.name)
print(f"Copied {RUN_DIR} to {OUTPUTS_DIR}")

# 4. Write PROVENANCE.txt
prov_text = f"""Run Folder: runs/20261117_162714_c1-full-v2
Commit Hash: 81c9ad4490bcefa60619177cc21c2f4157eadedf
Timestamp: {datetime.now(timezone.utc).isoformat()}
Dataset Path: data/PS_20174392719_1491204439457_log.csv
Dataset SHA256: 16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b
"""
with open(OUTPUTS_DIR / 'PROVENANCE.txt', 'w') as f:
    f.write(prov_text)
print("PROVENANCE.txt written.")

# 5. Make results_snapshot writable before copying
if RESULTS_SNAPSHOT_DIR.exists():
    for root, dirs, files in os.walk(RESULTS_SNAPSHOT_DIR):
        for f in files:
            p = Path(root) / f
            try:
                os.chmod(p, stat.S_IWRITE)
            except Exception:
                pass

RESULTS_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
for item_name in ['data_quality.json', 'evaluation.json', 'scalability.json', 'run_info.json', 'PROVENANCE.txt']:
    p = OUTPUTS_DIR / item_name
    if p.exists():
        dst = RESULTS_SNAPSHOT_DIR / item_name
        shutil.copyfile(p, dst)
        os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)
print(f"Copied core run artifacts to {RESULTS_SNAPSHOT_DIR}")
