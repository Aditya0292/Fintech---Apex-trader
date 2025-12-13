import subprocess
import os
import sys

def run_command(command_list):
    print(f"Running: {' '.join(command_list)}")
    # Use shell=False (default) and pass list to avoid quoting issues with empty strings on Windows
    res = subprocess.run(command_list, shell=False)
    if res.returncode != 0:
        print(f"Command failed: {command_list}")
        return False
    return True

def train_intraday():
    tasks = [
        {"name": "30 Min", "data": "data/XAUUSD_30m.csv", "suffix": "_30m"},
        {"name": "15 Min", "data": "data/XAUUSD_15m.csv", "suffix": "_15m"},
    ]
    
    for task in tasks:
        print(f"\n[{task['name']}] Processing...")
        
        # Check Data Existence
        if not os.path.exists(task['data']):
            print(f"  Warning: Data file {task['data']} not found. Skipping training for {task['name']}.")
            continue
            
        # 1. Feature Engineering
        print(f"  > Generating Features for {task['name']}...")
        cmd_feat = [sys.executable, "src/features/feature_pipeline.py", "--data", task['data'], "--suffix", task['suffix']]
        if not run_command(cmd_feat): continue
        
        # 2. Train Model
        print(f"  > Training Ensemble for {task['name']}...")
        cmd_train = [sys.executable, "src/models/train_ensemble.py", "--suffix", task['suffix']]
        if not run_command(cmd_train): continue
        
        print(f"✅ {task['name']} Training Complete.")

if __name__ == "__main__":
    train_intraday()
