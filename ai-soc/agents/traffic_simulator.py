import time
import requests
import pandas as pd
import numpy as np
import glob
import os
import argparse
import random

# Default endpoint
API_URL = "http://localhost:8000/predict"

def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    pattern = os.path.join(data_dir, "Friday-WorkingHours-*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("Error: Could not find dataset files in data/ directory.")
        return None
        
    print("Loading datasets into memory for simulation...")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        
    data = pd.concat(dfs, ignore_index=True)
    
    # Clean up column names
    data.columns = data.columns.str.strip()
    
    # Identify label column
    label_col = 'Label' if 'Label' in data.columns else [c for c in data.columns if 'label' in c.lower()][0]
    
    # Clean data just like preprocessing (drop specific cols, handle inf/nan)
    cols_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
    actual_cols_to_drop = [c for c in cols_to_drop if c in data.columns]
    data.drop(columns=actual_cols_to_drop, inplace=True)
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    
    return data, label_col

def simulate_traffic(data, label_col, attack_type, speed=1.0):
    """
    attack_type: 'normal', 'ddos', 'portscan', 'bot'
    speed: delay in seconds between requests
    """
    # Map friendly names to exact dataset labels
    label_mapping = {
        'normal': 'BENIGN',
        'ddos': 'DDoS',
        'portscan': 'PortScan',
        'bruteforce': 'Bot', # We use Bot as a stand-in for BruteForce in the Friday dataset
        'bot': 'Bot'
    }
    
    if attack_type.lower() == 'mix':
        print("Starting simulation for MIXED traffic (All available rows)")
        filtered_data = data
        target_label = "MIXED"
    else:
        target_label = label_mapping.get(attack_type.lower())
        if not target_label:
            print(f"Unknown attack type: {attack_type}. Defaulting to BENIGN.")
            target_label = 'BENIGN'
            
        # Filter dataset for the specific traffic type
        filtered_data = data[data[label_col] == target_label]
    
    if filtered_data.empty:
        print(f"No data found for label {target_label}!")
        return
        
    print(f"Starting simulation for {target_label} traffic ({len(filtered_data)} available rows)")
    print(f"Sending to {API_URL} every {speed} seconds...")
    print("Press Ctrl+C to stop.\n")
    
    # Drop the label column so we only send the 80 features
    features_only = filtered_data.drop(columns=[label_col])
    
    # Convert to list of lists for easy sending
    records = features_only.values.tolist()
    
    # Shuffle so we don't always start at the same row
    random.shuffle(records)
    
    try:
        for i, row in enumerate(records):
            payload = {
                "features": row
            }
            
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    # Print beautiful output
                    color = "\033[92m" if result['attack_name'] == 'BENIGN' else "\033[91m"
                    reset = "\033[0m"
                    print(f"[{i+1}] Sent flow -> AI Prediction: {color}{result['attack_name']}{reset} "
                          f"(Confidence: {result['confidence_score']*100:.1f}%)")
                else:
                    print(f"[{i+1}] Error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                print(f"[{i+1}] Failed to connect to {API_URL}. Is FastAPI running?")
                time.sleep(2) # Wait a bit before trying again
                
            time.sleep(speed)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI SOC Traffic Simulator")
    parser.add_argument("--type", type=str, default="mix", 
                        choices=['normal', 'ddos', 'portscan', 'bruteforce', 'bot', 'mix'],
                        help="Type of traffic to simulate (normal, ddos, portscan, bruteforce, mix)")
    parser.add_argument("--speed", type=float, default=1.0, 
                        help="Delay in seconds between requests (default: 1.0)")
    
    args = parser.parse_args()
    
    data, label_col = load_data()
    if data is not None:
        simulate_traffic(data, label_col, args.type, args.speed)
