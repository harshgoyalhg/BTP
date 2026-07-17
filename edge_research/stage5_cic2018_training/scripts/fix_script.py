with open('evaluate_no_portscan.py', 'r') as f:
    lines = f.readlines()

with open('evaluate_no_portscan.py', 'w') as f:
    for line in lines:
        if 'lycos' in line and 'load' in line and 'npy' in line:
            f.write('    data_lycos = np.load(os.path.join(BASE_DIR, "artifacts", "lycos_processed.npz"))\n')
            f.write('    X_ly = data_lycos["X_test"].astype(np.float32)\n')
            f.write('    y_ly = data_lycos["y_test"].astype(np.int64)\n')
        elif 'y_ly = np.load' in line:
            pass # skip it
        else:
            f.write(line)
