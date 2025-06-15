# src/acquire_data.py

import subprocess
import json
import re
import os

def run_command(command):
    """Runs a shell command and returns its output."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, shell=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error executing command '{command}': {e.stderr.strip()}"

def parse_lscpu(output):
    """Parses the output of lscpu."""
    data = {}
    for line in output.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip()
    # Prioritize the full model name
    return {
        "architecture": data.get("Architecture"),
        "cpu_cores": data.get("CPU(s)"),
        "vendor_id": data.get("Vendor ID"),
        "model_name": data.get("Model name"), # This is the key field
        "mhz": data.get("CPU MHz"),
        "l1d_cache": data.get("L1d cache"),
        "l1i_cache": data.get("L1i cache"),
        "l2_cache": data.get("L2 cache"),
        "l3_cache": data.get("L3 cache"),
    }

def parse_free(output):
    """Parses the output of free -h."""
    lines = output.split('\n')
    if len(lines) > 1:
        headers = lines[0].split()
        values = lines[1].split()
        mem_data = dict(zip(headers, values))
        return {
            "total_memory": mem_data.get("total"),
            "used_memory": mem_data.get("used"),
            "free_memory": mem_data.get("free"),
        }
    return {}

def get_gpu_info():
    """Gets GPU information from lspci."""
    output = run_command("lspci -vnn | grep -i 'vga compatible controller'")
    gpus = []
    for line in output.split('\n'):
        match = re.search(r'\[(.*?)\]', line)
        if match:
            gpus.append(match.group(1))
    return gpus

def main():
    """Acquires system data and saves it to a JSON file."""
    print("--- Acquiring Structured System Profile ---")
    
    profile = {
        "cpu_info": parse_lscpu(run_command("lscpu")),
        "memory_info": parse_free(run_command("free -h")),
        "gpu_info": get_gpu_info(),
        "os_info": run_command("cat /etc/os-release"),
        "kernel_info": run_command("uname -a"),
        "disk_info_df": run_command("df -h"),
        "disk_info_lsblk": run_command("lsblk"),
    }
    
    # Define the output path
    output_dir = "data/system_profile"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "system_profile.json")

    with open(output_path, 'w') as f:
        json.dump(profile, f, indent=4)
        
    print(f"Successfully created system profile at: {output_path}")
    print("--- Data Acquisition Finished ---")

if __name__ == "__main__":
    main()
