#!/bin/bash
#
# ARK System Information Gathering Script
#
# This script collects comprehensive details about the system's hardware,
# software, and file system, saving it into text files for the AI assistant.
# It should be run from the ~/ARK/src/scripts directory.

echo "Starting system information gathering for ARK..."

# Define the output directory relative to the ARK project root.
OUTPUT_DIR="$HOME/ARK/data/system_info"
echo "Output will be saved to $OUTPUT_DIR"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# --- 1. Hardware Information ---
echo "Gathering hardware information..."
{
    echo "=================================================="
    echo "            HARDWARE INFORMATION"
    echo "=================================================="
    echo ""
    echo "--- CPU Information (lscpu) ---"
    lscpu
    echo ""
    echo "--- Memory Information (free -h) ---"
    free -h
    echo ""
    echo "--- Detailed Hardware List (lshw -short) ---"
    sudo lshw -short
    echo ""
    echo "--- PCI Devices (lspci) ---"
    lspci
    echo ""
    echo "--- USB Devices (lsusb) ---"
    lsusb
    echo ""
} > "$OUTPUT_DIR/01-hardware_info.txt"

# --- 2. Storage Information ---
echo "Gathering storage information..."
{
    echo "=================================================="
    echo "            STORAGE INFORMATION"
    echo "=================================================="
    echo ""
    echo "--- Block Devices (lsblk) ---"
    lsblk -a
    echo ""
    echo "--- Filesystem Disk Space Usage (df -h) ---"
    df -h
    echo ""
    echo "--- Partition Information (parted -l) ---"
    sudo parted -l
    echo ""
} > "$OUTPUT_DIR/02-storage_info.txt"

# --- 3. Software & Network Information ---
echo "Gathering software and network information..."
{
    echo "=================================================="
    echo "        SOFTWARE & NETWORK INFORMATION"
    echo "=================================================="
    echo ""
    echo "--- Kernel and OS Release (uname & os-release) ---"
    uname -a
    cat /etc/os-release
    echo ""
    echo "--- Installed Packages (dpkg list) ---"
    echo "Note: This is a partial list of manually installed packages for brevity."
    grep " install " /var/log/dpkg.log
    echo ""
    echo "For a full list, you could run 'dpkg -l', but it is very long."
    echo ""
    echo "--- Network Interfaces (ip a) ---"
    ip a
    echo ""
    echo "--- Current Network Connections (ss -tuln) ---"
    ss -tuln
    echo ""
} > "$OUTPUT_DIR/03-software_network_info.txt"

echo "---------------------------------------------------"
echo "System information gathering complete."
echo "Files created in $OUTPUT_DIR:"
ls -l "$OUTPUT_DIR"
echo "---------------------------------------------------"
