#!/usr/bin/env python3

import sys
import os
import platform

def test_windows_hdbscan_solutions():
    """Test and provide guidance for Windows hdbscan build issues"""
    print("=== Windows hdbscan Build Issue Diagnostic ===")
    
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    if platform.system() != "Windows":
        print("ℹ️  This diagnostic is for Windows systems only")
        print("✅ On Linux/macOS, use: sudo apt-get install gcc python3-dev")
        return True
    
    print("\n=== Checking Visual Studio Build Tools ===")
    
    vs_paths = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools",
        r"C:\Program Files\Microsoft Visual Studio\2019\Community"
    ]
    
    vs_found = False
    for path in vs_paths:
        if os.path.exists(path):
            print(f"✅ Found Visual Studio at: {path}")
            vs_found = True
        else:
            print(f"❌ Not found: {path}")
    
    if not vs_found:
        print("\n🔧 SOLUTION: Install Visual Studio Build Tools")
        print("1. Download from: https://visualstudio.microsoft.com/downloads/")
        print("2. Select 'C++ build tools' workload")
        print("3. Include 'Windows 10/11 SDK'")
        return False
    
    print("\n=== Checking Windows SDK ===")
    
    sdk_paths = [
        r"C:\Program Files (x86)\Windows Kits\10\Include",
        r"C:\Program Files\Windows Kits\10\Include"
    ]
    
    sdk_found = False
    for path in sdk_paths:
        if os.path.exists(path):
            print(f"✅ Found Windows SDK at: {path}")
            for subdir in os.listdir(path):
                io_path = os.path.join(path, subdir, "ucrt", "io.h")
                if os.path.exists(io_path):
                    print(f"✅ Found io.h at: {io_path}")
                    sdk_found = True
                    break
    
    if not sdk_found:
        print("\n🔧 SOLUTION: Install Windows SDK")
        print("1. Open Visual Studio Installer")
        print("2. Modify your installation")
        print("3. Add 'Windows 10/11 SDK' component")
        print("4. Restart command prompt")
        return False
    
    print("\n=== Alternative Solutions ===")
    print("If build tools are installed but still failing:")
    print("1. Use conda: conda install -c conda-forge hdbscan")
    print("2. Use minimal requirements: pip install -r requirements-minimal.txt")
    print("3. Use Developer Command Prompt for VS 2022")
    
    return True

if __name__ == "__main__":
    success = test_windows_hdbscan_solutions()
    if success:
        print("\n✅ Windows build environment appears configured")
    else:
        print("\n❌ Windows build environment needs configuration")
        print("See solutions above or use conda/minimal requirements")
