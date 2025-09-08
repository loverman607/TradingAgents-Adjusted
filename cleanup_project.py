#!/usr/bin/env python3
"""
Project Cleanup Script

This script organizes the project by:
1. Moving files to appropriate subfolders
2. Removing unnecessary files
3. Ensuring clean project structure
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """Clean up and organize the project structure"""
    
    print("🧹 Starting project cleanup and organization...")
    
    # Create organized directory structure
    directories = [
        "reports/analysis",
        "reports/charts", 
        "reports/backtests",
        "logs",
        "scripts",
        "docs",
        "demos",
        "configs",
        "dataflows/data_cache"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except FileExistsError:
            print(f"✅ Directory already exists: {directory}")
    
    # Files to remove (unnecessary)
    files_to_remove = [
        "test_*.py",
        "demo_*.py", 
        "launch.py",
        "FIXED_SUMMARY.md",
        "ORGANIZATION_SUMMARY.md",
        "crypto_backtest_simple_*.json",
        "crypto_backtest_simple.log"
    ]
    
    print("\n🗑️ Removing unnecessary files...")
    for pattern in files_to_remove:
        if "*" in pattern:
            # Handle wildcard patterns
            import glob
            for file in glob.glob(pattern):
                if os.path.exists(file):
                    os.remove(file)
                    print(f"  ❌ Removed: {file}")
        else:
            if os.path.exists(pattern):
                os.remove(pattern)
                print(f"  ❌ Removed: {pattern}")
    
    # Move documentation files to docs/
    docs_to_move = [
        "ARCHITECTURE_ANALYSIS.md",
        "CRYPTO_24_7_OPTIMIZATION.md", 
        "SYSTEM_OPTIMIZATION_SUMMARY.md",
        "ORGANIZATION_SUMMARY.md"
    ]
    
    print("\n📚 Moving documentation to docs/...")
    for doc in docs_to_move:
        if os.path.exists(doc):
            shutil.move(doc, "docs/")
            print(f"  📄 Moved: {doc} → docs/")
    
    # Move script files to scripts/
    scripts_to_move = [
        "main_enterprise.py",
        "main_optimized.py",
        "run_trading_system.py"
    ]
    
    print("\n🔧 Moving scripts to scripts/...")
    for script in scripts_to_move:
        if os.path.exists(script):
            shutil.move(script, "scripts/")
            print(f"  🔧 Moved: {script} → scripts/")
    
    # Move chart files to reports/charts/
    charts_to_move = [
        "crypto_backtest_charts"
    ]
    
    print("\n📊 Moving charts to reports/charts/...")
    for chart in charts_to_move:
        if os.path.exists(chart):
            shutil.move(chart, "reports/")
            print(f"  📊 Moved: {chart} → reports/")
    
    # Clean up __pycache__ directories
    print("\n🧹 Cleaning up __pycache__ directories...")
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                shutil.rmtree(pycache_path)
                print(f"  🗑️ Removed: {pycache_path}")
    
    print("\n✅ Project cleanup completed!")
    print("\n📁 Organized structure:")
    print("  📄 main.py - Main trading system")
    print("  📁 reports/ - All generated reports")
    print("  📁 reports/analysis/ - Trading analysis reports")
    print("  📁 reports/charts/ - Charts and visualizations")
    print("  📁 reports/backtests/ - Backtest results")
    print("  📁 scripts/ - Alternative scripts")
    print("  📁 docs/ - Documentation")
    print("  📁 demos/ - Demo files")
    print("  📁 configs/ - Configuration files")
    print("  📁 logs/ - Log files")
    print("  📁 tradingagents/ - Core system code")

if __name__ == "__main__":
    cleanup_project()
