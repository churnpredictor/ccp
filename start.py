"""
Quick Start Script for Customer Churn Prediction System
This script provides a guided setup and running experience
"""

import os
import sys

def print_banner():
    print("=" * 80)
    print("🎯 SMART CUSTOMER CHURN PREDICTION SYSTEM")
    print("=" * 80)
    print()

def check_dataset():
    """Check if dataset exists"""
    dataset_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    if os.path.exists(dataset_path):
        print("✅ Dataset found!")
        return True
    else:
        print("❌ Dataset not found!")
        print("\nPlease download the dataset:")
        print("1. Visit: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("2. Download 'WA_Fn-UseC_-Telco-Customer-Churn.csv'")
        print("3. Place it in the 'data/' folder")
        print()
        return False

def check_models():
    """Check if models are trained"""
    model_files = [
        'models/logistic_regression.pkl',
        'models/decision_tree.pkl',
        'models/random_forest.pkl',
        'models/xgboost.pkl'
    ]
    
    all_exist = all(os.path.exists(f) for f in model_files)
    
    if all_exist:
        print("✅ All models trained!")
        return True
    else:
        print("❌ Models not trained yet!")
        return False

def main():
    print_banner()
    
    print("STEP 1: Checking Requirements")
    print("-" * 80)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✅ Python version: {python_version.major}.{python_version.minor}")
    else:
        print(f"⚠️ Python version: {python_version.major}.{python_version.minor}")
        print("    Recommended: Python 3.8 or higher")
    
    print()
    
    # Check dataset
    print("STEP 2: Checking Dataset")
    print("-" * 80)
    dataset_exists = check_dataset()
    print()
    
    if not dataset_exists:
        print("Please download the dataset first, then run this script again.")
        return
    
    # Check if models are trained
    print("STEP 3: Checking Models")
    print("-" * 80)
    models_exist = check_models()
    print()
    
    if not models_exist:
        print("Models need to be trained first.")
        response = input("Do you want to train models now? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            print("\n🚀 Starting model training...")
            print("This may take 2-5 minutes depending on your system.\n")
            os.system('python train.py')
            print("\n✅ Model training complete!")
        else:
            print("\nTo train models manually, run: python train.py")
            return
    
    print()
    print("=" * 80)
    print("✅ ALL CHECKS PASSED!")
    print("=" * 80)
    print()
    
    # Ask to run the app
    response = input("Do you want to start the Streamlit app now? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\n🚀 Starting Streamlit application...")
        print("The app will open in your default browser.\n")
        os.system('streamlit run app.py')
    else:
        print("\n📝 To start the app manually, run: streamlit run app.py")
        print("\n🎓 For viva preparation, see: VIVA_NOTES.md")
        print("📚 For full documentation, see: README.md")

if __name__ == "__main__":
    main()
