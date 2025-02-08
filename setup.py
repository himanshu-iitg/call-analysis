import subprocess
import sys

def download_nltk_dependencies():
    """Download required NLTK data and install required packages from requirements.txt."""
    # Install packages from requirements.txt
    print("Installing required packages from requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    # Download NLTK data
    required_packages = [
        'punkt',          # For tokenization
        'averaged_perceptron_tagger',  # For POS tagging
        'wordnet',        # For lemmatization
        'stopwords'       # For stopwords
    ]
    
    print("Downloading NLTK data...")
    import nltk
    for package in required_packages:
        nltk.download(package)

if __name__ == "__main__":
    download_nltk_dependencies()