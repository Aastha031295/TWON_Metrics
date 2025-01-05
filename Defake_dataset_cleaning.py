import pandas as pd
import re
import os  # Added for file and directory handling
import unicodedata

# Function to clean text data
def clean_text(text):
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions (e.g., @username) and hashtags (e.g., #hashtag)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to load, clean, and save the dataset
def process_dataset(input_file, output_file):
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load the dataset
    dataset = pd.read_csv(input_file)
    
    # Display initial dataset info for reference
    print("Initial Dataset Info:")
    print(dataset.info())
    print("\nFirst 5 Rows of Dataset:")
    print(dataset.head())
    
    # Clean the 'text' column
    dataset['cleaned_text'] = dataset['text'].apply(clean_text)
    
    # Reorder and keep only necessary columns
    cleaned_dataset = dataset[['cleaned_text', 'binary_label']]
    
    # Save the cleaned dataset to a new CSV file
    print(f"Saving cleaned dataset to: {output_file}")
    cleaned_dataset.to_csv(output_file, index=False)
    print(f"\nCleaned dataset saved to: {output_file}")
    return cleaned_dataset

# File paths
input_file = r'D:\STUDY\ResearchCaseStudy\Dataset_Defakts' # Ensure the file extension is included
output_file = r'D:/STUDY/ResearchCaseStudy/Dataset_Defakts/DefaktS_Twitter_Cleaned.csv'  # Ensure the file extension is included

# Process the dataset
try:
    cleaned_dataset = process_dataset(input_file, output_file)
    
    # Display summary of cleaned dataset
    print("\nCleaned Dataset Info:")
    print(cleaned_dataset.info())
    print("\nFirst 5 Rows of Cleaned Dataset:")
    print(cleaned_dataset.head())
except FileNotFoundError as e:
    print(e)
