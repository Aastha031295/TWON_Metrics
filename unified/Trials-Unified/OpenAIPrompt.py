import pandas as pd
import asyncio
import sklearn.metrics
from openai import AsyncOpenAI

# Configuration
API_KEY = ""  # Replace with your OpenAI API key
DATA_FILE = "/home/s2shsinh/TWON_Metrics/unified/merged_dataset_19jan.csv"
OUTPUT_FILE = "/home/s2shsinh/TWON_Metrics/unified/predictionsFinal19jan2025.csv"
N_SAMPLES = 15000
MODEL_NAME = "gpt-4o-mini"  # Specify the desired GPT-4 Mini model
BATCH_SIZE = 100

# Load dataset
dataset = pd.read_csv(DATA_FILE)
print(dataset.head())

# Set OpenAI API key
client = AsyncOpenAI(
    api_key=API_KEY,  # This is the default and can be omitted
)

# Define system message
system_message = """
Role:
Role:
You are a content analyzer specializing in detecting fake news, hate speech, and toxicity in German social media posts or tweets. Your job is to classify each tweet into three categories:

1. Fake News Detection:
   - 0: Neutral Post
   - 1: Possible Fake News

2. Hate Speech Classification:
   - 0: Neutral Comment
   - 1: Possible Hate Speech

3. Toxicity Detection:
   - 0: Non-Toxic
   - 1: Possible Toxic

Guidelines:

1. Focus on Text Content:
   - Classify based only on the text content provided.

2. Output Format:
   - Always return classifications for all three categories in this order: Fake News, Hate Speech, Toxicity.
   - Format response as comma-separated numbers, e.g., 0,1,0.
   - Do not include explanations or extra text.

Task-Specific Rules:

1. Fake News:
   - Characteristics:
     - Sensational or exaggerated headlines.
     - Emotional or viral content.
     - Unsupported claims or contradictory statements.
     - Fear-inducing or emotionally charged language.
   - Rule:
     - If any characteristic is present, classify as 1. Otherwise, classify as 0.

2. Hate Speech:
   - Characteristics:
     - Targeting individuals or groups based on identity (e.g., race, religion, gender).
     - Demeaning, abusive, or discriminatory language.
     - Stereotyping or implicit derogatory remarks.
   - Rule:
     - If any characteristic is present, classify as 1. Otherwise, classify as 0.

3. Toxicity:
   - Characteristics:
     - Hostile, offensive, vulgar, or disrespectful language.
     - Rude or belittling remarks.
     - Statements implying harm or aggression.
   - Rule:
     - If any characteristic is present, classify as 1. Otherwise, classify as 0.

Instructions:
- Always provide all three labels in the specified order.
- Use only the required format for output.
"""

# Initialize lists to store results
labels_fake, labels_hate, labels_toxic = [], [], []
preds_fake, preds_hate, preds_toxic = [], [], []

# Helper function to call OpenAI API asynchronously
async def get_prediction_async(text):
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Classify this text: {text}"}
        ],
        temperature=0.3  # To ensure deterministic results
    )
    return response.choices[0].message.content

# Main asynchronous processing function
async def process_dataset():
    counter =0
    # Check for existing output file and resume if possible
    try:
        processed_dataset = pd.read_csv(OUTPUT_FILE)
        start_index = len(processed_dataset)
        dataset_to_process = dataset[start_index:]
        print(f"Resuming from index {start_index}.")
    except FileNotFoundError:
        processed_dataset = pd.DataFrame()
        dataset_to_process = dataset
        print("Starting fresh processing.")

    results = []
    counter =0
    for _, sample in dataset_to_process.iterrows():
        response = await get_prediction_async(sample["text"])

        # Parse the response
        try:
          parsed_response = list(map(int, response.split(",")))
          isFake, isHateSpeech, isToxic = parsed_response
          results.append({
                "text": sample["text"],
                "isFake": sample["isFake"],
                "isHateSpeech": sample["isHateSpeech"],
                "isToxic": sample["isToxic"],
                "predicted_isFake": isFake,
                "predicted_isHateSpeech": isHateSpeech,
                "predicted_isToxic": isToxic
            })
        except Exception as e:
          print(f"Error parsing response: {response}")

        if len(results) >= BATCH_SIZE:
            save_results(results,counter)
            results = []
            counter +=100

    # Save any remaining results
    if results:
        save_results(results)

    # Append labels and predictions, ignoring -1 values
    
# Save results to CSV
def save_results(results,counter):
    new_data = pd.DataFrame(results)
    new_data.to_csv(OUTPUT_FILE, mode='a', header=not pd.io.common.file_exists(OUTPUT_FILE),index=False)
    print(f"Intermediate results saved to {OUTPUT_FILE} and counter {counter}")      

# Filter metrics for valid rows only
def calculate_metrics(labels, preds, category_name):
    if len(labels) > 0:
        print(f"\n{category_name} Metrics:")
        print(sklearn.metrics.classification_report(labels, preds, zero_division=0))
    else:
        print(f"\n{category_name} Metrics: No valid data for evaluation.")

# Save predictions to a CSV file
def save_predictions():
    dataset["predicted_isFake"] = preds_fake 
    dataset["predicted_isHateSpeech"] = preds_hate 
    dataset["predicted_isToxic"] = preds_toxic

    output_file = "/home/s2shsinh/TWON_Metrics/unified/predictionsFinal.csv"
    dataset.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Main entry point
async def main():
    await process_dataset()
    # calculate_metrics(labels_fake, preds_fake, "Fake News")
    # calculate_metrics(labels_hate, preds_hate, "Hate Speech")
    # calculate_metrics(labels_toxic, preds_toxic, "Toxicity")
    # save_predictions()

# Run the asynchronous main function
if __name__ == "__main__":
    asyncio.run(main())
