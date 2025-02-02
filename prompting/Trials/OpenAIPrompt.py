import pandas as pd
import asyncio
import sklearn.metrics
from openai import AsyncOpenAI

# Configuration
API_KEY = ""  # Replace with your OpenAI API key
DATA_FILE = "/Users/shivangsinha/Downloads/ROCMetrics/balanced_dataset.csv"
OUTPUT_FILE = "/Users/shivangsinha/Downloads/ROCMetrics/predictionsFinal25jan2025-test2.csv"
N_SAMPLES = 6000
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

1.Fake News Detection:

- Characteristics of Fake News include:

- Sensational or exaggerated headlines.

- Lack of specific details or logical reasoning.

- Emotionally charged content designed to go viral.

- Contradictory statements or claims unsupported by evidence.

- Language designed to incite fear, panic, or strong emotions.

- Classification Rule:

- If the tweet exhibits one or more of these characteristics, classify it as ⁠possible_fake_news (1)⁠.

- Otherwise, classify it as ⁠neutral_post (0)⁠.

2.⁠ ⁠Hate Speech Classification:

- Characteristics of Hate Speech include:

- Targeting individuals or groups based on protected characteristics (e.g., race, religion, gender).

- Language that demeans, harms, or perpetuates discrimination.

- Overt abuse or implicit stereotyping and insinuations.

- Classification Rule:

- If the tweet exhibits one or more of these characteristics, classify it as ⁠ possible_hate_speech (1) ⁠.

- Otherwise, classify it as ⁠ neutral_comment (0) ⁠.

3.⁠ ⁠Toxicity Detection:

- Characteristics of Toxic Language include:

- Offensive, disrespectful, or hostile language.

- Use of vulgar or obscene expressions.

- Statements implying harm or aggression.

- Rude or belittling remarks that may not explicitly target a group.

- Classification Rule:

- If the tweet exhibits one or more of these characteristics, classify it as ⁠ possible_toxic (1) ⁠.

- Otherwise, classify it as ⁠ non_toxic (0) ⁠.

Examples

Fake News Detection:

•⁠⁠Text: "Die NASA bestätigt, dass der Mond im Jahr 2025 zum ersten Mal grün sein wird."

Label: ⁠ 1 ⁠ (Fake News)

•⁠ ⁠Text: "Die Weltgesundheitsorganisation hat neue Leitlinien für die psychische Gesundheit veröffentlicht."

Label: ⁠ 0 ⁠ (Not Fake News)

Hate Speech Classification:

•⁠ ⁠Text: "Frauen sind faul und leisten keinen Beitrag zur Gesellschaft."

Label: ⁠ 1 ⁠ (Hate Speech)

•⁠ ⁠Text: "Wir müssen zusammenarbeiten, um eine bessere Zukunft zu schaffen."

Label: ⁠ 0 ⁠ (Not Hate Speech)

Toxicity Detection:

•⁠ ⁠Text: "Du bist der schlimmste Mensch, den ich je getroffen habe. Ich hoffe, du versagst bei allem."

Label: ⁠ 1 ⁠ (Toxic)

•⁠ ⁠Text: "Ich danke Ihnen für Ihre Hilfe. Ich weiß das wirklich zu schätzen."

Label: ⁠ 0 ⁠ (Not Toxic)

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
          print("counter = " ,counter)
          counter+=1
        except Exception as e:
          print(f"Error parsing response: {response}")
          results.append({
                "text": sample["text"],
                "isFake": sample["isFake"],
                "isHateSpeech": sample["isHateSpeech"],
                "isToxic": sample["isToxic"],
                "predicted_isFake": -1,
                "predicted_isHateSpeech": -1,
                "predicted_isToxic": -1
            })

        if len(results) >= BATCH_SIZE:
            save_results(results,counter)
            results = []
            counter +=1

    # Save any remaining results
    if results:
        save_results(results,counter)

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
