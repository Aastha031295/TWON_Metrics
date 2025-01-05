import requests

# Define the API endpoint
API_URL = "https://inf.cl.uni-trier.de/chat/"

# List of MetaAI models
META_AI_MODELS = [
    "llama3.1:8b-instruct-q6_K",  
    "llama3.1:70b-instruct-q6_K",
    "llama3.3:70b-instruct-q6_K"
]

# Parameters for the model
CONFIG = {
    'seed': 42,
    'temperature': 0.8,
    'num_predict': 128
}

# Prompt for classification
PROMPT_TEMPLATE = """
You specialize in analyzing social media posts to detect misinformation, hate speech, and toxicity. Each post must be classified into these categories:
- Misinformation: 1 if it contains possible_fake_news, otherwise 0.
- Hate speech: 1 if it contains possible_hate_speech, otherwise 0.
- Toxicity: 1 if it contains possible_toxic, otherwise 0.

Guidelines:
Fake News Characteristics:
- Text with contradictions or logic gaps, emotional language, or unsupported claims.
- Focus on recent topics or sensitive issues with exaggerated content.
- Use of hedging words, manipulated information, or hyperbolic language.

Hate Speech Characteristics:
- Targets individuals/groups based on identity traits like race, religion, or gender.
- May use explicit abuse or subtle stereotypes.
- Includes discriminatory or exclusionary language against protected groups.

Toxicity Characteristics:
- Offensive, aggressive, or vulgar language that demeans or incites harm.
- Disrespectful statements targeting individuals or groups.

Respond in the format: [Misinformation: 0/1], [Hate Speech: 0/1], [Toxicity: 0/1]. Always include all three classifications.

Examples:
- Dieses Land wird von korrupten Politikern regiert, die nur an sich selbst denken!
[Misinformation: 0], [Hate Speech: 0], [Toxicity: 0]

- Alle Wissenschaftler lügen uns an! Klimawandel ist eine Erfindung, um uns zu kontrollieren!
[Misinformation: 1], [Hate Speech: 0], [Toxicity: 0]

- Frauen sind schlechtere Fahrer, sie sollten keine Autos fahren dürfen!
[Misinformation: 0], [Hate Speech: 1], [Toxicity: 0]

- Die Medien verheimlichen die Wahrheit! Impfstoffe sind gefährlich und die Regierung weiß es!
[Misinformation: 1], [Hate Speech: 0], [Toxicity: 1]

- Menschen mit Behinderungen sollten weniger Rechte haben, weil sie der Gesellschaft nichts beitragen.
[Misinformation: 0], [Hate Speech: 1], [Toxicity: 1]
"""

# Function to send a post to the API
def analyze_tweet(tweet_text):
    print(f"Processing tweet: {tweet_text}")
    request_payload = {
        "model": META_AI_MODELS[1],  # Selecting a specific MetaAI model
        "messages": [
            {"role": "system", "content": PROMPT_TEMPLATE},
            {"role": "user", "content": tweet_text}
        ],
        "options": CONFIG
    }

    try:
        # Sending the request
        response = requests.post(API_URL, json=request_payload)
        response_data = response.json()
        return response_data.get("response", "No response received.")
    except Exception as error:
        return f"Error occurred: {error}"

# Function to parse response into binary format
def parse_response_to_binary(response):
    """
    Converts the API response into binary format.
    """
    label_map = {
        "neutral_post": 0,
        "possible_fake_news": 1,
        "neutral_comment": 0,
        "possible_hate_speech": 1,
        "non_toxic": 0,
        "possible_toxic": 1
    }

    try:
        classifications = response.split(", ")
        binary_result = [label_map[label.strip()] for label in classifications]
        return binary_result  # Example output: [1, 0, 0]
    except Exception as e:
        return f"Error in parsing response: {e}"

# Main loop for user interaction
if __name__ == "__main__":
    while True:
        user_input = input("Enter a tweet to classify (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting the program.")
            break
        
        raw_result = analyze_tweet(user_input)
        print("#### Raw Classification Result ####")
        print(raw_result)
        
        binary_result = parse_response_to_binary(raw_result)
        print("#### Binary Classification Result ####")
        print(binary_result)
        print("################################")
