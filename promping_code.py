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
- For misinformation: neutral_post or possible_fake_news
- For hate speech: neutral_comment or possible_hate_speech
- For toxicity: non_toxic or possible_toxic

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

Respond in the format: [neutral_post/possible_fake_news], [neutral_comment/possible_hate_speech], [non_toxic/possible_toxic]. Always include all three classifications.

Examples:
- Dieses Land wird von korrupten Politikern regiert, die nur an sich selbst denken!
neutral_post, neutral_comment, non_toxic

- Alle Wissenschaftler lügen uns an! Klimawandel ist eine Erfindung, um uns zu kontrollieren!
possible_fake_news, neutral_comment, non_toxic

- Frauen sind schlechtere Fahrer, sie sollten keine Autos fahren dürfen!
neutral_post, possible_hate_speech, non_toxic

- Die Medien verheimlichen die Wahrheit! Impfstoffe sind gefährlich und die Regierung weiß es!
possible_fake_news, neutral_comment, possible_toxic

- Menschen mit Behinderungen sollten weniger Rechte haben, weil sie der Gesellschaft nichts beitragen.
neutral_post, possible_hate_speech, possible_toxic

- Die Regierung unterstützt heimlich eine Organisation, die alle unsere Daten verkauft. Wehrt euch!
possible_fake_news, neutral_comment, non_toxic

- Muslime sind eine Gefahr für die Gesellschaft. Wir sollten uns gegen sie verteidigen!
possible_fake_news, possible_hate_speech, possible_toxic

- Transmenschen sind einfach verwirrt. Es gibt nur Männer und Frauen, alles andere ist Unsinn.
neutral_post, possible_hate_speech, non_toxic

- Alle Milliardäre haben sich verschworen, um die Weltbevölkerung zu reduzieren!
possible_fake_news, neutral_comment, non_toxic

- Diese Gruppe verdient keine Rechte, sie sind minderwertig und sollten ausgeschlossen werden.
neutral_post, possible_hate_speech, possible_toxic

- Hunde sind bessere Haustiere als Katzen. Wer mag schon Katzen?
neutral_post, neutral_comment, non_toxic

- Die Wahlen sind manipuliert worden, das Ergebnis ist nicht rechtmäßig!
possible_fake_news, neutral_comment, non_toxic

- Diese Regierung tut nichts für uns! Wir müssen endlich Widerstand leisten!
neutral_post, neutral_comment, non_toxic

- Alle Juden kontrollieren die Weltwirtschaft, und niemand redet darüber!
possible_fake_news, possible_hate_speech, possible_toxic

- Flüchtlinge kommen nur hierher, um Sozialhilfe zu kassieren. Das muss gestoppt werden!
possible_fake_news, possible_hate_speech, possible_toxic
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

# Main loop for user interaction
if __name__ == "__main__":
    while True:
        user_input = input("Enter a tweet to classify (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting the program.")
            break
        
        result = analyze_tweet(user_input)
        print("#### Classification Result ####")
        print(result)
        print("################################")
