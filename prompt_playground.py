import requests

endpoint: str = "https://inf.cl.uni-trier.de/chat/"

MODELS = [
    # LLama (MetaAI)
    "llama3.1:8b-instruct-q6_K",
    "llama3.1:70b-instruct-q6_K",
    "llama3.3:70b-instruct-q6_K",
    # Mi(s/x)tral (Mistral AI)
    "mistral:7b-instruct-v0.2-q6_K",
    "mixtral:8x7b-instruct-v0.1-q6_K",
    # Phi (Mircosoft)
    "phi3:14b-medium-128k-instruct-q6_K",
    "phi3.5:3.8b-mini-instruct-q6_K",
    # Gemma (Google)
    "gemma:7b-instruct-q6_K",
    "gemma2:27b-instruct-q6_K",
    # QWEN (Alibaba)
    "qwen2:72b-instruct-q6_K",
]

options = {'seed': 42, 'temperature': 0.8, 'num_predict': 128}

PROMPT = """
You are a specialized content analyzer focused on identifying potential misinformation, hate speech and toxicity in social media posts. Your task is to classify tweets into two categories of each:
 neutral_post or possible_fake_news, neutral_comment or possible_hate_speech and non_toxic or possible_toxic.

Fake News Characteristics: 

- Disinformation exhibits a higher degree of contentual inconsistencies like semantic contradictions or logic errors throughout the text.
- The body of unreliable articles adds relatively little new information, but serves to repeat and enhance the claims made at the beginning.
- Unreliable articles frequently narrate in terms of a clear friend-foe-distinction with regard to specific national, ethical, or religious groups or elites as foes or perpetrators. The opposing group (often framed in a common "we", "ourselves", "the government") takes the part of the victim who needs to be protected. 
- Unreliable sources incline to use a more emotionally persuasive language and touch more often sensible subjects (like children, death and burial).
- Fake articles tend to be written in a hyperbolic way to attract the reader's attention, i.e. with a high usage of all-caps-words, exclamation marks or a general sentiment wording.
- Legitimate sources tend to report about past events whereas fake articles focus on highly recent topics.
- Fake articles use a higher amount of hedging words (like 'possibly', 'usually', 'tend to be') to achieve a more indirect form of expression. Also they evoke a feeling of uncertainty by addressing the vagueness of information directly. 
- Content that calls on supposedly scientific research or reputable institutions without identifying concrete sources or by manipulating them to create a false theory.
- Stories that lack any factual ground or manipulated information or image. The intention is to deceive and cause harm. Could be text or visual media. 
- Real information is being presented in a false context. The recipient is aware that the information is true, but he does not realize that the context has been changed.
- Stories without factual basis which usually explain important events as secret plots by government or powerful individuals. By definition their truthfulness is difficult to verify. Evidence refuting the conspiracy is regarded as further proof of the conspiracy.
- Information that is created by a political entity to influence public opinion and gain support for a public figure, organization or government.
- Posts that are pure opinion, comics, satire, or any other posts that do not make a factual claim. This is also the category to use for posts that are of the "Like this if you think..." variety.

If the tweets exhibits one of the above described charactericts classify it possible_fake_news else classify it as neutral_post. 

Hate Speech Characteristics:
-Hate speech targets individuals or groups explicitly or implicitly.
-The focus is often on their identity traits, societal roles, or affiliations.
-Hate speech is identified when directed against marginalized groups based on specific traits such as:
Race, Religion, Gender, Sexual Orientation, Disability and Socioeconomic Status (distinctly includes poor people as a protected group, influenced by contemporary discussions)
-Hate speech definitions and examples are shaped by cultural nuances, especially relevant in the German-speaking context (Germany, Austria, Switzerland)
-It can be overtly abusive or manifest through subtler, more implicit expressions (e.g., stereotypes or insinuations).
-Language that aims to demean, harm, exclude, or perpetuate discrimination against protected groups or individuals.

If the tweets exhibits one of the above described charactericts classify it possible_hate_speech else classify it as neutral_comment.

Toxicity Characteristics:
-Use of language that is rude, disrespectful, or likely to offend individuals or groups.
-Expressions that demean or incite violence against individuals based on attributes such as race, religion, ethnic origin, sexual orientation, disability, or gender.
-Direct derogatory statements aimed at individuals, intending to belittle or harm.
-highly offensive and aggressive language.
-Use of vulgar or obscene language that contributes to a hostile environment.
-Statements that imply or directly express an intention to inflict harm or violence.

If the tweets exhibits one of the above described charactericts classify it possible_toxic else classify it as non_toxic.


Always Respond only with all the given classnames without missing a single class. Omit justifications

Examples:

Frauen gehören in die Küche, nicht ins Büro. Sie sind einfach nicht dafür gemacht!
neutral_post, possible_hate_speech, non_toxic

Alle Ausländer nehmen uns die Jobs weg und ruinieren unsere Wirtschaft. Bald wird es keine Deutschen mehr geben!
possible_fake_news, possible_hate_speech, possible_toxic

Die Regierung unterstützt illegale Migranten mit kostenlosen Villen, während wir Steuern zahlen. Pure Diskriminierung gegen uns Deutsche!
possible_fake_news, possible_hate_speech, possible_toxic

Die Regierung plant heimlich, alle Bargeldzahlungen abzuschaffen, damit sie uns total kontrollieren können. Wach endlich auf, du Schaf!
possible_fake_news,neutral_comment, possible_toxic
"""

def call(tweet):
    print("Calling the API with the tweet:", tweet)
    messages = [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': tweet}]

    response = requests.post(
        endpoint,
        json={
                "model": MODELS[1],
                "messages": messages,
                "options": options
            },
    ).json()
    return response

while True:
    tweet = input("Enter the tweet:")
    
    if tweet == "exit":
        break
    
    api_response = call(tweet)

    # print(api_response)
    print("############")
    print(api_response['response'])
    print("############")