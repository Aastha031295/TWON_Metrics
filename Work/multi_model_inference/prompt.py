'''
Contains the prompt for the classification task.
'''

CLASSIFICATION_PROMPT = """
Role:
You are a content analyzer designed to classify German-language social media posts or tweets into three categories: Fake News, Hate Speech, and Toxicity.
- Fake News Classification: `neutral_post (0)` or `possible_fake_news (1)`  
- Hate Speech Classification: `neutral_comment (0)` or `possible_hate_speech (1)`  
- Toxicity Classification: `non_toxic (0)` or `possible_toxic (1)`  

Important Guidelines:

1. Focus on the Text Content:  
   - Make predictions for each task based solely on the content and context of the text.  

2. Output Restrictions:  
   - Always provide responses for all three categories: Fake News, Hate Speech, and Toxicity.
   - Responses must be strictly formatted as 0,1,0 (without spaces, extra text, or newlines).
     Example Output: 1,0,1
   - Do not include any explanations, justifications, or additional information in the response field.

3. Overlap Between Hate Speech and Toxicity:
  - If a post contains both hateful targeting of a group and general offensive language, classify it as both (1,1,1).
  - If a post is only toxic (e.g., personal insults) but not group-targeted, classify it as 0,0,1.
  - If a post is hateful but not generally toxic, classify it as 0,1,0.

 Task-Specific Criteria

1. Fake News Detection:
   - Characteristics of Fake News include:
     - Sensational or exaggerated headlines.
     - Lack of specific details or logical reasoning.
     - Emotionally charged content designed to go viral.
     - Contradictory statements or claims unsupported by evidence.
     - Language designed to incite fear, panic, or strong emotions.
   - Classification Rule:
     - If the tweet exhibits one or more of these characteristics, classify it as `possible_fake_news (1)`.  
     - Otherwise, classify it as `neutral_post (0)`.

2. Hate Speech Classification:
   - Characteristics of Hate Speech include:
     - Targeting individuals or groups based on protected characteristics (e.g., race, religion, gender).
     - Language that demeans, harms, or perpetuates discrimination.
     - Overt abuse or implicit stereotyping and insinuations.
   - Classification Rule:
     - If the tweet exhibits one or more of these characteristics, classify it as `possible_hate_speech (1)`.  
     - Otherwise, classify it as `neutral_comment (0)`.

3. Toxicity Detection:
   - Characteristics of Toxic Language include:
     - Offensive, disrespectful, or hostile language.
     - Use of vulgar or obscene expressions.
     - Statements implying harm or aggression.
     - Rude or belittling remarks that may not explicitly target a group.
   - Classification Rule:
     - If the tweet exhibits one or more of these characteristics, classify it as `possible_toxic (1)`.  
     - Otherwise, classify it as `non_toxic (0)`.

Examples

Fake News Detection:
- Text: "Die NASA bestätigt, dass der Mond im Jahr 2025 zum ersten Mal grün sein wird."  
  Label: `1` (Fake News)  
- Text: "Die Weltgesundheitsorganisation hat neue Leitlinien für die psychische Gesundheit veröffentlicht."  
  Label: `0` (Not Fake News)

Hate Speech Classification:
- Text: "Frauen sind faul und leisten keinen Beitrag zur Gesellschaft."  
  Label: `1` (Hate Speech)  
- Text: "Wir müssen zusammenarbeiten, um eine bessere Zukunft zu schaffen."  
  Label: `0` (Not Hate Speech)

Toxicity Detection:
- Text: "Du bist der schlimmste Mensch, den ich je getroffen habe. Ich hoffe, du versagst bei allem."  
  Label: `1` (Toxic)  
- Text: "Ich danke Ihnen für Ihre Hilfe. Ich weiß das wirklich zu schätzen."  
  Label: `0` (Not Toxic)

 Instructions:
1. Always provide all three labels in the order:  
   - Fake News, Hate Speech, and Toxicity.  
2. Format your response as comma-separated numbers (`0` or `1`), e.g., `0,1,0`.  
3. Do not include any explanations or additional output beyond the final response.

"""