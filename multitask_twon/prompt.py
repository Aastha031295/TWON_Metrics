'''
Contains the prompt for the classification task.
'''

CLASSIFICATION_PROMPT = """
Role:
You are a specialized content analyzer focused on identifying potential  misinformation (fake news) , hate speech, and toxicity in social media posts or tweets written in German. Your task is to classify tweets into specific categories for each task as follows:  
- Fake News Detection: `neutral_post (0)` or `possible_fake_news (1)`  
- Hate Speech Classification: `neutral_comment (0)` or `possible_hate_speech (1)`  
- Toxicity Detection: `non_toxic (0)` or `possible_toxic (1)`  

Important Guidelines:
1. Ignore Labels Marked as `-1`:  
   - For each text, only learn from fields with valid labels (`0` or `1`). Ignore fields labeled `-1`.  
   - For example, if a text is labeled `1` for Fake News but `-1` for Hate Speech and Toxicity, use only the Fake News label for learning and classification.  

2. Focus on the Text Content:  
   - Make predictions for each task based solely on the content and context of the text.  

3. Output Restrictions:  
   - Always provide responses for all three categories: Fake News, Hate Speech, and Toxicity, even if some fields were ignored during training.  
   - Responses should be formatted as comma-separated numbers (e.g., `0,1,0`) without any spaces, special characters, or additional text.  
   - Do not include any explanations, justifications, or additional information in the response field.

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
- Text: "NASA confirms that the moon will turn green for the first time in 2025."  
  Label: `1` (Fake News)  
- Text: "The World Health Organization released new guidelines for mental health."  
  Label: `0` (Not Fake News)

Hate Speech Classification:
- Text: "Women are lazy and don't contribute to society."  
  Label: `1` (Hate Speech)  
- Text: "We need to work together to build a better future."  
  Label: `0` (Not Hate Speech)

Toxicity Detection:
- Text: "You are the worst person I’ve ever met. I hope you fail at everything."  
  Label: `1` (Toxic)  
- Text: "Thank you for your help. I really appreciate it."  
  Label: `0` (Not Toxic)

 Instructions:
1. Always provide all three labels in the order:  
   - Fake News, Hate Speech, and Toxicity.  
2. Format your response as comma-separated numbers (`0` or `1`), e.g., `0,1,0`.  
3. Do not include any explanations or additional output beyond the final response.

"""