from g4f.client import Client

client = Client()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": " what is todays date and share me todays current affair news"}],
    web_search=False
)
print(response.choices[0].message.content)