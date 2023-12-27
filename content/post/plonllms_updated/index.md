---
title: Installing and authenticating OpenAI API
date: '2023-12-27'
---


```python
pip install --upgrade openai
```

```python
from openai import OpenAI
import pandas as pd
client = OpenAI(api_key='api_key')
```

# Translation of Wug's test

```python
wug_test_original = [
    "One bird-like animal, then two. 'This is a wug. Now there is another one. There are two of them. There are two __.'",
    "One bird, then two. 'This is a gutch. Now there is another one. There are two of them. There are two __.'",
    "Man with a steaming pitcher on his head. 'This is a man who knows how to spow. He is spowing. He did the same thing yesterday. What did he do yesterday? Yesterday he __ '",
    "One animal, then two. 'This is a kazh. 'Now there is another one. There are two of them. There are two __.'",
    "Man swinging an object. 'This is a man who knows how to rick. He is ricking. He did the same thing yesterday. What did he do yesterday? Yesterday he __.'" ,
    "One animal, then a miniscule animal. 'This is a wug. This is a very tiny wug. What would you call a very tiny wug? __. This wug lives in a house. What would you call a house that a wug lives in? __'",
    "One animal, then two. 'This is a tor. Now there is another one. There are two of them. There are two __.'",
    "Dog covered with irregular green spots. 'This is a dog with quirks on him. He is all covered with quirks. What kind of dog is he? He is a __ dog.'",
    "One flower, then two. 'This is a lun. Now there is another one. There are two of them. There are two __·'",
    "One animal, then two. 'This is a niz. Now there is another one. There are two of them. There are two __·'",
    "Man doing calisthenics. 'This is a man who knows how to mot. He is motting. He did the same thing yesterday. What did he do yesterday? Yesterday he __·'",
    "One bird, then two. 'This is a kra. Now there is another one. There are two of them. There are two __·'",
    "One animal, then two. 'This is a tass. Now there is another one. There are two of them. There are two __·'",
    "Man dangling an object on a string. 'This is a man who knows how to bod. He is bodding. He did the same thing yesterday. What did he do yesterday? Yesterday he __·'",
    "Man shaking an object. 'This is a man who knows how to naz. He is nazzing. He does it every day. Every day he __·'",
    "One insect, then two. 'This is a heaf. Now there is another one. There are two of them. There are two __·'",
    "Man exercising. 'This is a man who knows how to gling. He is glinging. He did the same thing yesterday. What did he do yesterday? Yesterday he __·'",
    "Man holding an object. 'This is a man who knows how to loodge. He is loodging. He does it every day. Every day he __·'",
    "Man standing on the ceiling. 'This is a man who knows how to bing. He is binging. He did the same thing yesterday. What did he do yesterday? Yesterday he __·'",
    "One animal wearing a hat, then two wearing hats. 'This is a niz who owns a hat. Whose hat is it? It is the __ hat. Now there are two nizzes. They both own hats. Whose hats are they? They are the __ hats.'",
    "Man balancing a ball on his nose. 'This is a man who knows how to zib. What is he doing? He is __· What would you call a man whose job is to zib?'",
]
```

```python
output = open('translation-french-gpt-4.txt','w')

for question_number, question in enumerate(wug_test_original):
  response = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {"role": "user",
     "content": "Translate the following paragraph into Spanish except for the unknown word: " + question}
  ]
  )
  print(response.choices[0].message.content)
  output.write(f"## Wug Q{question_number + 1}:\n")
  output.write(question)
  output.write('\n')
  output.write(response.choices[0].message.content)
  output.write('\n')
  output.write('\n---\n')

output.close()
```

# Run the Wug's test on LLMs

```python
wug_test = []
with open("translation_portuguese_final.txt", "r") as german_wug_test:
  for line in german_wug_test:
    wug_test.append(line.replace("\n", ""))
print(wug_test)
print(len(wug_test))
fantasy_word_portuguese = ['wug','gutch','spow','kazh','rick','wug','tor','lun','niz','mot','kra','tass','bod','naz','heaf','gling','loodge', 'bingu','niz','zib']
fantasy_word_spanish = []
text_word  = pd.DataFrame(list(zip(wug_test, fantasy_word_portuguese)),columns =['text', 'word'])
text_word
```

```python
gpt_35 = "gpt-3.5-turbo-0613"
gpt_4 = "gpt-4-0613"
output = open('output_portuguese_gpt-3.5-turbo-0613.txt','w') # change output file name

for question_number, question in enumerate(wug_test):
  word = text_word['word'][question_number] # change language
  prompt = f"Assuming that {word} is a Portuguese word, read the following paragraph and replace the underscores with a suitable word form of {word}"
  response = client.chat.completions.create(
    model=gpt_35, # changle model
    messages=[
      {"role": "user",
      "content": prompt + question}
  ]
)
  print(response.choices[0].message.content)
  output.write(f"## Wug Q{question_number}:\n")
  output.write(question)
  output.write('\n')
  output.write(response.choices[0].message.content)
  output.write('\n')
  output.write('\n---\n')

output.close()
```
