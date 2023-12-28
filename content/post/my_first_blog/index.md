---
title: How Words Are Represented In NLP Systems
date: '2023-12-28'
---


 NLP systems such as large language models (e.g., GPT-3.5, Claude) can do a wide range of language tasks, from assigning labels to text to writing an essay. In order to do that, the systems must have an efficient way to process language. The computers can only understand numbers. Therefore, it is crucial to convert text into numbers and ensure that those numbers reflect the important properties of the text it represents. The subsections below, I will explain the steps to achieve this. Traditional and mordern techniques are presented. I will also discuss the strength and weakness of each approach.

## Character Encoding

In order for the computer to do the tasks, the data must be in a certain numerical format which readable by the computer. Character encoding in English is straightforward because English words consists of distinct characters and each one can be represented by a unique code.

In NLP, character encoding is often performed according to UTF-8 (Unicode Transformation Format-8), one of the encoding system developed by the Unicode Stanfard. It is a variable-length character encoding system that convert each character into one or more byte, depending on how many features it has. A word, which usually consists of several characters is encoded as a sequence of bytes, called a byte-string. The code cells below illustrate how words in different languages are encoded using UTF-8.

```python
# Cell 1
english_word = 'time'
encoding3 = english_word.encode('utf-8')
print("The byte-string for 'time' is: " + str(encoding3))
utf8_encoding = [byte for byte in encoding3]
print("The utf-8 encoding for 'time' is: "+ str(utf8_encoding))
```

```python
# Output: Cell 1
# The byte-string for 'time' is: b'time'
# The utf-8 encoding for 'time' is: [116, 105, 109, 101]
```

```python
# Cell 2
vietnamese_word = "tuần"
encoding2 = vietnamese_word.encode('utf-8')
print("The byte-string for 'tuần' is: " + str(encoding2))
utf8_encoding2 = [byte for byte in encoding2]
print("The utf-8 encoding for 'time' is: "+ str(utf8_encoding2))
```

```python
# Output: Cell 2
# The byte-string for 'tuần' is: b'tu\xe1\xba\xa7n'
# The utf-8 encoding for 'time' is: [116, 117, 225, 186, 167, 110]
```

Cell 1 illustrate how character encoding is for the English language. b'time' represents the UTF-8 encoding of the word 'time'. The actual encoding of 'time' is the string of numerical value [116, 105, 109, 101]. Every English word, or more generally, word in the Latin alphabet is encoded straightforwardly as one byte. However, other language which use Latin alphabet but with tone marks such as Vietnamese, German, and French cause a challenge to character encoding. In cell 2, the Vietnamese word 'tuần' (week) also has four characters. However, the character 'ầ' has two tone marks. Therefore, two more bytes are needed to represent this character, resulting in a total six of bytes for the word 'tuần'.

```python
# Cell 3
hebrew_word = "אגס"
hebrew_char = 'ס'
encoding4 = hebrew_word.encode('utf-8')
utf8_encoding4 = [byte for byte in encoding4]
encoding5 = hebrew_char.encode('utf-8')
utf8_encoding5 = [byte for byte in encoding5]
print(encoding4)
print(utf8_encoding4)
print(utf8_encoding5)
```

    b'\xd7\x90\xd7\x92\xd7\xa1'
    [215, 144, 215, 146, 215, 161]
    [215, 161]

```python
# Output: Cell 3
# b'\xd7\x90\xd7\x92\xd7\xa1'
# [215, 144, 215, 146, 215, 161]
# [215, 161]
```

```python
# Cell 4
hebrew_char2 = 'בָּיִ'
encoding6 = hebrew_char2.encode('utf-8')
utf8_encoding6 = [byte for byte in encoding6]
print(utf8_encoding6)
```

```python
# Output Cell 4
# [215, 145, 214, 184, 214, 188, 215, 153, 214, 180]
```

Languages with different orthographical system like Hebrew, Thai, Russian, and Arabic also pose a computational problem to character encoding. In Cell 3, the Hebrew word 'אגס', meaning 'pear', require six bytes to represent although it only has three characters. Moreover, some Hebrew words have diacritics and ligatures, which require some more bytes to encode. Using more bytes make the encoding process more computationally expensive and cause problem for other tasks such as slicing, indexing, and tokenizing because the additional bytes may blur the boundaries between the characters. In addition, unlike other languages, Hebrew character, words, and sentences are read from left to right. This characteristic challenge the encoding task. From the code cell, it can be seen that Hebrew characters are also encoded from left to right. However, in some cases, Hebrew words may be accompanied by Roman numerals, then they must be read from left to right. Such inconsistencies may cause confusion to the encoding task.

```python
# Cell 5
chinese_word  = '龍'
encoding7 =  chinese_word.encode('utf-8')
print(encoding7)
utf8_encoding7 = [byte for byte in encoding7]
print(utf8_encoding7)
```

    b'\xe9\xbe\x8d'
    [233, 190, 141]

```python
# Output cell 5
# b'\xe9\xbe\x8d'
# [233, 190, 141]
```

In some logographic languages, such as Chinese, a character can consist of a few letters. Assigning each unique character to a byte-string is computationally expensive. In addition, in the case of Chinese, a character can have two different forms, which are traditional and simplified forms. This also create problems with mappings.

https://arxiv.org/pdf/1805.03330.pdf
(solution for encoding Chinese character)

## Text Normalization

### Tokenization

Tokenization is the process of transforming text into meaningful segments, called tokens, that the computer can distinguish the linguistic units in the data. Tokenization is a part of text normalization process, which aims to transform raw text data into a standardized format that is understandable by the computer.

There are several methods to tokenize text. The most basic approach is to treat one word as a token. In English, tokenization can be done by splitting text by the white space character. The code cell below show the UNIX tool for tokenization works in English.

#### Word Tokenization

In practice, one of the oldest approach to tokenization is the UNIX tools. It works the same as the code in Cell 6 but is implemented by a Unix command 'tr'. It replaces everything that is not Latin letters into newline characters. This results in a list of types separated by lines and the number of tokens of the respective types. However, this method has a lot of problems. It removes important elements of the text, such as number, special characters and punctuations. These elements are nessesary to expresses non-word entities such as dates, websites, and hashtags. In addition, it does not allow for multiword expression such as idioms.

```python
# Cell 6
import re

text = "In in practice, these embedding vectors have 1000 dimensions, and and capture the the semantic similarities and relationships between the words."
word_count = {}
words = re.findall(r'\b\w+\b', text.lower())
for word in words:
    word = re.sub(r'[^a-z]', '', word)
    if word:
        word_count[word] = word_count.get(word, 0) + 1
        print(str(word_count[word]) + "  " + word)
```

```python
# Output cell 6
# 1  in
# 2  in
# 1  practice
# 1  these
# 1  embedding
# 1  vectors
# 1  have
# 1  dimensions
# 1  and
# 2  and
# 1  capture
# 1  the
# 2  the
# 1  semantic
# 1  similarities
# 3  and
# 1  relationships
# 1  between
# 3  the
# 1  words
```

However, in many cases, whitespace is not sufficient to be token boundary and single words may not be meaningful enough to represent tokens. For example, proper names such as Golden Gate Bridge are beter treated as one token rather than three. In these cases, an algorithm called named entity recogition is usually applied to recognize unique multiword entities like proper names, events, and dates.  Another method to tokenize word is to use regular expression, a language to search for text pattern. The Pentreebank tokenizer is a well-known example of this method. Negative marks like 'doesn't' is tokenize into 'does' and 'n't'.   

```python
# Cell 7
import nltk
from nltk.tokenize import TreebankWordTokenizer
sentence = "She doesn't have 1$ after three days in San Francisco."
TreebankWordTokenizer().tokenize(sentence)
```

```python
# Output: Cell 7
#['She','does',"n't",'have','1','$','after','three','days','in','San','Francisco','.']
```

Word tokenization is effective and simple for tasks that require processing at word level such as text classification, sentiment analysis, and named entity recogntion. It is effective for languages which has clear word boundary like English and other Indo-European languages. However, languages rich morphological systems can cause difficulty to the tokenization task. Words can have prefixes and suffixes, which carry meaning. Therefore, they should be encoded so that further NLP models can understand the morphological knowledge of languages.     

#### Subword Tokenization

Recently, it is more common to NLP researchers to do ***sub-word tokenization***, which means that words are segmented into smaller units. This approach has advantages over traditional word tokenization. First, it helps the system understand morphology. Second, it helps dealing with out-of-vocabulary words by segmenting the part of words that is known to the model.  

Several algorithms have been proposed to do sub-word tokenization. The subsections below decribe each althorithm in detail. There are byte-fair encoding, wordpiece encoding, and unigram language modeling (Jurafsky and Martin 2023, p.xx).

##### Byte-Pair Encoding Tokenization (BPE)

BPE is one of the most common tokenization algorithms in morden NLP. It is used by many Tranformer-based models. BPE works by merging the most frequent sequences of characters into a larger sub-word within a word and building a vocabulary consisting of meaningful sub-words. The starting point is the character strings that have been separated by the white space character, so-called words. Each character is considered as a token. The algorithm will first select the most frequent adjacent characters in the training corpus and treat them as a single token. Then this 2-character token is frequntly occur with another character and they are merged into another single token. The same procedure applies to the rest of the characters.

It is important to note that BPE only works within word boundaries. This mean that two adjacent characters separated by white space are not merged. This way, it can learn the prefixes and suffixes of words. It can effectively recognize that -ment and -ous, for example, are common sub-word. This method work well for aggutinative languages such as English and German, which use affixes to express grammatical and semantic relations. BPE does not work exactly the same way for isolated languages such as Chinese and Vietnamese because they do not have subwords. It will likely to treat sequences of characters separated by white space as a single token.

In principle, it does not work with multiword expressions such as idioms. The code cell below demonstrate how BPE works on GPT-3.5 turbo model. The BPE tokenizer is used by GPT-3.5 turbo is called *tiktoken*.

```python
# Cell 8
!pip install --upgrade tiktoken
!pip install --upgrade openai
```

```python
# Cell 9
import tiktoken
encoding = tiktoken.get_encoding('p50k_base')
```

```python
# Cell 10
byte_string_eng = encoding.encode("His disrespectfullness is very annoying!")
byte_string_eng_unknown = encoding.encode("The wug is tassing me.")
byte_string_vie = encoding.encode("Sự phát triển của trí tuệ nhân tạo đang rất mạnh mẽ.")
token_list_eng =  [encoding.decode_single_token_bytes(token) for token in byte_string_eng]
token_list_vie =  [encoding.decode_single_token_bytes(token) for token in byte_string_vie]
token_list_eng_unknown =  [encoding.decode_single_token_bytes(token) for token in byte_string_eng_unknown]
print(token_list_eng)
print(token_list_vie)
print(token_list_eng_unknown)
```

```python
# Output: Cell 10
# [b'His', b' disrespect', b'full', b'ness', b' is', b' very', b' annoying', b'!']
# [b'S', b'\xe1', b'\xbb', b'\xb1', b' ph', b'\xc3\xa1', b't', b' tri', b'\xe1', b'\xbb', b'\x83', b'n', b' c', b'\xe1', b'\xbb', b'\xa7', b'a', b' tr', b'\xc3\xad', b' tu', b'\xe1', b'\xbb', b'\x87', b' n', b'h', b'\xc3\xa2', b'n', b' t', b'\xe1', b'\xba', b'\xa1', b'o', b' \xc4', b'\x91', b'ang', b' r', b'\xe1', b'\xba', b'\xa5', b't', b' m', b'\xe1', b'\xba', b'\xa1', b'n', b'h', b' m', b'\xe1', b'\xba', b'\xbd', b'.']
# [b'The', b' w', b'ug', b' is', b' t', b'assing', b' me', b'.']
```

From the output of the code cell, the BPE algorithm works well for English. It tokenize the morphologically complex word "disrespectfullness" into the morphemes "disrespect", "full", and "ness". The second output show how it deals with the unknown words "wug" and "tassing". "wug" is tokenized into "w" and "ug" because "ug" is a more common combination, as in "ugly" and "bug". For "tassing", it is segmented into "t" and "assing" as in "embarassing" and "harassing".

##### Unigram Tokenization

Another method for tokenization is Unigram model. It works based on the rule of probability. This means that the set of characters which have the highest probability are grouped into one token. This algorithm assumes that characters are independent from each other. According the rule of probability. If the events are independent from each other, then probability of observing the events occuring together is the product of their probabilities. Thus, the probability of the word "time" is equal to the product of the probabilities of "t", "i", "m", and "e", as shown in the following equation.

$$
P('time') = P('t') * P('i') * P('m') * P('e')
$$

When the

##### WordPiece Tokenization

WordPiece is the sub-word tokenization algorithm used for BERT. It words relatively similar to BPE in the way that it first split all characters in a word, then merge certain characters to form a sub-word unit. However, the way it merge characters is different from BPE. It combines the pair of characters that when the probability of them being together is higher than the product of the probabilities of them being alone.

#### Character-Based Tokenization

In some languages, the word boundaries are not clear. In Chinese, some characters such as "天", meaning sky or "好", meaning good, are single words. However, some words consist of two or more characters. Thus, it is more effective to treat each character as a token. The code cell below demonstrates how to do this.

```python
# Cell 11
chinese_text = "今天天气很好，我们去公园。"
chinese_characters = [char for char in chinese_text]
print("Tokenized Chinese characters:")
print(chinese_characters)
```

```python
# Output Cell 11
# Tokenized Chinese characters:
# ['今', '天', '天', '气', '很', '好', '，', '我', '们', '去', '公', '园', '。']
```

### Lemmatization and Stemming

 Agglutinative language are morphologically complex. Their words usuallyconsists of multiple morphemes, including a base and one or more affixes, which can be prefixes or suffixes. In some NLP tasks, it is common to make the system recognize that some words have the same base so their meanings are closely related. Lemmatization is the task of stemming affixes from the words, only keeping the base. The cell below shows how it is implemented using the Porter Stemmer.

```python
# Cell 12
from nltk.stem import *
stemmer = PorterStemmer()
word_list = ['bags','students','studying','moved','developing','estimation']
word_stemed = [stemmer.stem(word) for word in word_list]
print(word_stemed)
```

```python
# Cell 12: Output
# ['bag', 'student', 'studi', 'move', 'develop', 'estim']
```

The output shows that the stemmer successfully remove the affixes for most words. It removed the morpheme -s, -ed, and -ing from the words. In some cases it incorrectly stemmed the words, as with 'studi' and 'estim'.  

### Parsing

Apart from words, there are larger units of text. The most basic one is sentences. Many sentences are separated by a period "." but some types of sentences may end in other special characters, such as "!" and "?". In NLP, it is important to help the computer understand that each sentence carry a certain meaning. The task of segmenting text into sentences is called parsing. Intuitively, it is possible to split text by ".", "!", and "?". However, sometimes this basic approach is problematic because these characters do not always mark the end of a sentence. For examole, decimal numbers contain periods (e.g., 0.5). A more linguistically-driven approach is to segment sentences based on the relationship between words, called dependency parsing or constituency parsing. The algorithm first assigns a part-of-speech to each token in the sentence. This task is called part-of-speech tagging. It aims to identify the syntactic relationship between words. The code below shows how this can be done using spacy.

```python
# Cell 13
import spacy
nlp = spacy.load("en_core_web_sm")
text = "I want to take over the housing contract. The room is beautiful and bright"
doc = nlp(text)
for token in doc:
    print(token.text, token.dep_,
          [child for child in token.children])
```

```python
# Cell 13: Output
# I nsubj []
# want ROOT [I, take, .]
# to aux []
# take xcomp [to, over, contract]
# over prt []
# the det []
# housing compound []
# contract dobj [the, housing]
# . punct []
# The det []
# room nsubj [The]
# is ROOT [room, beautiful]
# beautiful acomp [and, bright]
# and cc []
# bright conj []
```

### Case Folding

The computer are not sensitive to case. It treats Los Angeles and los angeles as two different types. However, in certain tasks such as named entity recognition and information retrieval, it is important to help the computer recognizes that they refer to the same entity. Therefore, it is helpful to convert all text into either uppercase or lowercase. Lowercase is most often chosen. Moreover, convert the text into a standard case helps save computational resources. The code cell below demonstrate how case folding can be conducted.

```python
# Cell 14
text_case = "ThE QuicK bRoWn Fox JumPs OVeR The LazY Dog."
folded_text = text_case.lower()
print("Case Folded Text:", folded_text)
```

```python
# Cell 14: Output
# Case Folded Text: the quick brown fox jumps over the lazy dog.
```

However, in some cases, it is not needed to do case folding. In tasks like sentiment analysis, cases may carry emotions. Words with uppercase may indicate strong emotions or emphasis. Therefore, cases should be remained in such tasks. Uppercase words may be the abbreviation of longer phrases so doing case folding may alter the meaning of such words. For example, TELL is the short form of Technology-Enhanced Language Learning. If it is reduced to lowercase, it becomes a totally different word. Another situation that case folding should not be done is when the language itself is case-sensitive. For instance, German nouns are always captitalized, so cases signal the part of speech of the word. Avoiding standardizing cases would help the NLP system learn the language and produce it with the correct capitalization norm.

### Stopword Removal

Common function words like 'the', 'a/an', and 'and' are often removed because they do not carry much meaning and thus are not useful for NLP tasks. The code cell below demonstrates how to remove these words using NLTK.

```python
# Cell 15
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
nltk.download('stopwords')
nltk.download('punkt')
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!

    True

```python
# Cell 16
stop_words = set(stopwords.words('english'))
word_tokens = TreebankWordTokenizer().tokenize('I am an English teacher in Vietnam.')
filtered_sentence = [w for w in word_tokens if not w in stop_words]
print("Token list:", filtered_sentence)
```

    Token list: ['I', 'English', 'teacher', 'Vietnam', '.']

```python
# Output: Cell 17
# Token list: ['I', 'English', 'teacher', 'Vietnam', '.']
```

```python
# Cell 17
print(stopwords.words('english'))
```

```python
# Output: Cell 17
# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
```

Stopwords removal is useful in tasks that require finding out the overall meaning of the text such as topic modeling and information retrieval. However, it is not always necessary to remove stopwords because they may be useful for certain tasks, especially text generation tasks such as machine translation. They help the NLP system learn the language and ensure that the generated text is grammatically correct. In addition, in the list of stopwords, there are words that do have meaning, such as 'doesn't' and 'couldn't'. Thus, removing them may alter the meaning of the text.

# How NLP Tasks Are Performed?

After being transformed into numbers, the NLP system takes a large amount of text data and train it to perform certain tasks. The following subsections will present the tasks that can be done by NLP systems and the algorithm that is used for training.

## Which kind of tasks can be done by NLP systems?

NLP systems can do a wide range of language tasks after being trained on a sufficient amount of data. They can classify text into different categories, translate text, summarize text, and generate various kinds of text that is specific to one's need.

***Text classification*** is the task of giving a predetermined label to an input. The NLP is trained on a set of labeled data using an algorithm. It can then predict the label of new text by assigning the probabilities to all possible labels and pick the one with the highest probability.

*Sentiment analysis:* Text often carry emotions. Some words express positivity while other can give a negative feeling. The task of deciding whether a text input is positve or negative is called sentiment analysis.

*Topic modeling:* The task of finding the main topic of a text is called topic modeling.

***Text generation:*** The NLP systems can also genenerate new text, depending on the user's input and the training algorithm. State-of-the-art NLP technology such as ChatGPT is able to answer questions, write essays and code, and produce other kinds of text. Below are some of the most important tasks.

*Machine translation:* The NLP system can translate text from one language to another.

*Text summarization:* The system takes input as a long text and shorten it.

In the subsection, I will present the traditional and also the state-of-the-art algorithms that are used to train those systems. I will also demonstrate how text classification and generation can be done by NLP systems.

### Simple Text Classification with Navie Bayes

One of the traditional algorithm used for text classifiction is the Navie Bayes Classifiers, which is based on Bayes's rule of probability. In the context of text classification, the Bayes's rule aim to find the probability of the posterier probability of the predicted label given the input to the system. Suppose that there is a set of label $ x
 (x_1, x_2, x_3)$ and a text input y. The probability of the input y has the label $x_2$ is as follows.

$$
P(x_2|y) = \frac{P(y|x_2)P(x_2)}{P(y)}
$$

$P(y|x_2)$ by the likelihood of the text given the label. For example, the likelihood of the word "policians" to appear in a food review is lower than the likelihood of the word "mouth-watering". $P(x_2)$ is the probability of observing the label $x_2$. This is called the pior probability. This probability can be the ammount of text with the label $x_2$ in the training dataset. Finally, $P(y)$ is the probability of the text $y$ in general. It can be the frequency of the text $y$ in the training corpus.

The classification task aims to find the predicted label $x_p$ of the data $y$. Therefore, the system must find the label that has the highest posterior probability. Since $P_y$ is a constant, $x_p$ must be the label that maximize $P(y|x_2)P(x_2)$. Since the text input $y$ denote a set of words, the likelyhood $P(y|x_2)$ becomes the probability of observing all of the individual words $(y_1, y_2, y_3, y_4, y_5,...)$ in the text.  

$$
x_p = argmax P(y_1, y_2, y_3, y_4, y_5,...|x_2)P(x_2)
$$

The Navie Bayes algorithm has an assumption, which is the ***bag-of-word*** assumption. It treats text as a collection of unordered words. Only the words and their counts are considered important. Therefore, the Navie Bayes algorithm is unable to capture the order in which words appear. This assumption results in the independent occurence of words. According to the rule of probability, the probability of a mutiple independent event is the product of the probability of each event. The equation below thus become:

$$
x_p = argmax P(y_1|x_2)P(y_2|x_2)P(y_3|x_2)P(y_4,x_2) P(y_5|x_2)
$$

To do text classification using Navie Bayes algorithm, the text first has to be converted into numerical values. Each text input has a unique vector consiting of the words in the text and their counts. The code below shows how to convert text into numerical features for Navie Bayes text classification using the CountVectorizer function in scikit-learn.

```python
# Cell 18
from sklearn.feature_extraction.text import CountVectorizer
sample_reviews= [
    "The movie was great.",
    "I didn't like the acting.",
    "Amazing film, fantastic cast!"
]
feature_vector = CountVectorizer()
X = feature_vector.fit_transform(sample_reviews)
feature_names = feature_vector.get_feature_names_out()
import pandas as pd
pd.DataFrame(X.toarray(), columns=feature_names)
```

![Screenshot (856).png](data:image/png;base64,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)

The output shows a matrix of counts of the words in the entiretext. The first row is the words, and the next three rows, corresponding to the feature vector of the three reviews, is the token counts of the word in each review. However, in practice, stopwords such as 'the' and 'was' are removed before the vectorization. The code block below show how to train a text classification using Navie Bayes algorithm. The training set include 4 movie reviews. They are labeled as either positive or negative. The classifier is then used to predict the sentiment of a movie review. It give 0.9 probability to the negative label and 0.09 for the positive label. Thus, the label with the higher probability is chosen.

```python
# Cell 19
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
training_texts = ["This is a good movie",
         "That was a terrible movie",
         "I like the plot of this book",
         "I didn't like the characters in the novel"]

labels = ['positive', 'negative', 'positive', 'negative']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_texts)
naive_bayes = MultinomialNB()
naive_bayes.fit(X, labels)
sentence = "It was a terrible movie yet the characters are well-developed"
sentence_vector = vectorizer.transform([sentence])
predicted_label = naive_bayes.predict(sentence_vector)
label_probabilities = naive_bayes.predict_proba(sentence_vector)
print("Predicted Label:", predicted_label[0])
print("Label Probabilities:", label_probabilities)
```

    Predicted Label: negative
    Label Probabilities: [[0.90856225 0.09143775]]

```python
# Cell 19: Output
# Predicted Label: negative
# Label Probabilities: [[0.90856225 0.09143775]]
```

The Navies Bayes algorithm has several advantages for text classification tasks. Firstly, it is fast and does not require a large amount of training data. Secondly, it can help with both binary and multi-class classification. More importantly, if the features of the input are independent, it can be very sucesssful in classifying data. However, the beg-of-word assumption is not always true. In fact, in the context of text, this assumption is often violated because words are dependent on each other to convey meaning. In addition, this algorithm is unable to handle unknown words. It will give zero probability to any unknown words that the system encounters.

Overall, Navie Bayes is good at capturing the gist of the text without concerning the relationship between words, such as sentiment analysis, topic modeling, spam detaction, and authorship attribution.

### Transformers

Besides the classic Navie Bayes algorithm, there is now a more powerful approach, called Transformer. It can do both text classification and generation. It can capture word orders as well as the relationship between words, even long-range dependencies. Modern language models, such as BERT, GPT-3, and GPT-3.5 are built on top of Transformer.

Transformer is a neural network architecture that uses a multihead self-attention machenism. Self-attention means that the model is able to process multiple elements of an input simutaneously and compare one element with all of the others to infer the relationship between the current element with the others. In addition, with the multihead self-attention layers, the model can learn different types of relatioships and importances between one element and the others.

Another strength of Transformer architechture compared to other algorithm is that it processes words in the order that their appear in the text. Unfortunately, the self-attention mechanism does not consider the order of words, neither neural network architecture. This is done by another task called positional embedding. It gets the position of each word in the text and embeds it into a vector then add it to the embedding of the word. This results in a single embedding that capture both the semantic content of the word and its position.

The position of each token in the input is obtained by a position encoding function. The input is the position of the word in the text and dimension of the embedding. The output is a positional embedding with the same dimension. For example, given the sentence "Time flies like an arrow.". The word embedding of the word "time" is $[0.03, 0.006, 0.275]$ and the positional embedding of the word "time" is $[0.1, 0, 0.9]$. These two vectors are added to form a complete embedding of the word "time".

```python
# Cell 20
from transformers import pipeline
text_generator = pipeline("text-generation", model="gpt2")
generated_text = text_generator("I will tell you a story. Yesterday, when I was ")
print(generated_text[0]['generated_text'])
```

    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

    I will tell you a story. Yesterday, when I was  getting to high school in the Bay Area and I was in class with a group of people, I realized my body had begun to shake, and I lost the ability to concentrate.

```python
# Output: Cell 20
# I will tell you a story. Yesterday, when I was  getting to high school in the Bay Area and I was in class with a group of people, I realized my body had begun to shake, and I lost the ability to concentrate.
```

The below code cell is an example about how to do text generation with transformer-based model. It did a simple text generate task which takes input as an incomplete paragraph and generate some text to complete it.

### Probabilistic (N-gram) Language Modeling

Another algorithm that works based on the rule of probability is called probabilistic or n-gram language modeling. At the core of this algorithm is the assumption that the probability of a word given the previous n words is independent of the probability of the word given the previous n-1 words. In other words, this follows the chain rule of probability.

$$
P(w_n|w_1,...,w_{n-1}) = P(w_1)P(w_2|w_1)P(w_3|w_1,w_2)P(w_n|w_1,...,w_n)
$$

However, computing the probability of a word given all of the previous words is not very efficient given the large number possible combinations. To solve this problem, the Markov assumption is used. It states that the future event is only dependent on the present. In the context of text generation, this means that the probability of the next word given the previous word is only dependent on the most previous words. Thus, the probability or a word given all of the previous context is roughly equal to the its probability given the previous $i$ words. $i$ is a very small number, normally ranging from 1 to 5. When $i=1$, the model is called unigram language modeling. It is thus called bigram when $i=2$ and trigram when $i=3$. Trigram has been the most common model in NLP because the window size is long enough to capture previous context but short enough to avoid sparsing. Below is the formula of a trigram model.

$$
P(w_n|w_1,...,w_{n-1}) = P(w_n|w_{n-1},w_{n-2},w_{n-3})
$$

To calculate $P(w_n|w_{n-1},w_{n-2},w_{n-3})$, the model has to divide the count of the sequence ($w_n, w_{n-1},w_{n-2},w_{n-3}$) by the count of the previous sequence ($w_{n-1},w_{n-2},w_{n-3}$). For example, to know the probability of the word "by" given the context "He is bitten". The model search in the training corpus for every instance "He is bitten by" and divide its count to the count of "He is bitten". This approach is called ***Maximum Likelihood Estimation.*** The exact formula is showb below. It is taken from Jurafsky and Martin (2023).

$$
P('by'|'he', 'is','bitten') = \frac{Count('he','is','bitten','by')}{Count('he','is','bitten')}
$$

```python
# Cell 21
from nltk import trigrams
from nltk.tokenize import word_tokenize
training_data = [
    "I love machine learning a lot",
    "Machine learning is a lot more fascinating",
    "Learning new things is exciting",
    "I enjoy learning new concepts",
    "The learning process is continuous"
]
tokenized_data = [word_tokenize(sentence) for sentence in training_data]
trigram_model = {}
for sentence in tokenized_data:
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        key = (w1, w2)
        if key not in trigram_model:
            trigram_model[key] = []
        trigram_model[key].append(w3)
def predict_next_word(word1, word2):
    next_words = trigram_model.get((word1, word2))
    if next_words:
        return max(set(next_words), key=next_words.count)
    return None
# Test the trigram model by predicting the next word
word1 = "machine"
word2 = "learning"
predicted_word = predict_next_word(word1, word2)

print(f"The next word after '{word1} {word2}' is '{predicted_word}'")
```

    The next word after 'machine learning' is 'a'

```python
# Cell 21: Output
# The next word after 'machine learning' is 'a'
```

While n-gram model is computationally efficient and has been widely used, it has a few limitations. Firstly, dues to the fixed context size, the model is unable to process long-range dependencies. In some certain cases, a larger context is needed to predict the next word. Secondly, it cannot deal with unseen words. If an n-gram model encounters a word that is not in the training data, it is usually replaced with an unknown token. The model will then ignore this token and extend the context to the next word. However    

# References

Jurafsky, D., & Martin, J. H. (2023). *Speech and language processing: An introduction to natural language processing, computational linguistics, and speech recognition*. Retrieved from https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf
