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
