# Task 3 - Sentiment NLP Task

**Liav Ariel (212830871)**

**Nehorai Josef (322774852)**

In this exercise, we return to work on the original files of 15000 documents (A, B and C).

We also get data of names, we will use it next for cleaning the documents from names.

We start the exercise with this data:

1. source data of 3 groups of 15000 documents split into 3 groups A, B and C:
   1. group A: './data/15000_docs/original_A.xlsx'
   2. group B: './data/15000_docs/original_B.xlsx'
   3. group C: './data/15000_docs/original_C.xlsx'

2. data of names:
   1. first names collection: './data/names/first-names.xlsx'
   2. last names collection: './data/names/last-names.xlsx'

We need first to preprocess the source data. 

There is two things we should do. 

First, translated the data to english for use 'vaderSentiment' model to tag every document as Positive / Negative / Neutral. 

Second, we want to search if the names entity effect on the sentiment tagging, so we want to create a copy of the source data, but after we clean the names.

After prepare all the data we need. 

We execute 'Sentiment' task on the translated data using 'vaderSentiment' model. 

Then, we execute 'Sentiment' task using AlephBertGimmel on the source files and on the files without real names. 

Finally, we execute 'Sentiment' task using heBert on the source files and on the files without real names.

Now, you can read about how we have done every thing step by step:

> **Translating Data**

**Nehorai Josef was Assigned to this task.**

We worked on Google Colab in this task.

**Link for the notebook:**

1. Todo: insert here

**Output path:**

1. './data/translated_15000_docs/translated_A.xlsx'
2. './data/translated_15000_docs/translated_B.xlsx'
3. './data/translated_15000_docs/translated_C.xlsx'

The vaderSentiment library does not work in Hebrew, so there is no choice but to translate it into English. 

The translation will be done for us by the Google Translate Python API.

**example of use:**

```python
from googletrans import Translator
translator = Translator()

he_sentence = translator.translate('איזה יום שמח לי היום')
print(he_sentence.text)
```

We have a problem with some documents, because their length was too big, the Google Translate Python API crashed.

For solving that, we saved all the problematics documents in different Excel file. 

We, splitting the documents into small chunks, translating them, and finally we concat the part of documents to one document.

> **Clean Data From Names** 

**Liav Ariel was Assigned to this task.**

**Script path:** 

1. './replace_names_using_ner.py'

**Output path:** 

1. '/data/cleaned_from_names_15000_docs/without_names_A.xlsx'
2. '/data/cleaned_from_names_15000_docs/without_names_B.xlsx'
3. '/data/cleaned_from_names_15000_docs/without_names_C.xlsx'

First step was to find out which strings on the document represent names of persons.

I used dictaBert model that was pre-trained on NER (Name Entity Recognition) Task.

Here is a link for the model: https://huggingface.co/dicta-il/dictabert-ner

**Sample usage:**

```python
from transformers import pipeline

oracle = pipeline('ner', model='dicta-il/dictabert-ner', aggregation_strategy='simple')

# if we set aggregation_strategy to simple, we need to define a decoder for the tokenizer. Note that the last wordpiece of a group will still be emitted
from tokenizers.decoders import WordPiece
oracle.tokenizer.backend_tokenizer.decoder = WordPiece()

sentence = '''דוד בן-גוריון (16 באוקטובר 1886 - ו' בכסלו תשל"ד) היה מדינאי ישראלי וראש הממשלה הראשון של מדינת ישראל.'''
oracle(sentence)
```

**Output:**

```python
[
  {
    "entity_group": "PER",
    "score": 0.9999443,
    "word": "דוד בן - גוריון",
    "start": 0,
    "end": 13
  },
  {
    "entity_group": "TIMEX",
    "score": 0.99987966,
    "word": "16 באוקטובר 1886",
    "start": 15,
    "end": 31
  },
  {
    "entity_group": "TIMEX",
    "score": 0.9998579,
    "word": "ו' בכסלו תשל\"ד",
    "start": 34,
    "end": 48
  },
  {
    "entity_group": "TTL",
    "score": 0.99963045,
    "word": "וראש הממשלה",
    "start": 68,
    "end": 79
  },
  {
    "entity_group": "GPE",
    "score": 0.9997943,
    "word": "ישראל",
    "start": 96,
    "end": 101
  }
]
```

I write a script that scan all the source files, and find out all the strings that belongs to 'entity_group': 'PER'.

For each string from this group, I get random names from 'first-name' and 'last-name' data.

I finally, replace the real names with random name.

> **Sentiment On Translated Data Using 'vaderSentiment':**

**Liav Ariel was Assigned to this task.**

**Script path:** 

1. './tag_translated_data.py'

**Output path:** 

1. './output/tagged_translated_15000_docs/tagged_translated_A.xlsx'
2. './output/tagged_translated_15000_docs/tagged_translated_B.xlsx'
3. './output/tagged_translated_15000_docs/tagged_translated_C.xlsx'
4. './output/tagged_translated_15000_docs/result_translated_A.json'
5. './output/tagged_translated_15000_docs/result_translated_B.json'
6. './output/tagged_translated_15000_docs/result_translated_C.json'

I write script that scan all the translated data.

For every document I do Sentiment Tagging using 'vaderSentiment'.

**Link I used for learning this model:**

1. https://reshetech.co.il/machine-learning-tutorials/sentiment-analysis-in-hebrew-kind-of

I create new Excel file that contains the same information as the translated data but with some new columns:

1. Sentiment: column of the decided label of tagging - 'Positive' / 'Negative' / 'Neutral'.
2. Compound: column of the total sentiment score of 'vaderSentiment'.
3. Neg: column of the score for Negative result by 'vaderSentiment'.
4. Neu: column of the score for Neutral result by 'vaderSentiment'.
5. Pos: column of the score for Positive result by 'vaderSentiment'.

And I also creat json file that summarize how much from each label I have in each group.

Json file structure: {'Positive': <amount_of_positives>, 'Negative': <amount_of_negatives>, 'Neutral': <amount_of_neutrals>}

**How to import 'vaderSentiment':**

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
```

**How to use 'vaderSentiment':**

**Code:**

```python
sentence = 'I happily got out of bed at half past ten'
sid.polarity_scores(sentence)
```

**Output:**

```python
{'compound': 0.5574, 'neg': 0.0, 'neu': 0.69, 'pos': 0.31}
```

**How I decided to tag each document:**

I decided according to 'compound_score':

1. 'compound_score' > 0.5 ==> 'Positive'.
2. -0.5 < 'compound_score' < 0.5 ==> 'Neutral'.
3. 'compound_score' < -0.5 ==> 'Negative'.

**Let's discuss the results:**

| Group | Positive | Negative | Neutral |
|-------|----------|----------|---------|
| A     | 60.7 %   | 37.02 %  | 2.28 %  |
| B     | 56.7 %   | 40.58 %  | 2.72 %  |
| C     | 53.2 %   | 42.24 %  | 4.56 %  | 

**Conclusion:**

The results look good, but I would expect different results because I have prior knowledge of the data that is supposed to be more negative.

I think, my classification decision according to the 'compound_score' is incorrect.

Therefore, I decide to run another experience, but now I decide to tag every document according to this rules:

1. 'compound_score' > 1/3 ==> 'Positive'.
2. -1/3 < 'compound_score' < 1/3 ==> 'Neutral'.
3. 'compound_score' < -1/3 ==> 'Negative'.

I save the result output here:

1. './output/tagged_translated_15000_docs/tagged_translated_A_NEW.xlsx'
2. './output/tagged_translated_15000_docs/tagged_translated_B_NEW.xlsx'
3. './output/tagged_translated_15000_docs/tagged_translated_C_NEW.xlsx'
4. './output/tagged_translated_15000_docs/result_translated_A_NEW.json'
5. './output/tagged_translated_15000_docs/result_translated_B_NEW.json'
6. './output/tagged_translated_15000_docs/result_translated_C_NEW.json'

**Let's discuss the results:**

| Group | Positive | Negative | Neutral |
|-------|----------|----------|---------|
| A     |          |          |         |
| B     |          |          |         |
| C     |          |          |         | 

**Conclusion:**



> **Sentiment Task Using 'heBert-Sentiment':**

**Nehorai Josef was Assigned to this task.**

**Script path:** 

1. './tag_with_heBert.py'

**Output path:** 

1. './output/tagged_heBert_15000_docs/tagged_with_heBert_A.xlsx'
2. './output/tagged_heBert_15000_docs/tagged_with_heBert_B.xlsx'
3. './output/tagged_heBert_15000_docs/tagged_with_heBert_C.xlsx'
4. './output/tagged_heBert_15000_docs/result_heBert_A.json'
5. './output/tagged_heBert_15000_docs/result_heBert_B.json'
6. './output/tagged_heBert_15000_docs/result_heBert_C.json'
7. './output/tagged_heBert_15000_without_names_docs/tagged_heBert_without_names_A.xlsx'
8. './output/tagged_heBert_15000_without_names_docs/tagged_heBert_without_names_B.xlsx'
9. './output/tagged_heBert_15000_without_names_docs/tagged_heBert_without_names_C.xlsx'
10. './output/tagged_heBert_15000_without_names_docs/result_heBert_without_names_A.json'
11. './output/tagged_heBert_15000_without_names_docs/result_heBert_without_names_B.json'
12. './output/tagged_heBert_15000_without_names_docs/result_heBert_without_names_C.json'

I do the same as before, just now I work on hebrew data with heBert-Sentiment pre-trained model.
First, on the source data.
Second, on the data without names.

**Link for the model:**

1. https://huggingface.co/avichr/heBERT_sentiment_analysis

**How to Use:**

**For sentiment classification model (polarity ONLY):**

```python
from transformers import AutoTokenizer, AutoModel, pipeline
tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT_sentiment_analysis") #same as 'avichr/heBERT' tokenizer
model = AutoModel.from_pretrained("avichr/heBERT_sentiment_analysis")

# how to use?
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="avichr/heBERT_sentiment_analysis",
    tokenizer="avichr/heBERT_sentiment_analysis",
    return_all_scores = True
)

>>>  sentiment_analysis('אני מתלבט מה לאכול לארוחת צהריים')	
[[{'label': 'natural', 'score': 0.9978172183036804},
{'label': 'positive', 'score': 0.0014792329166084528},
{'label': 'negative', 'score': 0.0007035882445052266}]]

>>>  sentiment_analysis('קפה זה טעים')
[[{'label': 'natural', 'score': 0.00047328314394690096},
{'label': 'possitive', 'score': 0.9994067549705505},
{'label': 'negetive', 'score': 0.00011996887042187154}]]

>>>  sentiment_analysis('אני לא אוהב את העולם')
[[{'label': 'natural', 'score': 9.214012970915064e-05}, 
{'label': 'possitive', 'score': 8.876807987689972e-05}, 
{'label': 'negetive', 'score': 0.9998190999031067}]]
```

**Let's discuss the results:**

**Sentiment on Source Data:**

| Group | Positive | Negative  | Neutral  |  
|-------|----------|-----------|----------|
| A     | 5.56 %   | 88.36 %   | 6.08 %   |
| B     | 5.1 %    | 90.5 %    | 4.4 %    |
| C     | 5.08 %   | 88.64 %   | 6.28 %   | 

**Sentiment on Data Without Real Names:**

| Group | Positive | Negative | Neutral |  
|-------|----------|----------|---------|
| A     | 6.34 %   | 86.32 %  | 7.34 %  |
| B     | 5.58 %   | 89 %     | 5.42 %  |
| C     | 5.68 %   | 86.4 %   | 7.92 %  | 

**Conclusion:**

We can see that ignoring real names does not affect the classification much.

Another conclusion is that the texts are very negative for the 3 groups.

> **Sentiment Task Using 'AlephBertGimmel-Sentiment':**

**Liav Ariel was Assigned to this task.**

**Script path:** 

1. './tag_with_ABG.py'

**Output path:** 

1. './output/tagged_ABG_15000_docs/tagged_with_ABG_A.xlsx'
2. './output/tagged_ABG_15000_docs/tagged_with_ABG_B.xlsx'
3. './output/tagged_ABG_15000_docs/tagged_with_ABG_C.xlsx'
4. './output/tagged_ABG_15000_docs/result_ABG_A.json'
5. './output/tagged_ABG_15000_docs/result_ABG_B.json'
6. './output/tagged_ABG_15000_docs/result_ABG_C.json'
7. './output/tagged_ABG_15000_without_names_docs/tagged_ABG_without_names_A.xlsx'
8. './output/tagged_ABG_15000_without_names_docs/tagged_ABG_without_names_B.xlsx'
9. './output/tagged_ABG_15000_without_names_docs/tagged_ABG_without_names_C.xlsx'
10. './output/tagged_ABG_15000_without_names_docs/result_ABG_without_names_A.json'
11. './output/tagged_ABG_15000_without_names_docs/result_ABG_without_names_B.json'
12. './output/tagged_ABG_15000_without_names_docs/result_ABG_without_names_C.json'

I do the same as before, just now I work on hebrew data with AlephBertGimmel-Sentiment pre-trained model.
First, on the source data.
Second, on the data without names.

**Link for the model:**

1. https://huggingface.co/Perlucidus/alephbertgimmel-base-sentiment

**How to Use:**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Perlucidus/alephbertgimmel-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("Perlucidus/alephbertgimmel-base-sentiment")

# Tokenize the text
text = "This is a great movie!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Perform forward pass to get the logits
outputs = model(**inputs)

# Convert logits to probabilities
probs = outputs['logits'].softmax(dim=-1)

# Get the predicted sentiment label
predicted_label = probs.argmax().item()

# Define the mapping of sentiment labels
sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Print the sentiment label
print(sentiment_labels[predicted_label])
```

**Let's discuss the results:**

**Sentiment on Source Data:**

| Group | Positive | Negative | Neutral |  
|-------|----------|----------|---------|
| A     | 0 %      | 24.44 %  | 75.56 % |
| B     | 0 %      | 25.92 %  | 74.08 % |
| C     | 0 %      | 30.12 %  | 69.88 % | 

**Sentiment on Data Without Real Names:**

| Group | Positive | Negative | Neutral |  
|-------|----------|----------|---------|
| A     | 0 %      | 26.54 %  | 73.46 % |
| B     | 0 %      | 27.92 %  | 72.08 % |
| C     | 0 %      | 30.62 %  | 69.38 % | 

**Conclusion:**

Something about this model isn't well-trained, but it's the only model I found in Hugging Face. 

It was also very strange to discover that there was no 'model card' on it and only a link to download the model. 

The results are quite delusional. 

As far as I'm concerned, he didn't succeed. 

But he did manage to show that there is no fundamental difference if real names are removed or replaced with random names. 

The results are still the same.
