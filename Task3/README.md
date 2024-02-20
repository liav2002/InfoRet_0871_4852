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

> **Sentiment On Source Data Using 'heBert-Sentiment':**

**Nehorai Josef was Assigned to this task.**

**Script path:** 

1. './tag_translated_data.py'

**Output path:** 

1. './output/tagged_translated_15000_docs/tagged_translated_A.xlsx'
2. './output/tagged_translated_15000_docs/tagged_translated_B.xlsx'
3. './output/tagged_translated_15000_docs/tagged_translated_C.xlsx'
4. './output/tagged_translated_15000_docs/result_translated_A.json'
5. './output/tagged_translated_15000_docs/result_translated_B.json'
6. './output/tagged_translated_15000_docs/result_translated_C.json'