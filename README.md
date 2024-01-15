# Task 1 - Doc2Vec By (TFIDF, W2V, D2V, BERT)

**Liav Ariel (212830871)**

**Nehorai Josef (322774852)**

We got 3 groups of files stored in 15000.xlsx
Every group with 5000 documets, and total number of documets is 15000.

> **First step was to preprocess the data.**

**Liav Ariel was Assigned to this task.**

Explenation of Steps to clean up the data:

1. Clean data from punctioations:
	I have in directory data\
	the file: 15000.xlsx

	I write script that open this file and add spaces arround punctioations. 
	and I get the output in: data\clean_punctuations.xlsx
	
	Edge cases I referred to:
	- acronym (for example: צה"ל).
	- dates (for example: 15/01/2024).
	- time (for example: 10:30:14).
	
2. Clean data from stop-words:
	After I get the result in data\clean_punctuations.xlsx.
	I write a function thats open the file that cleaned from punctioations and cleaning him from stop-words.
	I get the output in: data\full_clean_15000.xlsx
	
	First, I had to produce the list of stop-words in hebrew.
	
	For do that, I scan 10% from the OSCAR data in hebrew that was taken from: https://huggingface.co/datasets/oscar
	
	I count the frequency of any word in the 10% of OSCAR data, and save the top 3000 most frequencies words into stop_words/stopwords_from_oscar.json
	
	Then, I found a list of top 3000 frequencies hebrew words from Wikipedia: https://github.com/NNLP-IL/Stop-Words-Hebrew/blob/main/top_3000_most_freq_wiki.csv
	and I saved it into: stop_words/top_3000_most_freq_wiki.csv
	
	I do intersection between the both of the lists.
	
	The result is my own hebrew stop-words list.
	
	My function cleaned all the words that contains in my list from data\clean_punctuations.xlsx.
	The output is in: data/full_clean_15000.xlsx.

3. Clean data to conisder only lemot (hebpipe):
	Nehorai Josef was supposed to do that, he install the library of hebpipe, 
	but there was a lot of unexpected errors, 
	so we gave it up.
	
To use my script for cleaning the data, you only need the file data/15000.xlsx in your project directory.
and run clean_data.py
The output results will stored in data\

Then, I handly seperate the files into 3 files (file for each group).
I Create the folder input and save 3 groups for each data: source data, data cleaned from punctioations and data cleaned from punctioations and stop-words. (see input/ folder).

> **Second Step was to create TFIDF Matrix on data cleaned from punctioations and on data cleaned from punctioations and stop-words.**

**Nehorai Josef was Assigned to this task.**

Explenation of Steps to create TFIDF matrix:

Input files are: 
- input/data_cleaned_from_punctuations/dcp_A.xlsx
- input/data_cleaned_from_punctuations/dcp_B.xlsx
- input/data_cleaned_from_punctuations/dcp_C.xlsx
- input/data_cleaned_from_punctuations_and_stopwords/dcps_A.xlsx
- input/data_cleaned_from_punctuations_and_stopwords/dcps_B.xlsx
- input/data_cleaned_from_punctuations_and_stopwords/dcps_C.xlsx

Output files are:
- output/tfidf_on_dcp/dcp_A_vec.xlsx
- output/tfidf_on_dcp/dcp_B_vec.xlsx
- output/tfidf_on_dcp/dcp_C_vec.xlsx
- output/tfidf_on_dcps/dcps_A_vec.xlsx
- output/tfidf_on_dcps/dcps_B_vec.xlsx
- output/tfidf_on_dcps/dcps_C_vec.xlsx

How I create the output:

**TODO: Explain Here**

> **Third Step was to create W2V Matrix on data cleaned from punctioations and on data cleaned from punctioations and stop-words.**

**Liav Ariel was Assigned to this task.**

Explenation of Steps to create W2V matrix:

Input files are: 
- input/data_cleaned_from_punctuations/dcp_A.xlsx
- input/data_cleaned_from_punctuations/dcp_B.xlsx
- input/data_cleaned_from_punctuations/dcp_C.xlsx
- input/data_cleaned_from_punctuations_and_stopwords/dcps_A.xlsx
- input/data_cleaned_from_punctuations_and_stopwords/dcps_B.xlsx
- input/data_cleaned_from_punctuations_and_stopwords/dcps_C.xlsx

Output files are:
- output/w2v_on_dcp/dcp_A_vec.xlsx
- output/w2v_on_dcp/dcp_B_vec.xlsx
- output/w2v_on_dcp/dcp_C_vec.xlsx
- output/w2v_on_dcps/dcps_A_vec.xlsx
- output/w2v_on_dcps/dcps_B_vec.xlsx
- output/w2v_on_dcps/dcps_C_vec.xlsx

How I create the output:

I write the script: doc_vectors_using_w2v.py

the script scan every group, and for every document I geneare the vector using the next algorithem:
for and word in the document - convecrt it into w2v using the vectors from (https://github.com/Ronshm/hebrew-word2vec/blob/master/README.md)
and I sum all the vectors, the total vector is the document vector.
I saved a JSON file of dictionary with the pairs: 
key = doc_id, value = doc_vector (sum of w2v of any word in doc).

> **Forth Step was to create D2V Matrix on source data.**

**TODO: ??? was Assigned to this task.**

Explenation of Steps to create D2V matrix:

Input files are: 
- input/source_data/original_A.xlsx
- input/source_data/original_B.xlsx
- input/source_data/original_C.xlsx

Output files are:
- output/d2v_on_source/source_A_vec.xlsx
- output/d2v_on_source/source_B_vec.xlsx
- output/d2v_on_source/source_C_vec.xlsx

How I create the output:

**TODO: Explain Here**

> Last Step was to create Bert Matrix on source data.

**Liav Ariel was Assigned to this task.**

Explenation of Steps to create Bert matrix:

Input files are: 
- input/source_data/original_A.xlsx
- input/source_data/original_B.xlsx
- input/source_data/original_C.xlsx

Output files are:
- output/bert_on_source/source_A_vec.xlsx
- output/bert_on_source/source_B_vec.xlsx
- output/bert_on_source/source_C_vec.xlsx

How I create the output:

I read the inputs and generate for any doc the vector using alephbertgimel model.
The model know to handle with uncleand data.
He has tokenizer that seperate every sentence to his tokens, I generate the vector of '[CLS]' token,
it is the vector that represent the sentence.
Then, I sum all the sentences vectors -> it's the vector represent the document.
