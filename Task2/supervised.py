import pandas as pd

# Data Files

# Doc vectors of bert on source.
BERT_SOURCE_A_VECTORS = "./data/bert_on_source/bert_source_A_docvec.xlsx"
BERT_SOURCE_B_VECTORS = "./data/bert_on_source/bert_source_B_docvec.xlsx"
BERT_SOURCE_C_VECTORS = "./data/bert_on_source/bert_source_C_docvec.xlsx"

# Doc vectors of d2v on source.
D2V_SOURCE_A_VECTORS = "./data/d2v_on_source/d2v_source_A_docvec.xlsx"
D2V_SOURCE_B_VECTORS = "./data/d2v_on_source/d2v_source_B_docvec.xlsx"
D2V_SOURCE_C_VECTORS = "./data/d2v_on_source/d2v_source_C_docvec.xlsx"

# Doc vectors of tfidf on lemots.
TFIDF_LEMOTS_A_VECTORS = "./data/tfidf_on_lemots/tfidf_lemots_A_docvec.xlsx"
TFIDF_LEMOTS_B_VECTORS = "./data/tfidf_on_lemots/tfidf_lemots_B_docvec.xlsx"
TFIDF_LEMOTS_C_VECTORS = "./data/tfidf_on_lemots/tfidf_lemots_C_docvec.xlsx"

# Doc vectors of tfidf on words.
TFIDF_WORDS_A_VECTORS = "./data/tfidf_on_words/tfidf_words_A_docvec.xlsx"
TFIDF_WORDS_B_VECTORS = "./data/tfidf_on_words/tfidf_words_B_docvec.xlsx"
TFIDF_WORDS_C_VECTORS = "./data/tfidf_on_words/tfidf_words_C_docvec.xlsx"

# Doc vectors of w2v on lemots.
W2V_LEMOTS_A_VECTORS = "./data/w2v_on_lemots/w2v_lemots_A_docvec.xlsx"
W2V_LEMOTS_B_VECTORS = "./data/w2v_on_lemots/w2v_lemots_B_docvec.xlsx"
W2V_LEMOTS_C_VECTORS = "./data/w2v_on_lemots/w2v_lemots_C_docvec.xlsx"

# Doc vectors of w2v on words.
W2V_WORDS_A_VECTORS = "./data/w2v_on_words/w2v_words_A_docvec.xlsx"
W2V_WORDS_B_VECTORS = "./data/w2v_on_words/w2v_words_B_docvec.xlsx"
W2V_WORDS_C_VECTORS = "./data/w2v_on_words/w2v_words_C_docvec.xlsx"

# Output Folders

# ANN output
ANN_BERT_SOURCE_OUTPUT_FOLDER = "./output/ANN/Bert_On_Source_Groups/"
ANN_D2V_SOURCE_OUTPUT_FOLDER = "./output/ANN/D2V_On_Source_Groups/"
ANN_TFIDF_LEMOTS_OUTPUT_FOLDER = "./output/ANN/TFIDF_On_Lemots_Groups/"
ANN_TFIDF_WORDS_OUTPUT_FOLDER = "./output/ANN/TFIDF_On_Words_Groups/"
ANN_W2V_LEMOTS_OUTPUT_FOLDER = "./output/ANN/W2V_On_Lemots_Groups/"
ANN_W2V_WORDS_OUTPUT_FOLDER = "./output/ANN/W2V_On_Words_Groups/"


def get_vectors_from(file_path):
    df = pd.read_excel(file_path, header=None)
    return [list(df.iloc[i][1:]) for i in range(1, df.shape[0])]
