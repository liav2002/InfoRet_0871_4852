from collections import defaultdict
from tqdm import tqdm
from Utills import *
import string
import gzip
import json
import time

# GLOBAL VARIABLES
SOURCE_DATA = "./data/15000.xlsx"
DATA_CLEANED_FROM_PUNCTUATIONS = "./data/clean_punctuations_15000.xlsx"
DATA_CLEANED_FROM_STOPWORDS = "./data/full_clean_15000.xlsx"
OSCAR_DIRECTORY_PATH = "C:/Users/liavm/OneDrive - g.jct.ac.il/Year D/Annual courses/Bert2Vec - Final Project/Task 2 - Find Datasets of ABG Traning/OSCAR/"
STOP_WORDS_OSCAR_JSON = "./stop_words/stopwords_from_oscar.json"
STOP_WORDS_WIKI_CSV = "./stop_words/top_3000_most_freq_wiki.csv"


####################################################################################################################

# REMOVE PUNCTUATIONS METHODS
def add_spaces_around_punctuation(word):
    # Check if the word represents a time (HH:MM or HH:MM:SS)
    is_time = ":" in word and all(part.isdigit() for part in word.split(":"))
    # Check if the word represents a date (DD/MM/YYYY or DD/MM/YY)
    is_date = "/" in word and all(part.isdigit() for part in word.replace("/", "").split())

    if is_time or is_date:
        if word.startswith(tuple(string.punctuation)):
            word = word[0] + ' ' + word[1:]
        if word.endswith(tuple(string.punctuation)):
            word = word[:len(word) - 1] + ' ' + word[len(word) - 1]
        # If the word represents time or date, return it as is
        return word

    result = []
    is_acronym = False
    i = 0
    for index, char in enumerate(word):
        if (char == '"' or char == "'") and 0 < i < len(word) - 1 and not is_acronym:
            if index < len(word) - 1 and word[index + 1] in string.punctuation:
                result.append(f' {char} ')
            else:
                is_acronym = True
                result.append(char)
        elif char in string.punctuation:
            result.append(f' {char} ')
            i -= 1
        else:
            result.append(char)
        i += 1

    return ''.join(result)


def add_spaces_between_punctuations(text):
    if pd.isna(text):
        # If the phrase is nan (empty cell), return it as is
        return text

    sentences = text.split('\n')
    cleaned_sentences = []
    for s in sentences:
        cleaned_sentences.append(' '.join([add_spaces_around_punctuation(w) for w in s.split(' ')]))
    cleaned_text = '\n'.join(cleaned_sentences)

    return cleaned_text


def clean_punctuations_and_save_excel(input_file_path=SOURCE_DATA,
                                      output_file_path=DATA_CLEANED_FROM_PUNCTUATIONS):
    # Read the Excel file into a DataFrame
    df = read_excel_file(input_file_path)

    # Apply remove_punctuations function to clean the content column
    df['תוכן הקובץ'] = df['תוכן הקובץ'].apply(add_spaces_between_punctuations)

    # Save the cleaned DataFrame to a new Excel file
    df.to_excel(output_file_path, index=False)


####################################################################################################################

# REMOVE STOPWORDS METHODS
def count_hebrew_words(text, hebrew_words_count):
    words = text.split()
    for word in words:
        if all('\u0590' <= letter <= '\u05EA' for letter in word):  # Check if the word is in Hebrew
            hebrew_words_count[word] += 1


def process_file_for_stop_words(file_path, hebrew_words_count):
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        content = file.read()
        content_subset = content[:int(len(content) * 0.1)]
        count_hebrew_words(content_subset, hebrew_words_count)


def save_top_words_to_json(hebrew_words_count, output_file):
    sorted_words = sorted(hebrew_words_count.items(), key=lambda x: x[1], reverse=True)
    top_words = dict(sorted_words[:3000])

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(top_words, json_file, ensure_ascii=False, indent=4)


def generate_stop_words_list_from_oscar(oscar_data=OSCAR_DIRECTORY_PATH, output=STOP_WORDS_OSCAR_JSON):
    print("Generate Stopwords JSON Form OSCAR")
    hebrew_words_count = defaultdict(int)

    # Start the timer
    start_time = time.time()

    for i in tqdm(range(1, 12), desc='Processing OSCAR Files', unit='file'):
        file_name = f'he_part_{i}.txt.gz'
        file_path = os.path.join(oscar_data, file_name)
        process_file_for_stop_words(file_path, hebrew_words_count)

    # Stop the timer
    end_time = time.time()
    elapsed_time = end_time - start_time

    save_top_words_to_json(hebrew_words_count, output)
    print(f'Stop words saved to {output}')

    # Print the elapsed time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")


def intersect_stop_words(oscar_json=STOP_WORDS_OSCAR_JSON, wiki_csv=STOP_WORDS_WIKI_CSV):
    # Read stop words from the Oscar dataset
    with open(oscar_json, 'r', encoding='utf-8') as file:
        stop_words_oscar = set(json.load(file).keys())

    # Read stop words from the Wiki dataset
    stop_words_wiki_df = pd.read_csv(wiki_csv)
    stop_words_wiki = set(stop_words_wiki_df['word'].values)

    # Find the intersection of stop words
    intersection_stop_words = list(stop_words_oscar.intersection(stop_words_wiki))

    return intersection_stop_words


def remove_stopwords(text):
    if pd.isna(text):
        # If the phrase is nan (empty cell), return it as is
        return text

    stop_words = intersect_stop_words()
    sentences = text.split("\n")
    cleaned_sentences = []

    for s in sentences:
        words = s.split()
        filtered_words = [w for w in words if w not in stop_words]
        cleaned_s = ' '.join(filtered_words)
        cleaned_sentences.append(cleaned_s)

    cleaned_text = '\n'.join(cleaned_sentences)

    return cleaned_text


def clean_stopwords_and_save_excel(input_file_path=DATA_CLEANED_FROM_PUNCTUATIONS,
                                   output_file_path=DATA_CLEANED_FROM_STOPWORDS):
    # Read the Excel file into a DataFrame
    df = read_excel_file(input_file_path)

    # Apply remove_punctuations function to clean the content column
    df['תוכן הקובץ'] = df['תוכן הקובץ'].apply(remove_stopwords)

    # Save the cleaned DataFrame to a new Excel file
    df.to_excel(output_file_path, index=False)


####################################################################################################################

# MAIN FUNCTION
def main():
    try:
        if not all_files_exist([DATA_CLEANED_FROM_PUNCTUATIONS]):
            print("Work on remove punctuations.")
            clean_punctuations_and_save_excel()
            print(f"data clean from punctuations saved in: {DATA_CLEANED_FROM_PUNCTUATIONS}")
        if not all_files_exist([STOP_WORDS_OSCAR_JSON]):
            print("Work on generate stopwords list.")
            generate_stop_words_list_from_oscar()
            print(f"oscar stopwords json saved in: {STOP_WORDS_OSCAR_JSON}")
        if not all_files_exist([DATA_CLEANED_FROM_STOPWORDS]):
            print("Work on remove stopwords.")
            clean_stopwords_and_save_excel()
            print(f"data clean from stopwords saved in: {DATA_CLEANED_FROM_STOPWORDS}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    main()
