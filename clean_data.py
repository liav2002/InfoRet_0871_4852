import pandas as pd
import string


def read_excel_file(file_path="./data/15000.xlsx"):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    return df


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


def clean_and_save_excel(input_file_path="./data/15000.xlsx", output_file_path="./data/clean_15000.xlsx"):
    # Read the Excel file into a DataFrame
    df = read_excel_file(input_file_path)

    # Apply remove_punctuations function to clean the content column
    df['תוכן הקובץ'] = df['תוכן הקובץ'].apply(add_spaces_between_punctuations)

    # Save the cleaned DataFrame to a new Excel file
    df.to_excel(output_file_path, index=False)


clean_and_save_excel()
