import pandas as pd
from collections import Counter
import string
import openpyxl
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows

CONTENT_COL  = 2
ID_COL = 1
A_START = 2
A_FINISH = 5001
B_START = 5006
B_FINISH = 10005
C_START = 10010
C_FINISH = 15009
FILE_PATH = "C:\\Users\\nehor\\OneDrive - click\\שולחן העבודה\\InfoRet_0871_4852\\data\\clean_15000.xlsx"
SHEET = "Sheet1"
A_OUTPUT_PATH = "C:\\Users\\nehor\\Downloads\\A_CLEAN_VOCA.xlsx"
B_OUTPUT_PATH = "C:\\Users\\nehor\\Downloads\\B_CLEAN_VOCA.xlsx"
C_OUTPUT_PATH = "C:\\Users\\nehor\\Downloads\\C_CLEAN_VOCA.xlsx"
A_LEN_OUTPUT_PATH = "C:\\Users\\nehor\\Downloads\\A_CLEAN_LEN.xlsx"
B_LEN_OUTPUT_PATH = "C:\\Users\\nehor\\Downloads\\B_CLEAN_LEN.xlsx"
C_LEN_OUTPUT_PATH = "C:\\Users\\nehor\\Downloads\\C_CLEAN_LEN.xlsx"


def get_cells_in_range(excel_file_path, sheet_name, start_row, end_row, column_index, output_excel_path):
    column_values = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None, usecols=[column_index], skiprows=start_row-1, nrows=end_row-start_row+1)[column_index]
    all_text = ' '.join(map(str, column_values))
    word_frequency = Counter(all_text.split())
    result_list = list(word_frequency.items())
    result_list.sort(key=lambda x: x[1], reverse=True)
    df_output = pd.DataFrame(result_list, columns=["Word", "Count"])
    df_output.to_excel(output_excel_path, index=False)
    return result_list


def count_words_in_cell(cell_value):
    # Count the number of words in a cell
    words = cell_value.split()
    return len(words)

def get_cells_in_range(excel_file_path, sheet_name, start_row, end_row, column_index, output_excel_path):
    # Read Excel file using pandas
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

    # Create a new DataFrame with only the required columns
    result_df = pd.DataFrame(columns=["מזהה/מספר הקובץ", "paragraph number of words"])

    # Iterate through the specified range and count words in each cell
    for index, row in df.iloc[start_row - 2:end_row - 1].iterrows():
        paragraph_id = row["מזהה/מספר הקובץ"]
        cell_value = row["תוכן הקובץ"]
        word_count = count_words_in_cell(str(cell_value))

        # Append the results to the new DataFrame
        result_df = pd.concat([result_df, pd.DataFrame({"מזהה/מספר הקובץ": [paragraph_id], "paragraph number of words": [word_count]})],
                              ignore_index=True)

    # Write the new DataFrame to a new Excel file
    result_df.to_excel(output_excel_path, index=False)
#---------------------------------------------------------------------------------------------------------------
# def count_words_in_cell(cell_value):
#     # Count the number of words in a cell
#     words = cell_value.split()
#     return len(words)
#
# def get_cells_in_range(excel_file_path, sheet_name, start_row, end_row, column_index, output_excel_path):
#     # Read Excel file using pandas
#     df = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None)
#
#     # Iterate through the specified range and count words in each cell
#     for index, row in df.iloc[start_row-1:end_row].iterrows():
#         cell_value = row[column_index]
#         word_count = count_words_in_cell(str(cell_value))
#         df.at[index, column_index-1] = word_count
#
#     # Write the updated DataFrame to a new Excel file
#     df.to_excel(output_excel_path, index=False)
#---------------------------------------------------------------------------------------------------------------
# def get_cells_in_range(excel_file_path, sheet_name, start_row, end_row, column_index, output_excel_path):
#     # Read the Excel file into a DataFrame
#     df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
#
#     # Ensure that start_row and end_row are within the DataFrame's range
#     start_row = max(0, start_row - 1)
#     end_row = min(end_row, len(df))
#
#     # Extract the specified range of rows and the specified column
#     selected_data = df.iloc[start_row:end_row, column_index]
#
#     # Count the number of words in each cell and store the result in a new column
#     selected_data['Word Count'] = selected_data.apply(lambda cell: len(str(cell).split()))
#
#     # Save the updated DataFrame to a new Excel file
#     selected_data.to_excel(output_excel_path, index=False)
#---------------------------------------------------------------------------------------------------------------




def c(w, d):
    return d.count(w)



def main():
    # A_frequency_result = get_cells_in_range(FILE_PATH, SHEET, A_START, A_FINISH, CONTENT_COL, A_OUTPUT_PATH)
    # B_frequency_result = get_cells_in_range(FILE_PATH, SHEET, B_START, B_FINISH, CONTENT_COL, B_OUTPUT_PATH)
    # C_frequency_result = get_cells_in_range(FILE_PATH, SHEET, C_START, C_FINISH, CONTENT_COL, C_OUTPUT_PATH)

    get_cells_in_range(FILE_PATH, SHEET, A_START, A_FINISH, CONTENT_COL, A_LEN_OUTPUT_PATH)

    print("Word frequency in the specified range:")


if __name__ == "__main__":
    main()