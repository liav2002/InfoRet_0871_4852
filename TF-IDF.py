import pandas as pd
from collections import Counter
import string
import openpyxl

CONTENT_COL  = 2
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


def get_cells_in_range(excel_file_path, sheet_name, start_row, end_row, column_index, output_excel_path):
    column_values = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=None, usecols=[column_index], skiprows=start_row-1, nrows=end_row-start_row+1)[column_index]
    all_text = ' '.join(map(str, column_values))
    word_frequency = Counter(all_text.split())
    result_list = list(word_frequency.items())
    result_list.sort(key=lambda x: x[1], reverse=True)
    df_output = pd.DataFrame(result_list, columns=["Word", "Count"])
    df_output.to_excel(output_excel_path, index=False)
    return result_list



def c(w, d):
    return d.count(w)



def main():
    A_frequency_result = get_cells_in_range(FILE_PATH, SHEET, A_START, A_FINISH, CONTENT_COL, A_OUTPUT_PATH)
    B_frequency_result = get_cells_in_range(FILE_PATH, SHEET, B_START, B_FINISH, CONTENT_COL, B_OUTPUT_PATH)
    C_frequency_result = get_cells_in_range(FILE_PATH, SHEET, C_START, C_FINISH, CONTENT_COL, C_OUTPUT_PATH)
    print("Word frequency in the specified range:")


if __name__ == "__main__":
    main()