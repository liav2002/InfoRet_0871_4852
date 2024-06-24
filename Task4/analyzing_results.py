import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_f1_graph(model_type):
    base_dir = f'./output/{model_type}'
    iterations = range(1, 11)
    file_patterns = {
        'result_A&B': 'result_A&B-{}.json',
        'result_A&C': 'result_A&C-{}.json',
        'result_B&C': 'result_B&C-{}.json'
    }

    # Dictionary to store the f1 scores
    f1_scores = {key: [] for key in file_patterns}

    # Read the f1 scores from the JSON files
    for i in iterations:
        for key, pattern in file_patterns.items():
            file_name = pattern.format(i)
            file_path = os.path.join(base_dir, f"Iteration{i}", file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
                f1_scores[key].append(data['f1'])

    # Plot the f1 scores
    plt.figure(figsize=(10, 6))
    for key in file_patterns:
        plt.plot(iterations, f1_scores[key], label=key)

    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores for SVM Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.xticks(iterations)

    # Save the plot to a file
    output_path = os.path.join(base_dir, 'SVM_f1_scores.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")


def summary_iterations(model_type):
    base_dir = f'./output/{model_type}'
    iterations = range(1, 11)
    file_patterns = {
        'A&B': 'result_A&B-{}.json',
        'A&C': 'result_A&C-{}.json',
        'B&C': 'result_B&C-{}.json'
    }

    # List to store the summary data
    summary_data = []

    # Read the metrics from the JSON files
    for i in iterations:
        iteration_data = {'Iteration Number': i}
        for key, pattern in file_patterns.items():
            file_name = pattern.format(i)
            file_path = os.path.join(base_dir, f"Iteration{i}", file_name)
            with open(file_path, 'r') as f:
                data = json.load(f)
                iteration_data[f'accuracy_{key}'] = data['accuracy']
                iteration_data[f'precision_{key}'] = data['precision']
                iteration_data[f'recall_{key}'] = data['recall']
                iteration_data[f'f1_{key}'] = data['f1']
        summary_data.append(iteration_data)

    # Create a DataFrame from the summary data
    columns = ['Iteration Number',
               'accuracy_A&B', 'precision_A&B', 'recall_A&B', 'f1_A&B',
               'accuracy_A&C', 'precision_A&C', 'recall_A&C', 'f1_A&C',
               'accuracy_B&C', 'precision_B&C', 'recall_B&C', 'f1_B&C']

    df = pd.DataFrame(summary_data, columns=columns)

    # Save the DataFrame to an Excel file
    output_path = os.path.join(base_dir, 'iteration_results_summary.xlsx')
    df.to_excel(output_path, index=False)

    print(f"Excel file saved to {output_path}")

plot_f1_graph("SVM")
summary_iterations("SVM")
