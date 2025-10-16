import os
import pandas as pd
import math


root_dir = (
    "/path/to/closer-look-LLM-unlearning/results/tofu_phi1-5_npo_gd/phi1.5"
)
# Initialize a list to store the results
results = []

filter_method = [
    # "GA+GD",
    "NPO+GD"
    # "ME+GD",
    # "IDK+AP"
]

# Walk through the directory structure
for dirpath, dirnames, filenames in os.walk(root_dir):
    # Check if the current directory contains the file we're interested in
    if "all_unlearning_results.csv" in filenames:
        # Construct the full path to the CSV file
        csv_path = os.path.join(dirpath, "all_unlearning_results.csv")

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)

        # Extract the relevant columns
        metrics = [
            "Model Utility",
            "Forget Efficacy",
            "Forget ROUGE",
            "Forget Probability",
            "Retain ROUGE",
            "Retain Probability",
        ]

        # Ensure all columns exist in the DataFrame
        available_metrics = [col for col in metrics if col in df.columns]
        df = df[available_metrics]

        # Get the relative path to the CSV file for identification
        relative_path = os.path.relpath(dirpath, root_dir)

        # Split the path into "Set" and "Method"
        path_parts = relative_path.split("/")
        set_name = path_parts[0]  # First part is the set (e.g., forget01)
        method_name = path_parts[1]  # Second part is the method (e.g., ME+GD)

        df = df[:1]
        # Add the extracted data to the results list
        for _, row in df.iterrows():
            if method_name in filter_method:
                result = {"Set": set_name, "Method": method_name}
                # Add all available metrics
                for metric in available_metrics:
                    result[metric] = row[metric]
                results.append(result)

# Convert the results list to a DataFrame for easier manipulation
results_df = pd.DataFrame(results)
results_df["Avg Metric"] = (
    results_df["Model Utility"] + results_df["Forget Efficacy"]
) / 2

# Save the results to a new CSV file
output_csv_path = "parsed_results_with_metrics.csv"
results_df.to_csv(output_csv_path, index=False)
print(results_df)
print(f"Results have been saved to {output_csv_path}")

# Group by the "Set" column
grouped = results_df.groupby("Set")

# Initialize a dictionary to store the final results
final_results = {}

# Iterate over each group (each set)
for set_name, group in grouped:
    # Compute the average and standard deviation for each metric
    avg_std_dict = {}
    for metric in group.columns[2:]:  # Skip "Set" and "Method" columns
        avg = group[metric].mean()
        std = group[metric].std()
        if math.isnan(std):
            avg_std_dict[metric] = f"{avg:.4f}"
        else:
            avg_std_dict[metric] = f"{avg:.4f}±{std:.4f}"

    # Store the results for this set
    final_results[set_name] = avg_std_dict

# Convert the final results to a DataFrame for better visualization
final_results_df = pd.DataFrame(final_results).T

# Save the final results to a new CSV file
final_output_csv_path = "final_avg_std_results.csv"
final_results_df.to_csv(final_output_csv_path)

# Print the final results
print("\nFinal Results (avg±std):")
print(final_results_df)
print(f"Final results have been saved to {final_output_csv_path}")
