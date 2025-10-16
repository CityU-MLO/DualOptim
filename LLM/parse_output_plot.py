import os
import pandas as pd
import math
import numpy as np

root_dir = "/path/to/closer-look-LLM-unlearning/results/tofu_phi1-5_final_steps"
# Initialize a list to store the results
results = []

filter_method = [
    # "GA+GD",
    # "DPO+GD"
    "ME+GD",
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
        metrics = ["Model Utility", "Forget Efficacy", "unlearn_step"]

        # Ensure all columns exist in the DataFrame
        available_metrics = [col for col in metrics if col in df.columns]
        df = df[available_metrics]

        # Get the relative path to the CSV file for identification
        relative_path = os.path.relpath(dirpath, root_dir)

        # Split the path into "Set" and "Method"
        path_parts = relative_path.split("/")
        set_name = path_parts[-5]  # First part is the set (e.g., forget01)
        method_name = path_parts[-4]  # Second part is the method (e.g., ME+GD)
        type = path_parts[-6]
        if type == "ours":
            method_name += "+DO"

        # df = df[:1]
        # Add the extracted data to the results list
        for _, row in df.iterrows():
            if method_name:
                result = {"Set": set_name, "Method": method_name}
                # Add all available metrics
                for metric in available_metrics:
                    result[metric] = row[metric]
                results.append(result)

# Convert the results list to a DataFrame for easier manipulation
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
output_csv_path = "parsed_results_with_metrics_1.csv"
# results_df.to_csv(output_csv_path, index=False)
# print(results_df)
# print(f"Results have been saved to {output_csv_path}")

# esults_df = pd.read_csv(output_csv_path)
import matplotlib.pyplot as plt
import seaborn as sns

# Define colors corresponding to methods
method_colors = {
    "GA+GD": "#E6194B",  # Vivid Green
    "DPO+GD": "#3CB44B",  # Strong Red
    "IDK+AP": "#FFE119",  # Bright Yellow
    "ME+GD": "#4363D8",  # Intense Blue
    "ME+GD+DO": "#911EB4",  # Bold Orange
    "IDK+AP+DO": "#F58231",  # Deep Purple
}


# Create the plot
plt.figure(figsize=(15, 5))

# Unique sets for subplots
sets = results_df["Set"].unique()

for i, s in enumerate(sets, 1):
    plt.subplot(1, len(sets), i)
    subset = results_df[results_df["Set"] == s]

    for method in subset["Method"].unique():
        method_data = subset[subset["Method"] == method]
        # import pdb;pdb.set_trace
        plt.plot(
            method_data["Model Utility"],
            method_data["Forget Efficacy"],
            marker="o",
            linestyle="-",
            label=method,
            color=method_colors.get(method, "black"),
        )

    plt.axhline(
        y=1.0,
        color="black",
        linestyle="dashed",
        linewidth=2,
        label="Random Initialization",
    )
    plt.title(s)
    plt.xlabel("Model Utility")
    if i == 1:
        plt.ylabel("Forget Efficacy")

plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("out.png")


# Create two separate plots based on method grouping

# Group 1: ["GA+GD", "ME+GD", "ME+GD+DO"]
group1_methods = ["GA+GD", "ME+GD", "ME+GD+DO"]

# Group 2: ["DPO+GD", "IDK+AP", "IDK+AP+DO"]
group2_methods = ["DPO+GD", "IDK+AP", "IDK+AP+DO"]


def plot_method_group(method_group, filename):
    num_sets = len(results_df["Set"].unique())
    fig, axes = plt.subplots(1, num_sets, figsize=(5 * num_sets + 2, 7), sharey=True)

    for ax, s in zip(axes, results_df["Set"].unique()):
        subset = results_df[results_df["Set"] == s]

        for method in method_group:
            method_data = subset[subset["Method"] == method]
            method_data["unlearn_step_norm"] = (
                method_data["unlearn_step"] / method_data["unlearn_step"].max()
            )
            sizes = (
                np.log1p(method_data["unlearn_step_norm"]) * 300
            )  # Scale normalized step for visibility

            # Plot line connecting points
            ax.plot(
                method_data["Model Utility"],
                method_data["Forget Efficacy"],
                linestyle="-",
                color=method_colors.get(method, "black"),
                alpha=0.6,
                linewidth=3.8,
            )

            # Plot circles
            ax.scatter(
                method_data["Model Utility"],
                method_data["Forget Efficacy"],
                s=sizes,
                marker="o",
                label=method,
                color=method_colors.get(method, "black"),
                alpha=0.6,
            )

        # if "IDK+AP" not in method_group:
        #     ax.axhline(y=1.0, color='black', linestyle='dashed', linewidth=2, label="Random Initialization")
        ax.set_title(s, fontsize=18)
        # ax.set_xlabel("Model Utility", fontsize=18)
        # ax.set_ylabel("Forget Efficacy", fontsize=18)
        ax.set_xlim(-0.05, 0.9)
        ax.set_ylim(0.3, 0.88)  # Add some margin at the top of y-axis
        ax.tick_params(axis="both", which="major", labelsize=22)
        ax.grid(True, linestyle="--", alpha=0.7)

    fig.text(0.5, 0.02, "Model Capacity", ha="center", va="center", fontsize=22)
    fig.text(
        0.02,
        0.5,
        "Forget Efficacy",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=22,
    )

    # Move legend to the top in one line
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=len(method_group) + 1,
        fontsize=22,
    )

    # Adjust layout and save as PDF
    plt.tight_layout(
        rect=[0.04, 0.04, 1, 0.9]
    )  # Adjust the rect to make space for the legend
    plt.savefig(filename, format="pdf")
    plt.show()


# Generate and save the two plots
plot_method_group(group1_methods, "plot_group1.pdf")
plot_method_group(group2_methods, "plot_group2.pdf")


# # Group by the "Set" column
# grouped = results_df.groupby("Set")

# # Initialize a dictionary to store the final results
# final_results = {}

# # Iterate over each group (each set)
# for set_name, group in grouped:
#     # Compute the average and standard deviation for each metric
#     avg_std_dict = {}
#     for metric in group.columns[2:]:  # Skip "Set" and "Method" columns
#         avg = group[metric].mean()
#         std = group[metric].std()
#         if math.isnan(std):
#             avg_std_dict[metric] = f"{avg:.4f}"
#         else:
#             avg_std_dict[metric] = f"{avg:.4f}±{std:.4f}"

#     # Store the results for this set
#     final_results[set_name] = avg_std_dict

# # Convert the final results to a DataFrame for better visualization
# final_results_df = pd.DataFrame(final_results).T

# # Save the final results to a new CSV file
# final_output_csv_path = "final_avg_std_results.csv"
# final_results_df.to_csv(final_output_csv_path)

# # Print the final results
# print("\nFinal Results (avg±std):")
# print(final_results_df)
# print(f"Final results have been saved to {final_output_csv_path}")
