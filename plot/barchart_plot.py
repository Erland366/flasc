import os
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Configuration ---
ENTITY = "erlandpg" # Replace with your W&B entity name
PROJECT = "federated_merging" # Replace with your W&B project name

# List the names of the runs you want to compare runtime for
RUN_NAMES_TO_COMPARE = [
    "train_gpt2_seed2_c100_constant lr_r100_e1_early2.0_average_server yogi_client sgd",
    "train_gpt2_seed2_c100_constant lr_r100_e1_early2.0_fisher_merging_server yogi_client sgd",
    # Add more run names here if needed
]

# Optional: Provide custom display names for the bars (must match order and length of RUN_NAMES_TO_COMPARE)
CUSTOM_DISPLAY_NAMES = [
    "Average",
    "Fisher Merging",
    # Add more custom names here if needed
]
# Set to None to use the default W&B run names
# CUSTOM_DISPLAY_NAMES = None

PLOT_SAVE_DIR = "plots"

# --- Helper Function to Get Runtimes ---
def get_run_runtimes(api, entity, project, run_names, custom_display_names=None):
    """
    Fetches the final runtime for a list of W&B runs by name.

    Args:
        api: Initialized W&B API object.
        entity (str): W&B entity name.
        project (str): W&B project name.
        run_names (list): List of W&B run names (display names) to fetch.
        custom_display_names (list, optional): List of custom names for display.
                                               Defaults to None, using run_names.

    Returns:
        dict: A dictionary where keys are display names and values are final runtimes (float),
              or an empty dictionary if no runtimes are found.
    """
    runtimes = {}
    full_project_path = f"{entity}/{project}"
    print(f"Fetching runtimes for runs in project: {full_project_path}")

    for i, run_name_to_find in enumerate(run_names):
        # Determine the display name to use in the output and chart
        display_name = custom_display_names[i] if custom_display_names and i < len(custom_display_names) else run_name_to_find
        print(f"\nSearching for run name: '{run_name_to_find}' (Display: '{display_name}')...")

        try:
            # Filter runs by display name
            # Note: Filtering by display_name assumes display names are unique enough
            # You might need to use run IDs if display names are not unique
            runs_found = api.runs(full_project_path, filters={"display_name": run_name_to_find})

            if runs_found:
                # Assuming the first found run with this display name is the correct one
                run = runs_found[0]
                print(f"  Found run: '{run.name}' (ID: {run.id})")

                # Fetch only the _runtime history
                history = run.history(keys=['_runtime'], pandas=True)

                if '_runtime' in history.columns and not history.empty:
                    # Get the maximum value in the _runtime column, which should be the final runtime
                    final_runtime = history['_runtime'].max()
                    runtimes[display_name] = final_runtime
                    print(f"  Found final runtime: {final_runtime:.2f} seconds")
                else:
                    print(f"  WARNING: '_runtime' data not found or is empty for run '{display_name}' ('{run_name_to_find}').")
            else:
                print(f"  WARNING: No run found with name '{run_name_to_find}' in project '{ENTITY}/{PROJECT}'.")

        except Exception as e:
            print(f"  ERROR fetching runtime for run '{display_name}' ('{run_name_to_find}'): {e}")
            import traceback
            traceback.print_exc()

    return runtimes

# --- Function to Create Bar Chart ---
def create_runtime_bar_chart(runtimes, title="Total Training Runtime Comparison", xlabel="Method", ylabel="Runtime (seconds)", save_dir="plots"):
    """
    Creates and displays a bar chart of runtimes, including speed difference annotations.

    Args:
        runtimes (dict): Dictionary of method display names and their runtimes.
        title (str, optional): Title of the chart. Defaults to "Total Training Runtime Comparison".
        xlabel (str, optional): Label for the x-axis. Defaults to "Method".
        ylabel (str, optional): Label for the y-axis. Defaults to "Runtime (seconds)".
        save_dir (str, optional): Directory to save the plot. Defaults to "plots".
    """
    if not runtimes:
        print("No runtime data available to plot the bar chart.")
        return

    # Prepare data for plotting
    methods = list(runtimes.keys())
    durations = list(runtimes.values())

    # Sort methods by duration (ascending) for better visualization and base calculation
    sorted_indices = np.argsort(durations)
    methods = [methods[i] for i in sorted_indices]
    durations = [durations[i] for i in sorted_indices]

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, durations, color='skyblue')

    # Set chart title and labels
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels if they are long
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    # Add annotations: Runtime values and Speed Difference
    if len(durations) > 0:
        # Use the shortest runtime as the base for calculating the ratio
        base_runtime = durations[0]
        base_method = methods[0]
        print(f"\nUsing '{base_method}' ({base_runtime:.2f}s) as the base for speed difference calculation.")

        for i, bar in enumerate(bars):
            height = bar.get_height()
            method_name = methods[i]

            # Annotate with the exact runtime value
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}s',
                     ha='center', va='bottom', fontsize=10)

            # Annotate with the speed difference (ratio to the base runtime)
            if base_runtime > 0:
                ratio = height / base_runtime
                if ratio >= 1.0:
                    # For methods slower than or equal to the base, show how many times slower
                    ratio_text = f'{ratio:.2f}x slower' if ratio > 1.0 else 'Base'
                else:
                    # For methods faster than the base, show how many times faster they are
                    ratio_text = f'{1/ratio:.2f}x faster'

                # Use annotate for relative positioning with xytext
                plt.annotate(ratio_text,
                             xy=(bar.get_x() + bar.get_width()/2., height), # Anchor point at top center of bar
                             xytext=(0, 15), # Offset text 15 points upwards from anchor
                             textcoords='offset points',
                             ha='center', va='bottom', color='red', fontsize=9)


    # Add a horizontal grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Save and display the plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "runtime_bar_chart.png")
    print(f"Saving bar chart to '{save_path}'.")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("--- Bar Chart Generation Complete ---")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration Validation ---
    if not ENTITY or ENTITY == "your_entity":
        print("ERROR: Please update ENTITY with your W&B entity name.")
    elif not PROJECT or PROJECT == "your_project":
        print("ERROR: Please update PROJECT with your W&B project name.")
    elif not RUN_NAMES_TO_COMPARE:
        print("ERROR: Please specify at least one run name in RUN_NAMES_TO_COMPARE.")
    elif CUSTOM_DISPLAY_NAMES and len(CUSTOM_DISPLAY_NAMES) != len(RUN_NAMES_TO_COMPARE):
        print(f"ERROR: Length mismatch: CUSTOM_DISPLAY_NAMES ({len(CUSTOM_DISPLAY_NAMES)}) must match RUN_NAMES_TO_COMPARE ({len(RUN_NAMES_TO_COMPARE)}).")
    else:
        print("Configuration looks valid. Proceeding with runtime fetching and plotting.")
        try:
            # Initialize W&B API
            api = wandb.Api(timeout=60) # Increased timeout for API calls
            print("W&B API initialized.")

            # Fetch runtimes
            runtimes_data = get_run_runtimes(api, ENTITY, PROJECT, RUN_NAMES_TO_COMPARE, CUSTOM_DISPLAY_NAMES)

            # Create and display the bar chart
            create_runtime_bar_chart(runtimes_data, save_dir=PLOT_SAVE_DIR)

        except Exception as e:
            print(f"An error occurred during the process: {e}")
            import traceback
            traceback.print_exc()

