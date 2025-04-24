import os
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ENTITY = "erlandpg"
PROJECT = "federated_merging"

## RUN_PATHS IF USING IDS
RUN_PATHS = [
    # "your-run-id-1", # Example
    # "your-run-id-2",
]

seeds = [0, 1, 2]
merging_strategies = ["task_arithmetic", "ties_merging", "fisher_merging", "regmean_merging", "average"]


if ENTITY and PROJECT and RUN_PATHS and "/" not in RUN_PATHS[0]:
      RUN_PATHS = [f"{ENTITY}/{PROJECT}/{run_id}" for run_id in RUN_PATHS]

# Option 2: Specify runs by Name (Requires ENTITY and PROJECT)
# Ensure RUN_NAMES is populated if using this method
RUN_NAMES = [
    # "your-run-name-1", # Example
    # "your-run-name-2",
]

constructed_name = "train_gpt2_seed{seed}_c100_constant lr_r100_e6_early2.0_{merging_strategy}_server yogi_client sgd"
RUN_NAMES = [
      constructed_name.format(seed=0, merging_strategy=merging_strategy) for merging_strategy in merging_strategies
]

# Set RUN_NAMES to the two methods you want to compare runtime for
# RUN_NAMES = [
    # "train_gpt2_seed0_c100_constant lr_r100_e6_early2.0_fisher_merging_server yogi_client sgd",
    # "train_gpt2_seed0_c100_constant lr_r100_e6_early2.0_average_server yogi_client sgd"
# ]

# # Custom names for the legend (must match the order and length of RUN_PATHS or RUN_NAMES)
# CUSTOM_RUN_NAMES = [
#       "BS-1",
#       "BS-8",
#       "BS-16",
#       "BS-32",
#       "BS-64",
# ]
CUSTOM_RUN_NAMES = [
      merging_strategy.replace("_", " ").title() for merging_strategy in merging_strategies
]
# Set CUSTOM_RUN_NAMES to the display names for the two methods
# CUSTOM_RUN_NAMES = [
#     "Task Arithmethic",
#     "Ties Merging",
#     "RegMean Merging",
#     "Fisher Merging",
#     "Average"
# ]
# Set to None if you want to use the default W&B run names
# CUSTOM_RUN_NAMES = None

# Metrics to plot (will generate a separate plot for each)
METRICS_TO_PLOT = [
    "train/loss",
    "train/accuracy",
    "train/recall",
    "train/f1",
    "train/precision",
    "eval/loss",
    "eval/accuracy",
    "eval/recall",
    "eval/f1",
]

# X-axis keys to plot against (e.g., '_step', '_runtime')
# Make sure '_runtime' is included here if you want to calculate runtime multiplication
X_AXIS_KEYS_TO_PLOT = [
    "_step",
    "_runtime", # Include _runtime to enable runtime comparison
]

# For standard EMA (when USE_TIME_WEIGHTED_EMA = False)
SMOOTHING_FACTOR = 0.9 # (alpha = 1 - factor). Set to 0 to disable standard smoothing.


# For Time-Weighted EMA (when USE_TIME_WEIGHTED_EMA = True)
# Specifies the window in terms of time duration (e.g., seconds).
# Set to None or 0 to disable time-weighted smoothing.
SMOOTHING_HALFLIFE_SECONDS = 10 # Example: 10-second half-life

# Use time-weighted EMA (using '_runtime' and SMOOTHING_HALFLIFE_SECONDS)?
USE_TIME_WEIGHTED_EMA = True

PLOT_GRID = True
PLOT_SAVE_DIR = "plots"

def smooth_ema(series, factor):
    """Applies Exponential Moving Average smoothing based on data point order using alpha."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if factor <= 0 or factor >= 1:
        print("      Skipping standard smoothing (factor is <= 0 or >= 1).")
        return series
    alpha = 1.0 - factor
    print(f"      Applying Standard EWM using alpha={alpha:.3f} (factor={factor:.3f})...")
    return series.ewm(alpha=alpha, adjust=True, ignore_na=True).mean()

def smooth_ema_time_weighted(series, times):
    """
    Applies Time-Weighted Exponential Moving Average smoothing using provided timestamps (_runtime)
    and the SMOOTHING_HALFLIFE_SECONDS configuration.
    Handles potential NaNs and non-monotonic times internally.
    Converts numeric time (seconds) to datetime64[ns] for pandas ewm compatibility.
    Uses 'halflife' parameter compatible with time-based EWM.
    """
    global SMOOTHING_HALFLIFE_SECONDS # Access the global config

    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if not isinstance(times, pd.Series):
        times = pd.Series(times)

    if not SMOOTHING_HALFLIFE_SECONDS or SMOOTHING_HALFLIFE_SECONDS <= 0:
        print("      Skipping time-weighted smoothing (SMOOTHING_HALFLIFE_SECONDS not configured).")
        return series

    df = pd.DataFrame({'data': series, 'time': times}, index=series.index).dropna()

    if df.empty:
        print("      WARNING: No valid (data, time) pairs for time-weighted EMA after dropna.")
        return pd.Series(index=series.index, dtype=float)

    if not pd.api.types.is_numeric_dtype(df['time']):
        print("      WARNING: Time values are not numeric. Cannot convert to datetime64 for time-weighted EMA.")
        return series # Fallback to unsmoothed

    try:
        df['time_dt'] = pd.to_datetime(df['time'], unit='s', origin='unix')
    except Exception as e:
        print(f"      ERROR: Failed to convert time values to datetime64: {e}. Falling back.")
        return series

    if not df['time_dt'].is_monotonic_increasing:
        df = df.sort_values(by='time_dt')

    try:
        halflife_str = f"{SMOOTHING_HALFLIFE_SECONDS}s"
        print(f"      Applying Time-Weighted EWM using halflife='{halflife_str}'...")
        smoothed_data = df['data'].ewm(halflife=halflife_str, times=df['time_dt'], adjust=True, ignore_na=True).mean()
    except ValueError as ve:
        print(f"      ERROR: ValueError during EWM calculation with halflife='{halflife_str}': {ve}")
        if 'time_dt' in df:
             print(f"      Time data type: {df['time_dt'].dtype}")
        return series # Fallback
    except Exception as e:
        print(f"      ERROR: Unexpected error during EWM calculation: {e}")
        return series # Fallback


    return smoothed_data.reindex(series.index)


def _plot_single_run(
    run,
    metric_key,
    x_axis_key,
    smoothing_factor,
    time_weighted_smoothing,
    label_override=None
):
    """
    Fetches history and plots data for a single wandb.Run object.
    Applies time-weighted EMA (using _runtime and SMOOTHING_HALFLIFE_SECONDS)
    or standard EMA (using smoothing_factor) based on flags.
    """
    run_name_for_plot = label_override if label_override is not None else run.name
    print(f"    Processing run: '{run.name}' (ID: {run.id}) -> Plot Label: '{run_name_for_plot}'")

    # Determine keys to fetch
    keys_to_fetch = list(set([metric_key, x_axis_key]))
    # Fetch _runtime only if TWE is active and configured
    if time_weighted_smoothing and SMOOTHING_HALFLIFE_SECONDS and SMOOTHING_HALFLIFE_SECONDS > 0:
        keys_to_fetch.append('_runtime')
        keys_to_fetch = list(set(keys_to_fetch))

    try:
        # Fetch history
        print(f"      Fetching keys: {keys_to_fetch}")
        history = run.history(keys=keys_to_fetch, pandas=True)
        print(f"      History fetched. Shape: {history.shape}")

        # --- Data Cleaning ---
        cols_present = [col for col in keys_to_fetch if col in history.columns]
        if not cols_present:
             print(f"      WARNING: None of the requested keys {keys_to_fetch} found. Skipping.")
             return False
        history = history[cols_present].copy()

        essential_cols = [metric_key, x_axis_key]
        if not all(col in history.columns for col in essential_cols):
            print(f"      WARNING: Essential keys '{metric_key}' or '{x_axis_key}' not found. Skipping.")
            return False
        history = history.dropna(subset=essential_cols)

        if history.empty:
            print(f"      WARNING: No valid points after dropping NaNs in essentials. Skipping.")
            return False

        history[metric_key] = pd.to_numeric(history[metric_key], errors='coerce')
        history[x_axis_key] = pd.to_numeric(history[x_axis_key], errors='coerce')
        history = history.dropna(subset=essential_cols)

        runtime_available_and_valid = False
        if time_weighted_smoothing and SMOOTHING_HALFLIFE_SECONDS and SMOOTHING_HALFLIFE_SECONDS > 0:
            if '_runtime' in history.columns:
                history['_runtime'] = pd.to_numeric(history['_runtime'], errors='coerce')
                history = history.dropna(subset=['_runtime'])
                if not history.empty:
                    runtime_available_and_valid = True
                else:
                    print(f"      INFO: '_runtime' data became empty after cleaning. Cannot use Time-Weighted EMA.")
            else:
                print(f"      INFO: '_runtime' key not found. Cannot use Time-Weighted EMA.")

        if history.empty:
            print(f"      WARNING: No valid points after cleaning numerics (incl. _runtime if TWE). Skipping.")
            return False

        history = history.sort_values(by=x_axis_key)

        # --- Prepare Data for Plotting ---
        x_values = history[x_axis_key]
        y_values = history[metric_key]

        # --- Apply Smoothing ---
        y_values_smoothed = y_values # Default
        smoothing_applied = False
        if time_weighted_smoothing:
            if runtime_available_and_valid:
                # Call TWE function (which uses global SMOOTHING_HALFLIFE_SECONDS)
                runtime_values = history['_runtime']
                y_values_smoothed = smooth_ema_time_weighted(y_values, runtime_values)
                smoothing_applied = True # Indicate smoothing was attempted
            else:
                # Fallback to standard EMA if runtime isn't available but TWE was requested?
                # Or just skip smoothing? Let's skip for now to avoid confusion.
                print(f"      Skipping smoothing: TWE requested but _runtime not available/valid.")
        elif smoothing_factor > 0: # Apply standard EMA only if TWE is False
            y_values_smoothed = smooth_ema(y_values, smoothing_factor)
            smoothing_applied = True

        plot_df = pd.DataFrame({'x': x_values, 'y_smooth': y_values_smoothed}).dropna()

        if plot_df.empty:
             # Check if smoothing was applied; if so, maybe all points became NaN
             if smoothing_applied:
                 print(f"      WARNING: No valid points remain after smoothing. Skipping plot line.")
             else:
                  print(f"      WARNING: No valid points to plot even before smoothing. Skipping plot line.")
             return False

        plt.plot(plot_df['x'], plot_df['y_smooth'], label=run_name_for_plot, linewidth=2)
        print(f"      Plotted data points: {len(plot_df)}")
        return True

    except KeyError as e:
        print(f"      ERROR: Key '{e}' not found in history for run '{run.id}'.")
        return False
    except Exception as e:
        print(f"      ERROR: Unexpected error processing history for run '{run.id}'. Details: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- Plotting Orchestration Functions (_plot_runs_by_path, _plot_runs_by_name) ---
# Pass smoothing_factor down, TWE function uses global halflife setting

def _plot_runs_by_path(api, run_paths, custom_run_names, metric_key, x_axis_key, smoothing_factor, time_weighted_smoothing):
    """Plots runs specified by paths, using custom names if provided."""
    print(f"  Plotting {len(run_paths)} runs identified by path...")
    plot_count = 0
    for i, run_path in enumerate(run_paths):
        print(f"\n  Fetching run path: {run_path}")
        try:
            run = api.run(run_path)
            label = custom_run_names[i] if custom_run_names and i < len(custom_run_names) else None
            # Pass smoothing_factor for standard EMA fallback/case
            if _plot_single_run(run, metric_key, x_axis_key, smoothing_factor, time_weighted_smoothing, label_override=label):
                plot_count += 1
        except wandb.errors.CommError as e:
            print(f"    ERROR: Could not fetch run '{run_path}'. Check path/permissions. Details: {e}")
        except Exception as e:
            print(f"    ERROR: Unexpected error for run path '{run_path}'. Details: {e}")
    return plot_count > 0 # Return True if at least one run was plotted

def _plot_runs_by_name(api, entity, project, run_names, custom_run_names, metric_key, x_axis_key, smoothing_factor, time_weighted_smoothing):
    """Plots runs specified by names, handling duplicates and custom names appropriately."""
    full_project_path = f"{entity}/{project}"
    print(f"  Searching for runs by name in project: {full_project_path}")
    print(f"  Run names to search for: {run_names}")
    plot_count = 0

    for i, run_name_to_find in enumerate(run_names):
        print(f"\n  Searching for run name: '{run_name_to_find}'...")
        try:
            # Filter runs by display name
            runs_found = api.runs(full_project_path, filters={"display_name": run_name_to_find})

            if not runs_found:
                print(f"    WARNING: No run found with name '{run_name_to_find}'. Skipping.")
                continue

            print(f"    Found {len(runs_found)} run(s) with name '{run_name_to_find}'.")

            label_base = custom_run_names[i] if custom_run_names and i < len(custom_run_names) else None

            if len(runs_found) == 1:
                run = runs_found[0]
                label = label_base if label_base is not None else run.name
                # Pass smoothing_factor for standard EMA fallback/case
                if _plot_single_run(run, metric_key, x_axis_key, smoothing_factor, time_weighted_smoothing, label_override=label):
                    plot_count += 1
            else:
                print("    Handling duplicate names: Using '<name> (id)' format for labels.")
                for run in runs_found:
                     label = f"{label_base or run.name} ({run.id})"
                     # Pass smoothing_factor for standard EMA fallback/case
                     if _plot_single_run(run, metric_key, x_axis_key, smoothing_factor, time_weighted_smoothing, label_override=label):
                         plot_count += 1

        except wandb.errors.CommError as e:
            print(f"    ERROR: Could not query runs for project '{full_project_path}'. Details: {e}")
        except Exception as e:
            print(f"    ERROR: Unexpected error searching for run name '{run_name_to_find}'. Details: {e}")
    return plot_count > 0 # Return True if at least one run was plotted


# --- Main Plotting Function ---

def plot_wandb_runs(
    metric_key,
    x_axis_key='_step', # Default x-axis
    run_paths=None,
    entity=None,
    project=None,
    run_names=None,
    custom_run_names=None,
    smoothing_factor=0.0, # For standard EMA
    time_weighted_smoothing=False, # Use TWE based on _runtime?
    # SMOOTHING_HALFLIFE_SECONDS is used globally if time_weighted_smoothing is True
    title="W&B Run Comparison",
    xlabel=None, # Auto-generate if None
    ylabel=None, # Auto-generate if None
    grid=True,
    save_dir="plots"
    ):
    """
    Plots metrics from W&B runs, identified by path or name, with optional custom labels,
    configurable x-axis. Applies time-weighted smoothing (using _runtime and
    SMOOTHING_HALFLIFE_SECONDS) or standard smoothing (using smoothing_factor).
    """
    global SMOOTHING_HALFLIFE_SECONDS # Make global visible for label generation

    print(f"\n--- Generating Plot: '{title}' (Metric: {metric_key}, X-axis: {x_axis_key}) ---")

    # Determine run identification method
    use_path_method = bool(run_paths)
    use_name_method = bool(entity and project and run_names)

    # --- Input Validation ---
    # (Input validation remains the same)
    if use_path_method and use_name_method:
        print("ERROR: Configure EITHER run_paths OR entity/project/run_names, not both.")
        return
    if not use_path_method and not use_name_method:
        print("ERROR: No runs specified. Configure EITHER run_paths OR entity/project/run_names.")
        return
    if use_path_method and not run_paths:
         print("ERROR: `run_paths` is empty.")
         return
    if use_name_method and not run_names:
         print("ERROR: `run_names` is empty.")
         return
    if custom_run_names:
        expected_len = len(run_paths) if use_path_method else len(run_names)
        if len(custom_run_names) != expected_len:
             print(f"ERROR: CUSTOM_RUN_NAMES length ({len(custom_run_names)}) must match runs ({expected_len}).")
             return

    # --- Setup Plot Labels ---
    # Auto-generate Y-label based on smoothing method active
    if ylabel is None:
        smooth_desc = ""
        # Check if Time-Weighted EMA is active and configured
        if time_weighted_smoothing and SMOOTHING_HALFLIFE_SECONDS and SMOOTHING_HALFLIFE_SECONDS > 0:
            smooth_desc = f" (Smoothed, {SMOOTHING_HALFLIFE_SECONDS}s Halflife)"
        # Check if Standard EMA is active (and TWE is not)
        elif not time_weighted_smoothing and smoothing_factor > 0 and smoothing_factor < 1:
             smooth_desc = f" (Smoothed, Factor {smoothing_factor:.1f})"
        # Otherwise, no smoothing description
        ylabel = f"{metric_key.replace('_', ' ').title()}{smooth_desc}"


    # Auto-generate X-label if not provided
    if xlabel is None:
        if x_axis_key == '_step':
            xlabel = "Steps"
        elif x_axis_key == '_runtime':
            xlabel = "Runtime (seconds)"
        elif x_axis_key == '_timestamp':
            xlabel = "Timestamp (Unix)"
        else:
            xlabel = x_axis_key.replace('_', ' ').title() # Default formatting

    # --- Initialize W&B API ---
    # API initialization moved to main block for runtime calculation access
    # print("Initializing W&B API...")
    # try:
    #     api = wandb.Api(timeout=30)
    #     print("API initialized.")
    # except Exception as e:
    #     print(f"FATAL: Failed to initialize W&B API: {e}")
    #     return

    # --- Create Plot Figure ---
    plt.figure(figsize=(12, 7))
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # --- Fetch and Plot Data ---
    # API object is passed from the main block
    api = wandb.Api(timeout=30) # Initialize API here for plotting function scope
    plot_successful = False
    if use_path_method:
        plot_successful = _plot_runs_by_path(api, run_paths, custom_run_names, metric_key, x_axis_key, smoothing_factor, time_weighted_smoothing)
    elif use_name_method:
        plot_successful = _plot_runs_by_name(api, entity, project, run_names, custom_run_names, metric_key, x_axis_key, smoothing_factor, time_weighted_smoothing)

    # --- Finalize Plot ---
    print("\n  Finalizing plot...")
    if plot_successful and len(plt.gca().lines) > 0:
        num_lines = len(plt.gca().lines)
        legend_fontsize = 10 if num_lines < 10 else 8
        plt.legend(fontsize=legend_fontsize, loc='best')

        if grid:
            plt.grid(True, linestyle='--', alpha=0.6)

        os.makedirs(save_dir, exist_ok=True)
        filename_metric = metric_key.replace('/', '_')
        filename_xaxis = x_axis_key.lstrip('_')
        save_path = os.path.join(save_dir, f"wandb_compare_{filename_metric}_vs_{filename_xaxis}.png")

        plt.tight_layout()
        print(f"  Saving plot to '{save_path}'.")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        print("--- Plot Generation Complete ---")

    else:
        print("  WARNING: No data was successfully plotted for this configuration.")
        plt.close()
        print("--- Plot Generation Skipped ---")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration Validation ---
    method1_configured = bool(RUN_PATHS)
    method2_configured = bool(ENTITY and PROJECT and RUN_NAMES) and ENTITY != "your_entity" and PROJECT != "your_project"
    valid_config = True
    if not method1_configured and not method2_configured:
        print("\n" + "="*30 + " CONFIGURATION ERROR " + "="*30); print("Please configure EITHER RUN_PATHS OR (ENTITY, PROJECT, and RUN_NAMES)."); print("="*80 + "\n"); valid_config = False
    elif method1_configured and method2_configured:
        print("\n" + "="*30 + " CONFIGURATION WARNING " + "="*30); print("Both RUN_PATHS and RUN_NAMES are configured. Using RUN_PATHS."); print("Clear one of them to avoid ambiguity."); print("="*80 + "\n"); method2_configured = False
    elif CUSTOM_RUN_NAMES:
        expected_len = len(RUN_PATHS) if method1_configured else len(RUN_NAMES)
        if len(CUSTOM_RUN_NAMES) != expected_len:
             print("\n" + "="*30 + " CONFIGURATION ERROR " + "="*30); print(f"Length mismatch: CUSTOM_RUN_NAMES ({len(CUSTOM_RUN_NAMES)}) vs specified runs ({expected_len})."); print("="*80 + "\n"); valid_config = False

    # --- Generate Plots ---
    if valid_config:
        print("Starting plot generation process...")
        custom_names_to_pass = CUSTOM_RUN_NAMES

        for metric in METRICS_TO_PLOT:
            for x_key in X_AXIS_KEYS_TO_PLOT:
                # Determine if time-weighted smoothing flag is set
                use_tws = USE_TIME_WEIGHTED_EMA

                metric_title = metric.replace('_', ' ').title()
                x_axis_title = "Runtime" if x_key == '_runtime' else x_key.replace('_', ' ').title()
                plot_title = f"{metric_title} vs {x_axis_title}"

                plot_wandb_runs(
                    metric_key=metric,
                    x_axis_key=x_key,
                    run_paths=RUN_PATHS if method1_configured else None,
                    entity=ENTITY if method2_configured else None,
                    project=PROJECT if method2_configured else None,
                    run_names=RUN_NAMES if method2_configured else None,
                    custom_run_names=custom_names_to_pass,
                    smoothing_factor=SMOOTHING_FACTOR, # Pass factor for standard EMA
                    time_weighted_smoothing=use_tws, # Pass TWE flag
                    # TWE function uses global SMOOTHING_HALFLIFE_SECONDS
                    title=plot_title,
                    grid=PLOT_GRID,
                    save_dir=PLOT_SAVE_DIR
                )
        print("\nAll requested plots generated.")

    # --- Calculate Runtime Multiplication (if applicable) ---
    # This block runs after plotting
    if valid_config and len(RUN_NAMES) == 2 and '_runtime' in X_AXIS_KEYS_TO_PLOT:
        print("\n" + "="*30 + " RUNTIME MULTIPLICATION CALCULATION " + "="*30)
        run_name_1 = RUN_NAMES[0]
        run_name_2 = RUN_NAMES[1]
        runtime_1 = None
        runtime_2 = None

        display_name_1 = CUSTOM_RUN_NAMES[0] if CUSTOM_RUN_NAMES and len(CUSTOM_RUN_NAMES) == 2 else run_name_1
        display_name_2 = CUSTOM_RUN_NAMES[1] if CUSTOM_RUN_NAMES and len(CUSTOM_RUN_NAMES) == 2 else run_name_2


        print(f"Calculating runtime multiplication for '{display_name_2}' vs '{display_name_1}'...")

        try:
            # Initialize API specifically for this calculation if not already done in a shared scope
            api = wandb.Api(timeout=60) # Increased timeout for API calls
            print("W&B API initialized for runtime calculation.")

            # Fetch history for the first run
            print(f"Fetching runtime for run '{run_name_1}'...")
            runs_found_1 = api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": run_name_1})
            if runs_found_1:
                run1 = runs_found_1[0] # Assuming the first found run is the desired one
                history1 = run1.history(keys=['_runtime'], pandas=True)
                if '_runtime' in history1.columns and not history1.empty:
                    # Get the last recorded runtime value
                    runtime_1 = history1['_runtime'].max() # Using max() to get the final runtime
                    print(f"Found final runtime for '{display_name_1}' ('{run_name_1}'): {runtime_1:.2f} seconds")
                else:
                    print(f"WARNING: '_runtime' data not found or is empty for run '{display_name_1}' ('{run_name_1}').")
            else:
                print(f"WARNING: Run with name '{run_name_1}' not found in project '{ENTITY}/{PROJECT}'.")

            # Fetch history for the second run
            print(f"Fetching runtime for run '{run_name_2}'...")
            runs_found_2 = api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": run_name_2})
            if runs_found_2:
                run2 = runs_found_2[0] # Assuming the first found run is the desired one
                history2 = run2.history(keys=['_runtime'], pandas=True)
                if '_runtime' in history2.columns and not history2.empty:
                    # Get the last recorded runtime value
                    runtime_2 = history2['_runtime'].max() # Using max() to get the final runtime
                    print(f"Found final runtime for '{display_name_2}' ('{run_name_2}'): {runtime_2:.2f} seconds")
                else:
                    print(f"WARNING: '_runtime' data not found or is empty for run '{display_name_2}' ('{run_name_2}').")
            else:
                print(f"WARNING: Run with name '{run_name_2}' not found in project '{ENTITY}/{PROJECT}'.")


            # Calculate and print the multiplication factor
            if runtime_1 is not None and runtime_2 is not None and runtime_1 > 0:
                multiplication_factor = runtime_2 / runtime_1
                print(f"\nResult:")
                print(f"The runtime of '{display_name_2}' ({runtime_2:.2f} seconds) is {multiplication_factor:.2f} times the runtime of '{display_name_1}' ({runtime_1:.2f} seconds).")
            elif runtime_1 is not None and runtime_2 is not None and runtime_1 == 0:
                 print(f"\nCannot calculate multiplication factor: Runtime of '{display_name_1}' is zero.")
            else:
                print("\nCannot calculate multiplication factor: Runtime data not available for both runs.")

        except Exception as e:
            print(f"ERROR during runtime multiplication calculation: {e}")
            import traceback
            traceback.print_exc()

        print("="*80 + "\n")

    elif valid_config and len(RUN_NAMES) != 2 and '_runtime' in X_AXIS_KEYS_TO_PLOT:
         print("\n" + "="*30 + " RUNTIME MULTIPLICATION SKIPPED " + "="*30)
         print("Runtime multiplication calculation is enabled but requires exactly two run names in RUN_NAMES.")
         print("="*80 + "\n")
    elif valid_config and '_runtime' not in X_AXIS_KEYS_TO_PLOT:
         print("\n" + "="*30 + " RUNTIME MULTIPLICATION SKIPPED " + "="*30)
         print("Runtime multiplication calculation is enabled but '_runtime' is not included in X_AXIS_KEYS_TO_PLOT.")
         print("="*80 + "\n")
    else:
        print("Plot generation and runtime calculation skipped due to configuration errors.")

