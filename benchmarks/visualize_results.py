# visualize_results.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Load eval_stats.csv ---
csv_path = os.path.expanduser("~/tbp/results/monty/projects/surf_agent_1lm_2obj/eval/eval_stats.csv")

if not os.path.exists(csv_path):
    print(f"❌ File not found: {csv_path}")
    print("Run evaluation first!")
else:
    df = pd.read_csv(csv_path)
    print("✅ Loaded eval_stats.csv")
    print(df.head())

    # Try to find the object column
    obj_col = None
    for col in df.columns:
        if "target" in col and "object" in col:
            obj_col = col
            break
    if not obj_col:
        obj_col = "primary_target_object"  # fallback

    # Add success column
    df["success"] = df["primary_performance"] == "match"

    # --- Plot 1: Success Rate by Object ---
    success_rate = df.groupby(obj_col)["success"].mean()

    plt.figure(figsize=(10, 6))
    success_rate.plot(kind="bar", color=["#FF6B6B", "#4ECDC4"], alpha=0.8)
    plt.title("Recognition Success Rate by Object", fontsize=16)
    plt.ylabel("Success Rate", fontsize=12)
    plt.xlabel("Object", fontsize=12)
    plt.xticks(rotation=0)

    # Label bars
    for i, v in enumerate(success_rate):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Steps per Episode ---
    plt.figure(figsize=(10, 6))
    plt.hist(df["num_steps"], bins=10, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title("Number of Steps per Episode", fontsize=16)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Print Summary ---
    print("\n📋 Summary:")
    print(df.groupby(obj_col).agg(
        attempts=("success", "count"),
        successes=("success", "sum"),
        success_rate=("success", "mean"),
        avg_steps=("num_steps", "mean")
    ).round(3))