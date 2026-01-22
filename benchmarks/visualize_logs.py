# visualize_logs.py

# visualize_logs.py

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
PROJECT_DIR = os.path.expanduser("~/tbp/results/monty/projects")
EXPERIMENT_NAME = "surf_agent_1lm_2obj"
EVAL_DIR = f"{PROJECT_DIR}/{EXPERIMENT_NAME}/eval"

# Check if eval directory exists
if not os.path.exists(EVAL_DIR):
    raise FileNotFoundError(f"Evaluation directory not found: {EVAL_DIR}\nDid you run the experiment?")

# Get all subdirectories (should be numbered: 0, 1, 2, ...)
subdirs = [d for d in os.listdir(EVAL_DIR) if os.path.isdir(f"{EVAL_DIR}/{d}") and d.isdigit()]

if not subdirs:
    raise FileNotFoundError(f"No numbered experiment folders found in {EVAL_DIR}\nDid the evaluation save results?")

# Sort by number and pick the latest
latest_run = max(subdirs, key=int)
LOGS_PATH = f"{EVAL_DIR}/{latest_run}/logs.pkl"

# Check if logs.pkl exists
if not os.path.exists(LOGS_PATH):
    raise FileNotFoundError(f"logs.pkl not found in {EVAL_DIR}/{latest_run}\nCheck if the run completed fully.")

print(f"✅ Loading logs from run {latest_run}: {LOGS_PATH}")

# --- Load Logs ---
with open(LOGS_PATH, "rb") as f:
    logs = pickle.load(f)

print("Available sensor modules:", list(logs.keys()))
print("Number of steps:", len(logs["sensor_module_0"]))

# --- Extract Data ---
steps = range(len(logs["sensor_module_0"]))
locations = []
hsvs = []
curvatures_log = []

for obs in logs["sensor_module_0"]:
    loc = obs.get("location")
    hsv = obs.get("hsv")
    curv = obs.get("principal_curvatures_log")

    if loc is not None:
        locations.append(loc)
    if hsv is not None:
        hsvs.append(hsv)
    if curv is not None:
        curvatures_log.append(curv)

locations = np.array(locations)
hsvs = np.array(hsvs)
curvatures_log = np.array(curvatures_log)

# --- Plotting ---
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# 1. Location over time
axes[0].plot(steps, locations[:, 0], label="X", alpha=0.8)
axes[0].plot(steps, locations[:, 1], label="Y", alpha=0.8)
axes[0].plot(steps, locations[:, 2], label="Z", alpha=0.8)
axes[0].set_ylabel("Location (m)")
axes[0].set_title("Sensor Location Over Time")
axes[0].legend()
axes[0].grid(True)

# 2. HSV over time
axes[1].plot(steps, hsvs[:, 0], label="Hue", color='r', alpha=0.7)
axes[1].plot(steps, hsvs[:, 1], label="Saturation", color='g', alpha=0.7)
axes[1].plot(steps, hsvs[:, 2], label="Value", color='b', alpha=0.7)
axes[1].set_ylabel("HSV")
axes[1].set_title("Color (HSV) Over Time")
axes[1].legend()
axes[1].grid(True)

# 3. Principal Curvatures (log)
axes[2].plot(steps, curvatures_log[:, 0], label="k1_log", color='orange')
axes[2].plot(steps, curvatures_log[:, 1], label="k2_log", color='purple')
axes[2].set_ylabel("Log Curvature")
axes[2].set_title("Surface Curvature Over Time")
axes[2].set_xlabel("Step")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()