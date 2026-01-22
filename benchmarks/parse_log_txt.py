# parse_log_txt.py

import os
import re
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = os.path.expanduser("~/tbp/results/monty/projects/surf_agent_1lm_2obj/eval/log.txt")

if not os.path.exists(LOG_PATH):
    raise FileNotFoundError(f"Log file not found: {LOG_PATH}")

print(f"✅ Loading log from: {LOG_PATH}")

# Lists to store data
locations = []
hsvs = []
curvatures_log = []

# Regex patterns
loc_pattern = r"location': array\(\[([-\d\.e]+),\s*([-\d\.e]+),\s*([-\d\.e]+)"
hsv_pattern = r"hsv': array\(\[([-\d\.e]+),\s*([-\d\.e]+),\s*([-\d\.e]+)"
curv_pattern = r"principal_curvatures_log': array\(\[([-\d\.e]+),\s*([-\d\.e]+)"

with open(LOG_PATH, "r") as f:
    for line in f:
        # Extract location
        loc_match = re.search(loc_pattern, line)
        if loc_match:
            x, y, z = map(float, loc_match.groups())
            locations.append([x, y, z])

        # Extract HSV
        hsv_match = re.search(hsv_pattern, line)
        if hsv_match:
            h, s, v = map(float, hsv_match.groups())
            hsvs.append([h, s, v])

        # Extract curvature
        curv_match = re.search(curv_pattern, line)
        if curv_match:
            k1, k2 = map(float, curv_match.groups())
            curvatures_log.append([k1, k2])

# Convert to numpy arrays (safe even if empty)
locations = np.array(locations)
hsvs = np.array(hsvs)
curvatures_log = np.array(curvatures_log)

print(f"Found {len(locations)} location entries")
print(f"Found {len(hsvs)} HSV entries")
print(f"Found {len(curvatures_log)} curvature entries")

# Only plot if data exists
if len(locations) == 0:
    print("⚠️ No sensor data found in log.txt — cannot plot.")
    print("💡 Tip: Enable detailed logging to save full observations.")
else:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    steps = range(len(locations))

    # 1. Location
    axes[0].plot(steps, locations[:, 0], label="X", alpha=0.8)
    axes[0].plot(steps, locations[:, 1], label="Y", alpha=0.8)
    axes[0].plot(steps, locations[:, 2], label="Z", alpha=0.8)
    axes[0].set_ylabel("Location")
    axes[0].set_title("Sensor Location Over Time")
    axes[0].legend()
    axes[0].grid(True)

    # 2. HSV
    if len(hsvs) > 0:
        axes[1].plot(steps[:len(hsvs)], hsvs[:, 0], label="Hue", color='r', alpha=0.7)
        axes[1].plot(steps[:len(hsvs)], hsvs[:, 1], label="Saturation", color='g', alpha=0.7)
        axes[1].plot(steps[:len(hsvs)], hsvs[:, 2], label="Value", color='b', alpha=0.7)
    else:
        axes[1].text(0.5, 0.5, "No HSV data", ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_ylabel("HSV")
    axes[1].set_title("Color (HSV) Over Time")
    axes[1].grid(True)

    # 3. Curvature
    if len(curvatures_log) > 0:
        axes[2].plot(steps[:len(curvatures_log)], curvatures_log[:, 0], label="k1_log", color='orange')
        axes[2].plot(steps[:len(curvatures_log)], curvatures_log[:, 1], label="k2_log", color='purple')
    else:
        axes[2].text(0.5, 0.5, "No curvature data", ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_ylabel("Log Curvature")
    axes[2].set_title("Surface Curvature Over Time")
    axes[2].set_xlabel("Step")
    axes[2].legend() if len(curvatures_log) > 0 else None
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()