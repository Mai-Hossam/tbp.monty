# analyze_results.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

CSV_PATH = os.path.expanduser("~/tbp/results/monty/projects/surf_agent_1lm_2obj/eval/eval_stats.csv")

# Load data
df = pd.read_csv(CSV_PATH)
print("✅ Loaded data with columns:")
print(df.columns.tolist())
print("\n📋 First few rows:")
print(df[["primary_target_object", "primary_target_rotation_euler", "primary_performance", "num_steps"]])

# Add clean rotation column
def parse_rotation(rot_str):
    # Remove brackets and split
    rot_str = rot_str.replace('[', '').replace(']', '')
    return np.array([float(x) for x in rot_str.split()])

df['rotation'] = df['primary_target_rotation_euler'].apply(parse_rotation)
df['rotation_x'] = df['rotation'].apply(lambda x: x[0])
df['rotation_y'] = df['rotation'].apply(lambda x: x[1])
df['rotation_z'] = df['rotation'].apply(lambda x: x[2])

# Success: 1 if matched, 0 otherwise
df['success'] = (df['primary_performance'] == 'match').astype(int)
success_rate = df['success'].mean()
total_episodes = len(df)

print(f"\n📊 Summary:")
print(f"Total episodes: {total_episodes}")
print(f"Successes: {df['success'].sum()}")
print(f"Success rate: {success_rate:.1%}")

# Plot 1: Success by Object
plt.figure(figsize=(12, 8))

object_success = df.groupby('primary_target_object')['success'].agg(['sum', 'count'])
object_success['rate'] = object_success['sum'] / object_success['count']

plt.subplot(2, 2, 1)
object_success['rate'].plot(kind='bar', color=['skyblue', 'lightgreen'])
plt.title('Recognition Success Rate by Object')
plt.ylabel('Success Rate')
plt.xlabel('Object')
plt.xticks(rotation=0)
plt.ylim(0, 1)
for i, v in enumerate(object_success['rate']):
    plt.text(i, v + 0.02, f"{v:.0%}", ha='center')

# Plot 2: Steps vs Rotation X
plt.subplot(2, 2, 2)
for obj in df['primary_target_object'].unique():
    subset = df[df['primary_target_object'] == obj]
    plt.scatter(subset['rotation_x'], subset['num_steps'], label=obj, alpha=0.7)
plt.xlabel('Rotation X (deg)')
plt.ylabel('Number of Steps')
plt.title('Steps vs Rotation X')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Runtime by Object
plt.subplot(2, 2, 3)
df.boxplot(column='time', by='primary_target_object', ax=plt.gca())
plt.title('Runtime by Object')
plt.suptitle('')  # Remove default title
plt.ylabel('Time (s)')
plt.xlabel('Object')

# Plot 4: Detected Locations (if any)
plt.subplot(2, 2, 4)
locations = []
for loc_str in df['detected_location']:
    try:
        loc = eval(loc_str)  # Convert "[x, y, z]" → [x, y, z]
        if isinstance(loc, list) and all(isinstance(x, (int, float)) for x in loc):
            locations.append(loc)
    except:
        pass

if locations:
    locations = np.array(locations)
    plt.scatter(locations[:, 0], locations[:, 1], c='orange', label='Detected')
    plt.scatter([0], [1.5], c='red', marker='x', s=100, label='True (0,1.5)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Detected vs True Location (Top-Down)')
    plt.legend()
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No location predictions', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Detected Locations')
    plt.axis('off')

plt.tight_layout()
plt.show()