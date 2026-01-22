# visualize_eval_stats_enhanced.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

CSV_PATH = os.path.expanduser("~/tbp/results/monty/projects/surf_agent_1lm_2obj/eval/eval_stats.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"File not found: {CSV_PATH}")

print(f"✅ Loaded: {CSV_PATH}")

# Load Data
df = pd.read_csv(CSV_PATH)

print("\n📋 Data Overview:")
print(f"Total episodes: {len(df)}")
print(f"Columns available: {len(df.columns)}")

# Find key columns
performance_cols = [col for col in df.columns if 'performance' in col.lower()]
object_cols = [col for col in df.columns if 'object' in col.lower() and 'target' in col.lower()]
steps_cols = [col for col in df.columns if 'step' in col.lower() and 'performance' not in col.lower()]

print(f"\n🎯 Performance columns: {performance_cols}")
print(f"📦 Object columns: {object_cols}")
print(f"👣 Steps columns: {steps_cols}")

# Use available columns
perf_col = performance_cols[0] if performance_cols else None
object_col = object_cols[0] if object_cols else None
steps_col = steps_cols[0] if steps_cols else None

if not all([perf_col, object_col, steps_col]):
    print("❌ Missing required columns!")
    print("Available columns:", df.columns.tolist())
    exit(1)

print(f"\n📈 Using:")
print(f"Performance: {perf_col}")
print(f"Object: {object_col}")
print(f"Steps: {steps_col}")

# Analyze performance values
print(f"\n📊 Performance value counts:")
print(df[perf_col].value_counts())

# Define success criteria
success_keywords = ['correct', 'match', 'success', 'true', 'yes']
df['success'] = df[perf_col].astype(str).str.lower().str.contains('|'.join(success_keywords))

print(f"\n✅ Success rate: {df['success'].mean():.2%}")

# Create visualizations
plt.figure(figsize=(15, 10))

# 1. Success Rate by Object
plt.subplot(2, 2, 1)
success_by_object = df.groupby(object_col)['success'].mean()
colors = ['#4CAF50' if x > 0.5 else '#2196F3' for x in success_by_object]
success_by_object.plot(kind='bar', color=colors, alpha=0.8)
plt.title('Success Rate by Object', fontsize=16, fontweight='bold')
plt.ylabel('Success Rate')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

for i, v in enumerate(success_by_object):
    plt.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontweight='bold')

# 2. Steps Distribution
plt.subplot(2, 2, 2)
plt.hist(df[steps_col], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Steps Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Number of Steps')
plt.ylabel('Frequency')
plt.axvline(df[steps_col].mean(), color='red', linestyle='--', 
           label=f'Mean: {df[steps_col].mean():.1f}')
plt.legend()
plt.grid(alpha=0.3)

# 3. Performance Outcomes
plt.subplot(2, 2, 3)
performance_counts = df[perf_col].value_counts()
performance_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#4CAF50', '#F44336', '#FFC107'])
plt.title('Performance Outcomes', fontsize=16, fontweight='bold')
plt.ylabel('')

# 4. Steps vs Success
plt.subplot(2, 2, 4)
successful_steps = df[df['success'] == True][steps_col]
failed_steps = df[df['success'] == False][steps_col]

plt.hist([successful_steps, failed_steps], bins=15, alpha=0.7, 
         label=['Successful', 'Failed'], color=['green', 'red'])
plt.title('Steps by Outcome', fontsize=16, fontweight='bold')
plt.xlabel('Steps')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Detailed Analysis
print("\n" + "="*60)
print("📊 DETAILED ANALYSIS")
print("="*60)

# Summary statistics
summary = df.groupby(object_col).agg({
    'success': ['count', 'sum', 'mean'],
    steps_col: ['mean', 'std', 'min', 'max']
}).round(3)

print(summary)

# Performance insights
print(f"\n💡 INSIGHTS:")
print(f"Overall success rate: {df['success'].mean():.2%}")
print(f"Average steps: {df[steps_col].mean():.1f} ± {df[steps_col].std():.1f}")

if df['success'].mean() == 0:
    print("\n❌ CRITICAL: No successful recognitions!")
    print("Recommendations:")
    print("1. Loosen tolerances in GraphLM configuration")
    print("2. Increase max_total_steps in evaluation")
    print("3. Check if training was successful")
    print("4. Verify sensor noise parameters")
elif df['success'].mean() < 0.3:
    print("\n⚠️  POOR: Low success rate")
    print("Try loosening tolerances and increasing exploration steps")
elif df['success'].mean() < 0.7:
    print("\n⚠️  MODERATE: Room for improvement")
    print("Consider fine-tuning tolerances")
else:
    print("\n✅ EXCELLENT: Good performance!")

print(f"\n🎯 NEXT STEPS:")
print("1. Check training logs to ensure proper learning")
print("2. Adjust tolerances based on object complexity")
print("3. Consider adding more training rotations")
print("4. Verify sensor configurations")