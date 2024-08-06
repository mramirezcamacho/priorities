import matplotlib.pyplot as plt
import numpy as np

# Data
months = ['January', 'February', 'March', 'April']
categories = ['Category A', 'Category B', 'Category C', 'Category D']
data = np.array([
    [5, 7, 8, 0.1],  # January
    [6, 6, 10, 0.1],  # February
    [7, 8, 5, 0.1],  # March
    [8, 6, 6, 0.1],  # April
])

# Calculate the percentage contribution
totals = np.sum(data, axis=1)
percentages = data / totals[:, None] * 100

# Plotting the stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Define the bottom of each bar segment
bottom = np.zeros(len(months))

# Plot each category
for i in range(len(categories)):
    ax.bar(months, data[:, i], label=categories[i], bottom=bottom)

    # Add percentage labels
    for j in range(len(months)):
        percentage = f'{percentages[j, i]:.1f}%'
        ax.text(j, bottom[j] + data[j, i] / 2, percentage,
                ha='center', va='center', color='black', fontsize=10)

    bottom += data[:, i]

# Adding labels and title
ax.set_ylabel('Values')
ax.set_title('Stacked Bar Chart of Categories Over 4 Months with Percentages')
ax.legend(title='Categories')

# Show the plot
plt.tight_layout()
plt.show()
