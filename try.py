import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data


def exampleBarChartDivided():
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
    ax.set_title(
        'Stacked Bar Chart of Categories Over 4 Months with Percentages')
    ax.legend(title='Categories')

    # Show the plot
    plt.tight_layout()
    plt.show()


def example2():
    np.random.seed(10)
    x = np.arange(0, 10)
    data = pd.DataFrame({'x': x})
    for i in range(1, 8):  # Creating 7 series
        data[f'Series {i}'] = np.random.rand(10)

    # Set the style
    sns.set(style="whitegrid")

    # Create the figure and axes
    plt.figure(figsize=(12, 8))

    # Variables to control legend separation
    offset_step = 0.05
    last_y_pos = None

    # Plot each line
    for column in data.columns[1:]:
        plt.plot(data['x'], data[column], label=column)

        # Determine the vertical position of the label to avoid overlaps
        y_pos = data[column].iloc[-1]

        if last_y_pos is not None and abs(y_pos - last_y_pos) < offset_step:
            y_pos = last_y_pos + offset_step

        plt.text(data['x'].iloc[-1] + 0.2, y_pos, column,
                 horizontalalignment='left', size='medium', color=plt.gca().lines[-1].get_color(),
                 bbox=dict(facecolor='white', edgecolor='black', pad=2.0, boxstyle='round,pad=0.5'))

        last_y_pos = y_pos

    # Configure labels and title
    plt.xlabel('X axis')
    plt.ylabel('Y axis')

    # Remove the traditional legend
    plt.legend().set_visible(False)

    # Display the plot
    plt.show()


example2()
