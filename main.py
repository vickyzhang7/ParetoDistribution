
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data for years and female percentages
years = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
percentages = np.array([8.829, 8.382, 8.756, 9.698, 10.494, 10.854, 10.629, 10.723, 11.061, 11.381, 11.207, 11.484, 11.751])

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(years.reshape(-1, 1), percentages)

# Predict the female percentage for 2023
predicted_percentage_2023 = model.predict(np.array([[2023]])).item()

# Known female percentage for 2023
actual_percentage_2023 = 12.765

# Calculate the error for 2023
error_2023 = (predicted_percentage_2023 - actual_percentage_2023) / actual_percentage_2023

# Predict the female percentage for 2024
predicted_percentage_2024 = model.predict(np.array([[2024]])).item()

# Extend the years for plotting
years_extended = np.arange(2010, 2031, 1)

# Predict the percentages for the extended years
percentages_extended = model.predict(years_extended.reshape(-1, 1))

# Print the predicted values for each year on the extended line
for year, percentage in zip(years_extended, percentages_extended):
    print(f"Year: {year}, Predicted Percentage: {percentage.item()}")

print("\nPredicted female percentage for 2023:", predicted_percentage_2023)
print("Known female percentage for 2023:", actual_percentage_2023)
print("Predicted female percentage for 2024:", predicted_percentage_2024)
print("Error for 2023:", error_2023)

# Plot the data and linear regression line
plt.figure(figsize=(10, 6))  # Adjust the figure size
plt.plot(years, percentages, 'black', label='Actual Percentage (2010-2022)')
plt.plot(years_extended, percentages_extended, '--', color='black', label='Linear Regression Fit (2010-2030)')

# Add solid points for the data
plt.scatter(years, percentages, color='black', s=20, marker='o', linewidths=2)

# Add solid points for each year on the green dashed line
for year, percentage in zip(years_extended, percentages_extended):
    plt.scatter(year, percentage, color='black', s=20, marker='o', linewidths=2)

# Title and labels
# plt.title('Female Percentage Over the Years (2010-2023)', fontsize=18)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Percentage (%)', fontsize=20)
plt.xticks(np.arange(2010, 2031, 1), rotation='vertical', fontsize=16)
plt.yticks(fontsize=16)

# Legend
plt.legend(loc='upper left', fontsize=14)


# Show the plot
plt.grid(True)
plt.show()


########################## continent ####################
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/Users/xiaohuidexiaojiaoqi/Desktop/Data9622.csv"
df = pd.read_csv(csv_path)

# List of columns representing continent data for each year
columns = [col for col in df.columns if col.startswith("C20")]

# Define custom colors for each continent
colors = {
    "Africa": "purple",
    "Asia": "darkred",
    "Australia": "darkorange",
    "Europe": "darkgreen",
    "North America": "darkblue",
    "South America": "saddlebrown",
}

percentage_data = []

# Iterate through each year and calculate the percentage for each continent
for year in columns:
    year_data = df.groupby(year).size()
    total_billionaires = year_data.sum()

    percentages = {
        "Year": year,
        "Asia": year_data.get("Asia", 0) / total_billionaires * 100,
        "Europe": year_data.get("Europe", 0) / total_billionaires * 100,
        "North America": year_data.get("North America", 0) / total_billionaires * 100,
        "South America": year_data.get("South America", 0) / total_billionaires * 100,
        "Australia": year_data.get("Australia", 0) / total_billionaires * 100,
        "Africa": year_data.get("Africa", 0) / total_billionaires * 100,
    }

    percentage_data.append(percentages)
    print(f"Year: {year}, Percentages: {percentages}")

percentage_df = pd.DataFrame(percentage_data)

# Plot the stacked multicolored bar plot with custom colors
ax = percentage_df.set_index("Year").plot(kind="bar", stacked=True,
                                          color=[colors[col] for col in percentage_df.columns[1:]], figsize=(10, 6))
# plt.title("Percentage of Billionaires by Continent (2001-2023)", fontsize=24)
plt.xlabel("Year", fontsize=20)
plt.ylabel("Percentage (%)", fontsize=20)


handles, labels = plt.gca().get_legend_handles_labels()
# Sort the order of the legend according to the initial of the label
sorted_order = sorted(range(len(labels)), key=lambda i: labels[i].split(" ")[0])
sorted_handles = [handles[i] for i in sorted_order]
sorted_labels = [labels[i] for i in sorted_order]
plt.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

# Adjust the font size of the horizontal label
plt.xticks(range(len(columns)), [int(col[1:]) for col in columns], rotation=90, fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.grid(True)

plt.show()

##################################### alpha ##############
from scipy.stats import pareto

columns = [col for col in df.columns if col.startswith("W")]

# Initializes the list of alpha values and years to store
alpha_values = []
years = []

# Iterate through the years
for year_column in columns:
    year_data = df[year_column].dropna()

    # Fit the data to the Pareto distribution and get the alpha parameters
    fit_params = pareto.fit(year_data)
    alpha = fit_params[0]

    print(f"Year {int(year_column[1:])}: Alpha = {alpha:.4f}")

    # Add the alpha value and year to the appropriate list
    alpha_values.append(alpha)
    years.append(int(year_column[1:]))

# Create a linear plot to visualize annual alpha values
plt.figure(figsize=(12, 6))

# Draw a line plot of alpha values
plt.plot(years, alpha_values, marker='o', linestyle='-', label='Alpha Value', color='black', linewidth=2)
# plt.title("Alpha value over the years", fontsize=24)
plt.xlabel("Year", fontsize=20)
plt.ylabel("Alpha", fontsize=20)
plt.xticks(years, rotation='vertical', fontsize=16)
plt.yticks(fontsize=16)

# Set the range of the Y-axis
plt.ylim(1.4, 1.9)

plt.grid(True)
plt.legend(loc='upper left', fontsize=14)
plt.show()

######################## Age #####################

columns = [col for col in df.columns if col.startswith("Age")]

# Initializes the list of storage medians
medians = []

years_list = list(range(2001, 2024))

# Iterate through the years
for year_column in columns:
    # Extract data for the current year and ignore "-"
    year_data = pd.to_numeric(df[year_column], errors='coerce').dropna()

    median = year_data.median()
    medians.append(median)

# Create a linear plot to visualize the median for each year
plt.figure(figsize=(12, 6))

# Plot a line graph of the median
plt.plot(years_list, medians, marker='o', linestyle='-', label='Median', color='black', linewidth=2)

plt.xlabel("Year", fontsize=20)
plt.ylabel("Age", fontsize=20)
# plt.title("Median Age Over the Years (2001-2023)", fontsize=24)

plt.xticks(years_list, rotation='vertical', fontsize=16)
plt.yticks([int(y) for y in plt.yticks()[0]],fontsize=16)

plt.grid(True)
plt.legend(loc='upper left', fontsize=14)

plt.show()

# Print the median for each year
for i, year in enumerate(years_list):
    print(f"Year {year}: Median = {medians[i]:.2f}")

###################### p q ##########################
import numpy as np

# Example Initialize the value range of p
p_values = np.arange(0.85, 0.55, -0.01)

# Calculate the corresponding q value
q_values = 1 - p_values

# Calculate alpha
alpha_values = np.log(q_values) / (np.log(q_values) - np.log(p_values))

for p, alpha in zip(p_values, alpha_values):
    print(f"p = {p:.3f}, alpha = {alpha:.4f}")

#######
from scipy.optimize import bisect

columns = [col for col in df.columns if col.startswith("W")]

# Initializes the list that stores alpha values, p, and q
alpha_values = []
p_values = []
q_values = []

# Iterate through the years
for year_column in columns:
    year_data = df[year_column].dropna()

    # Fit the data to the Pareto distribution and get the alpha parameter
    fit_params = pareto.fit(year_data)
    alpha = fit_params[0]

    # Defining equation
    def equation(b):
        return alpha * (np.log(b) - np.log(1 - b)) - np.log(b)

    # Use dichotomies to approximate solutions
    q = bisect(equation, 0.01, 0.99)
    p = 1 -q

    # Add alpha, p, and q to the appropriate list
    alpha_values.append(alpha)
    p_values.append(p)
    q_values.append(q)

for i, year in enumerate(range(2001, 2024)):
    print(f"Year {year}: Alpha = {alpha_values[i]:.4f}, p = {p_values[i]:.4f}, q = {q_values[i]:.4f}")

# Create a stacked bar chart
plt.figure(figsize=(12, 6))
bars_p = plt.bar(columns, p_values, label='p', alpha=0.7, color='darkblue', width=0.4)
bars_q = plt.bar(columns, q_values, bottom=p_values, label='q', alpha=0.7, color='darkred', width=0.4)

plt.xlabel("Year", fontsize=20)
plt.ylabel("Value", fontsize=20)
# plt.title("p and q Values Over the Years (2001-2023)", fontsize=24)

plt.xticks(columns, [str(year) for year in range(2001, 2024)], rotation=90, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

plt.show()

###################### Gender ################

columns = [col for col in df.columns if col.startswith("G")]

# Initializes the list of male and female percentages to store
male_percentages = []
female_percentages = []
years = []

# Iterate through the years
for year_column in columns:
    male_data = df[year_column].str.count('M').sum()
    female_data = df[year_column].str.count('F').sum()

    # Calculate the proportion of men and women
    total = male_data + female_data
    male_percentage = (male_data / total) * 100
    female_percentage = (female_data / total) * 100

    # Add the proportion and year to the appropriate list
    male_percentages.append(male_percentage)
    female_percentages.append(female_percentage)
    years.append(int(year_column[1:]))

# Create a line chart to visualize the percentage of women and men each year
plt.figure(figsize=(12, 6))

# Draw a line chart of the proportion of women
plt.plot(years, female_percentages, label='Female %', marker='o', color='black', linewidth=2)

plt.xlabel("Year", fontsize=20)
plt.ylabel("Percentage (%)", fontsize=20)
# plt.title("Female Percentage Over the Years (2010-2023)", fontsize=24)

plt.xticks(years, [str(year) for year in range(2010, 2024)], rotation=90, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper right', fontsize=14, bbox_to_anchor=(1.21, 1))

plt.grid(True)
plt.show()

######################### Industry #####################

columns = [col for col in df.columns if col.startswith("I")]

# Define the required industry list
industries = ["Consumer", "Financials", "Communication Services", "Information Technology",
              "Real Estate", "Industrials", "Energy", "HealthCare", "Material"]

# Initializes dictionaries that store industry percentages per year
industry_percentages = {industry: [] for industry in industries}
years = []

# Iterate through the years
for year_column in columns:
    year_data = {industry: df[year_column].str.count(industry).sum() for industry in industries}

    # Calculate the total, used to calculate the percentage
    total = sum(year_data.values())

    # Calculate the percentage for each industry
    percentages = {industry: (count / total) * 100 for industry, count in year_data.items()}

    # Add the percentage and year to the appropriate list
    for industry in industries:
        industry_percentages[industry].append(percentages[industry])
    years.append(int(year_column[1:]))

# Create stacked bar charts to visualize industry percentages for each year
plt.figure(figsize=(12, 6))

# Define the colors for each industry and customize them according to your needs
industry_colors = {
    "Consumer": "darkblue",
    "Financials": "darkgreen",
    "Communication Services": "darkorange",
    "Information Technology": "darkred",
    "Real Estate": "purple",
    "Industrials": "saddlebrown",
    "Energy": "blue",
    "HealthCare": "gray",
    "Material": "cyan"
}

# Plot the percentage for each industry and use the specified color
bottom = [0] * len(years)
for industry in industries:
    plt.bar(years, industry_percentages[industry], label=industry, bottom=bottom, color=industry_colors[industry])
    bottom = [bottom[i] + industry_percentages[industry][i] for i in range(len(years))]

plt.xlabel("Year", fontsize=20)
plt.ylabel("Percentage (%)", fontsize=20)
# plt.title("Industry Percentage Over the Years (2001-2023)", fontsize=24)


handles, labels = plt.gca().get_legend_handles_labels()
sorted_order = sorted(range(len(labels)), key=lambda i: labels[i].split(" ")[0])
sorted_handles = [handles[i] for i in sorted_order]
sorted_labels = [labels[i] for i in sorted_order]
plt.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

plt.tight_layout()

plt.xticks(years, [str(year) for year in range(2001, 2024)], rotation=90, fontsize=16)
plt.yticks(fontsize=16)

plt.grid(True)
plt.show()



import pandas as pd
import numpy as np
from scipy.stats import pareto
import matplotlib.pyplot as plt

file_path ="/Users/xiaohuidexiaojiaoqi/Desktop/Money for different gender.csv"# Update with your actual file name
data = pd.read_csv(file_path)

columns = data.columns

def fit_pareto_and_get_alpha(column_data):
    column_data = np.array([float(value) for value in column_data if is_numeric(value)], dtype=float)
    valid_data = column_data[(column_data > 0) & np.isfinite(column_data)]

    if len(valid_data) < 2:
        return None

    shape, loc, scale = pareto.fit(valid_data)
    return shape


def is_numeric(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False



alpha_values = {}


for column in columns:
    alpha = fit_pareto_and_get_alpha(data[column])

    if alpha is not None:
        alpha_values[column] = alpha


female_alpha = [alpha_values[f'FemaleWorth{i}'] for i in range(2010, 2024)]
male_alpha = [alpha_values[f'MaleWorth{i}'] for i in range(2010, 2024)]


years = range(2010, 2024)


plt.figure(figsize=(10, 6))
plt.plot(years, female_alpha, marker='o', color='black', label='Female Alpha', linewidth=2)
plt.plot(years, male_alpha, '--', color='black', marker='o', label='Male Alpha', linewidth=2)


plt.legend(fontsize=14)

plt.xlabel('Year',fontsize=20)
plt.ylabel('Alpha',fontsize=20)
#plt.title('Pareto Alpha Values for Female and Male (2010-2023)',fontsize=24)

plt.xticks(np.arange(2010, 2024, 1), rotation='vertical', fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(1.4, 2.6)
plt.legend(loc='upper left', fontsize=14)
plt.grid(True)


plt.savefig('alpha_values_plot.png')

plt.show()

import numpy as np
import pandas as pd
from scipy.stats import pareto
import matplotlib.pyplot as plt

# 定义文件路径和洲名
file_paths = [
    ("/Users/xiaohuidexiaojiaoqi/Desktop/Asia_Fortune.csv", "Asia"),
    ("/Users/xiaohuidexiaojiaoqi/Desktop/Europe_Fortune.csv", "Europe"),
    ("/Users/xiaohuidexiaojiaoqi/Desktop/North_America_Fortune.csv", "North America")
]
# 创建一个空的DataFrame来存储alpha值
alpha_dfs = []



# 循环遍历每一个文件
for file_path, continent_name in file_paths:
    data = pd.read_csv(file_path)
    alpha_df = pd.DataFrame(columns=["Year", "Alpha"])

    # 循环遍历每一年的数据
    for year in range(2001, 2024):
        # 选择当前年份的数据
        year_data = data[data["Year"] == year]

        # 获取Worth列的数据
        worth_data = year_data["Worth"].values

        # 使用Pareto分布拟合数据
        fit_params = pareto.fit(worth_data, loc=0, scale=1)

        # 获取alpha值（拟合参数的第一个值）
        fit_alpha = fit_params[0]

        # 将结果添加到alpha_df中
        alpha_df = pd.concat([alpha_df, pd.DataFrame({"Year": [year], "Alpha": [fit_alpha]})], ignore_index=True)

    alpha_dfs.append((alpha_df, continent_name))


# 在拼接之前，检查并删除空的或全是 NA 值的列
alpha_df = alpha_df.dropna(axis=1, how='all')
# 进行拼接
alpha_df = pd.concat([alpha_df, pd.DataFrame({"Year": [year], "Alpha": [fit_alpha]})], ignore_index=True)


plt.figure(figsize=(12, 6))
linestyles = ['-', '--', ':']  # 实线、虚线、实线虚线相间
colors = ['black', 'black', 'black']  # 所有线条都是黑色

for (alpha_df, continent_name), linestyle, color in zip(alpha_dfs, linestyles, colors):
    plt.plot(alpha_df["Year"], alpha_df["Alpha"], marker='o', linestyle=linestyle, label=continent_name, color=color, linewidth=2)

    # 在折线图上标出每个数据点
    for year, alpha in zip(alpha_df["Year"], alpha_df["Alpha"]):
        plt.scatter(year, alpha, color=color, s=50)


#plt.title("Top Three Continents Alpha Over the Years")

plt.xlabel('Year',fontsize=20)
plt.ylabel('Alpha',fontsize=20)
plt.grid(True)
plt.legend(loc='upper right', fontsize=14)
plt.xticks(np.arange(2001, 2024, 1), rotation='vertical', fontsize=16)
plt.yticks(fontsize=16)

plt.show()



