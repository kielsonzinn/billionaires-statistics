# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:48.299139Z","iopub.execute_input":"2023-10-29T08:48:48.2999Z","iopub.status.idle":"2023-10-29T08:48:48.756958Z","shell.execute_reply.started":"2023-10-29T08:48:48.299858Z","shell.execute_reply":"2023-10-29T08:48:48.755851Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

from builder.px_builder import PxBuilder
from template.form_final_worth_by_country_display import FormFinalWorthByCountryDisplay

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:48.758748Z","iopub.execute_input":"2023-10-29T08:48:48.759302Z","iopub.status.idle":"2023-10-29T08:48:51.024936Z","shell.execute_reply.started":"2023-10-29T08:48:48.759255Z","shell.execute_reply":"2023-10-29T08:48:51.024036Z"}}
import pandas as pd
import seaborn as sns

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:51.02622Z","iopub.execute_input":"2023-10-29T08:48:51.027093Z","iopub.status.idle":"2023-10-29T08:48:51.08606Z","shell.execute_reply.started":"2023-10-29T08:48:51.027054Z","shell.execute_reply":"2023-10-29T08:48:51.085166Z"}}
billionaires = pd.read_csv('billionaires-statistics-dataset.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:51.089354Z","iopub.execute_input":"2023-10-29T08:48:51.090238Z","iopub.status.idle":"2023-10-29T08:48:51.121219Z","shell.execute_reply.started":"2023-10-29T08:48:51.09019Z","shell.execute_reply":"2023-10-29T08:48:51.120135Z"}}
billionaires.head().T

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:51.122718Z","iopub.execute_input":"2023-10-29T08:48:51.123065Z","iopub.status.idle":"2023-10-29T08:48:51.158689Z","shell.execute_reply.started":"2023-10-29T08:48:51.123033Z","shell.execute_reply":"2023-10-29T08:48:51.157447Z"}}
# Get the info of the dataset
billionaires.info()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:51.160246Z","iopub.execute_input":"2023-10-29T08:48:51.160678Z","iopub.status.idle":"2023-10-29T08:48:51.230308Z","shell.execute_reply.started":"2023-10-29T08:48:51.160638Z","shell.execute_reply":"2023-10-29T08:48:51.228945Z"}}
# Get the summary of the dataset
billionaires.describe()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:51.231605Z","iopub.execute_input":"2023-10-29T08:48:51.231937Z","iopub.status.idle":"2023-10-29T08:48:51.60334Z","shell.execute_reply.started":"2023-10-29T08:48:51.231907Z","shell.execute_reply":"2023-10-29T08:48:51.60197Z"}}
import matplotlib.pyplot as plt

# Filter the data to only include billionaires who are at least 30 years old
filtered_billionaires = billionaires[(billionaires['age'] >= 30) & (billionaires['finalWorth'] > 0)]

# Create a scatter plot to show the relationship between age and net worth
plt.scatter(filtered_billionaires['age'], filtered_billionaires['finalWorth'], color='red')
plt.xlabel('Age')
plt.ylabel('Net Worth')
plt.title('Relationship Between Age and Net Worth for Billionaires')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:51.605086Z","iopub.execute_input":"2023-10-29T08:48:51.605551Z","iopub.status.idle":"2023-10-29T08:48:52.307078Z","shell.execute_reply.started":"2023-10-29T08:48:51.60551Z","shell.execute_reply":"2023-10-29T08:48:52.305888Z"}}
# Plot a histogram of the distribution of ages
billionaires['age'].plot(kind='hist', bins=20)

# Plot a scatter plot of the relationship between age and net worth
billionaires.plot(x='age', y='finalWorth', kind='scatter')

# Calculate the correlation coefficient between age and net worth
correlation = billionaires['age'].corr(billionaires['finalWorth'])
print(f'Correlation between age and net worth: {correlation}\n')

# Calculate the mean and standard deviation of the net worth column
mean = billionaires['finalWorth'].mean()
std = billionaires['finalWorth'].std()
print(f'Mean of net worth: {mean},\nStandard Deviation of net worth: {std}\n')

# Calculate the percentage of billionaires who are male
male_count = len(billionaires[billionaires['gender'] == 'M'])
total_count = len(billionaires)
percentage = male_count / total_count * 100
print(f'Percentage of billionaires who are male: {percentage}%\n')

# Calculate the average net worth of billionaires by country
avg_finalWorth = billionaires.groupby('country')['finalWorth'].mean()
print(f'Average net worth of billionaires by country: {avg_finalWorth}\n')

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:52.308623Z","iopub.execute_input":"2023-10-29T08:48:52.309736Z","iopub.status.idle":"2023-10-29T08:48:52.777949Z","shell.execute_reply.started":"2023-10-29T08:48:52.30969Z","shell.execute_reply":"2023-10-29T08:48:52.777048Z"}}
# Get the distribution of the 'finalWorth' column
sns.distplot(billionaires['finalWorth'])

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:52.782238Z","iopub.execute_input":"2023-10-29T08:48:52.782893Z","iopub.status.idle":"2023-10-29T08:48:52.84218Z","shell.execute_reply.started":"2023-10-29T08:48:52.782859Z","shell.execute_reply":"2023-10-29T08:48:52.841043Z"}}
billionaires.dropna()
# Replace NaN values with None
billionaires.latitude_country = billionaires.latitude_country.fillna(0.0)
billionaires.longitude_country = billionaires.longitude_country.fillna(0.0)
billionaires.fillna(0)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:52.843598Z","iopub.execute_input":"2023-10-29T08:48:52.84394Z","iopub.status.idle":"2023-10-29T08:48:52.852005Z","shell.execute_reply.started":"2023-10-29T08:48:52.84391Z","shell.execute_reply":"2023-10-29T08:48:52.851164Z"}}
numerical_columns = billionaires.select_dtypes(include=['int64', 'float64']).columns

print(numerical_columns)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:52.853462Z","iopub.execute_input":"2023-10-29T08:48:52.854754Z","iopub.status.idle":"2023-10-29T08:48:54.384143Z","shell.execute_reply.started":"2023-10-29T08:48:52.854718Z","shell.execute_reply":"2023-10-29T08:48:54.383198Z"}}

import matplotlib.pyplot as plt
import seaborn as sns

# Get the correlation matrix of the numerical columns
corr_matrix = billionaires[numerical_columns].corr()

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, ax=ax)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:54.385147Z","iopub.execute_input":"2023-10-29T08:48:54.385472Z","iopub.status.idle":"2023-10-29T08:48:54.736135Z","shell.execute_reply.started":"2023-10-29T08:48:54.385444Z","shell.execute_reply":"2023-10-29T08:48:54.734846Z"}}
import matplotlib.pyplot as plt

# Calculate the average final worth of billionaires for each year
average_final_worth_by_year = billionaires.groupby('age')['finalWorth'].mean()

# Create a line chart showing the trend of the average final worth of billionaires over time
plt.figure(figsize=(10, 6))
plt.plot(average_final_worth_by_year.index, average_final_worth_by_year.values)
plt.title('Trend of Average Final Worth of Billionaires by Age')
plt.xlabel('Age')
plt.ylabel('Average Final Worth (USD Millions)')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:54.737543Z","iopub.execute_input":"2023-10-29T08:48:54.738518Z","iopub.status.idle":"2023-10-29T08:48:55.10375Z","shell.execute_reply.started":"2023-10-29T08:48:54.738476Z","shell.execute_reply":"2023-10-29T08:48:55.10259Z"}}
from bokeh.plotting import figure, show

# Create an interactive scatter plot showing the relationship between final worth and age
bokeh_plot = figure(
    title='Relationship Between Final Worth and Age',
    x_axis_label='Final Worth (USD Millions)',
    y_axis_label='Age',
)
bokeh_plot.circle(x='finalWorth', y='age', size=10, color='blue', alpha=0.5, source=billionaires)
show(bokeh_plot)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:55.105551Z","iopub.execute_input":"2023-10-29T08:48:55.106043Z","iopub.status.idle":"2023-10-29T08:48:57.632554Z","shell.execute_reply.started":"2023-10-29T08:48:55.10596Z","shell.execute_reply":"2023-10-29T08:48:57.631359Z"}}
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, show

fig, ax = plt.subplots(figsize=(24, 48))

# Box plot of final worth by country
sns.boxplot(
    y='country',
    x='finalWorth',
    showmeans=True,
    data=billionaires, ax=ax
)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:57.634475Z","iopub.execute_input":"2023-10-29T08:48:57.634924Z","iopub.status.idle":"2023-10-29T08:48:58.344568Z","shell.execute_reply.started":"2023-10-29T08:48:57.634881Z","shell.execute_reply":"2023-10-29T08:48:58.343585Z"}}
# Histogram of the age distribution of billionaires
sns.histplot(
    x='age',
    data=billionaires,
)
plt.xlabel('Age')
plt.ylabel('Number of Billionaires')
plt.title('Age Distribution of Billionaires')
plt.show()

# Scatter plot of final worth vs. age, with color coded by country
bokeh_plot = figure(x_axis_label='Age', y_axis_label='Final Worth')
bokeh_plot.scatter(
    x='age',
    y='finalWorth',
    color='country',
    source=billionaires,
)
show(bokeh_plot)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:58.345932Z","iopub.execute_input":"2023-10-29T08:48:58.346304Z","iopub.status.idle":"2023-10-29T08:48:58.399686Z","shell.execute_reply.started":"2023-10-29T08:48:58.346273Z","shell.execute_reply":"2023-10-29T08:48:58.39853Z"}}
# Bar chart of the number of billionaires by self-made status and gender
billionaires_by_self_made_status_and_gender = billionaires[['selfMade', 'gender']].value_counts().unstack()
bokeh_plot = figure(x_axis_label='Gender', y_axis_label='Number of Billionaires')
bokeh_plot.vbar(
    x='gender',
    top='selfMade_D',
    width=0.5,
    bottom=0,
    source=billionaires_by_self_made_status_and_gender,
    color='blue',
)
bokeh_plot.vbar(
    x='gender',
    top='selfMade_U',
    width=0.5,
    bottom=0,
    source=billionaires_by_self_made_status_and_gender,
    color='red',
)
show

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:58.401757Z","iopub.execute_input":"2023-10-29T08:48:58.402943Z","iopub.status.idle":"2023-10-29T08:48:58.742727Z","shell.execute_reply.started":"2023-10-29T08:48:58.402896Z","shell.execute_reply":"2023-10-29T08:48:58.741612Z"}}


# Bokeh interactive scatter plot to explore the relationship between final worth and age, country, and industry
bokeh_plot = figure(
    x_axis_label='Age',
    y_axis_label='Final Worth',
    tools='pan,wheel_zoom,box_zoom,reset',
)
bokeh_plot.scatter(
    x='age',
    y='finalWorth',
    color='country',
    fill_color='industries',
    source=billionaires,
)
show(bokeh_plot)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:48:58.744045Z","iopub.execute_input":"2023-10-29T08:48:58.74437Z","iopub.status.idle":"2023-10-29T08:49:07.903909Z","shell.execute_reply.started":"2023-10-29T08:48:58.744341Z","shell.execute_reply":"2023-10-29T08:49:07.902962Z"}}
import folium

# import color_palette as cp

billionaires.dropna()
# Replace NaN values with None
billionaires.latitude_country = billionaires.latitude_country.fillna(0.0)
billionaires.longitude_country = billionaires.longitude_country.fillna(0.0)

# Create a folium map
map = folium.Map(location=[0, 0], zoom_start=2)

# Add markers to the map for each billionaire, colored by industry
for index, billionaire in billionaires.iterrows():
    folium.Marker(
        location=[billionaire.latitude_country, billionaire.longitude_country],
        popup=f'{billionaire.personName} ({billionaire.finalWorth:.2f}B)',
        #       color=billionaire.industry,
    ).add_to(map)

# Add a layer control to the map
folium.LayerControl().add_to(map)

# Display the map
map.save('billionaires_map.html')
map

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:49:07.90525Z","iopub.execute_input":"2023-10-29T08:49:07.905573Z","iopub.status.idle":"2023-10-29T08:50:23.559374Z","shell.execute_reply.started":"2023-10-29T08:49:07.905545Z","shell.execute_reply":"2023-10-29T08:50:23.550359Z"}}
import networkx as nx

# Create a network graph to show the connections between billionaires and their businesses
fig = plt.figure(1, figsize=(200, 80), dpi=60)
G = nx.Graph()
for i, row in billionaires.iterrows():
    G.add_node(row['personName'])
    G.add_edge(row['personName'], row['source'])
nx.draw_networkx(G, pos=nx.draw_spectral(G), with_labels=True, font_weight='normal')
plt.title('Connections Between Billionaires and Their Businesses')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:50:23.561172Z","iopub.execute_input":"2023-10-29T08:50:23.561549Z","iopub.status.idle":"2023-10-29T08:50:28.268663Z","shell.execute_reply.started":"2023-10-29T08:50:23.561515Z","shell.execute_reply":"2023-10-29T08:50:28.267469Z"}}
# Create a map to show the location of billionaires by country
m = folium.Map(location=[40, -100], zoom_start=4)
for i, row in billionaires.iterrows():
    folium.Marker([row['latitude_country'], row['longitude_country']], popup=row['personName']).add_to(m)
m

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:50:28.270403Z","iopub.execute_input":"2023-10-29T08:50:28.27078Z","iopub.status.idle":"2023-10-29T08:50:31.545417Z","shell.execute_reply.started":"2023-10-29T08:50:28.270746Z","shell.execute_reply":"2023-10-29T08:50:31.544243Z"}}
sns.catplot(y='country', x='finalWorth', data=billionaires, kind='bar')
plt.title('Distribution of Billionaires by Country')
plt.ylabel('Country')
plt.xlabel('Net Worth')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:50:31.546725Z","iopub.execute_input":"2023-10-29T08:50:31.547102Z","iopub.status.idle":"2023-10-29T08:50:31.94522Z","shell.execute_reply.started":"2023-10-29T08:50:31.54707Z","shell.execute_reply":"2023-10-29T08:50:31.944063Z"}}
sns.relplot(y='age', x='finalWorth', data=billionaires, kind='scatter')
plt.title('Relationship Between Age and Net Worth for Billionaires')
plt.ylabel('Age')
plt.xlabel('Net Worth')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:50:31.94675Z","iopub.execute_input":"2023-10-29T08:50:31.947203Z","iopub.status.idle":"2023-10-29T08:50:35.101289Z","shell.execute_reply.started":"2023-10-29T08:50:31.947161Z","shell.execute_reply":"2023-10-29T08:50:35.100131Z"}}
x = plt.figure(figsize=(10, 22))
ax = sns.barplot(y=billionaires["country"],
                 x=billionaires["finalWorth"], estimator=sum, color="y")
ax.set_title('Country vs Worth ')
ax.set(xlabel='Final Worth', ylabel='Country')

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:50:35.10272Z","iopub.execute_input":"2023-10-29T08:50:35.103879Z","iopub.status.idle":"2023-10-29T08:50:36.610804Z","shell.execute_reply.started":"2023-10-29T08:50:35.103839Z","shell.execute_reply":"2023-10-29T08:50:36.609629Z"}}
x = plt.figure(figsize=(10, 22))
ax = sns.barplot(y=billionaires["industries"],
                 x=billionaires["finalWorth"], estimator=sum, color="b")
ax.set_title('Industry vs Worth ')
ax.set(xlabel='Final Worth', ylabel='Industry')

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:50:36.612524Z","iopub.execute_input":"2023-10-29T08:50:36.612882Z","iopub.status.idle":"2023-10-29T08:50:36.840417Z","shell.execute_reply.started":"2023-10-29T08:50:36.612851Z","shell.execute_reply":"2023-10-29T08:50:36.839164Z"}}
plt.boxplot(billionaires['finalWorth'])
plt.title('Final Worth Distribution')
plt.ylabel('Final Worth')

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:50:36.84185Z","iopub.execute_input":"2023-10-29T08:50:36.842215Z","iopub.status.idle":"2023-10-29T08:50:37.877195Z","shell.execute_reply.started":"2023-10-29T08:50:36.842183Z","shell.execute_reply":"2023-10-29T08:50:37.876014Z"}}
# Distribution of final worth
plt.hist(billionaires['finalWorth'], bins=50)
plt.xlabel('Final Worth (USD)')
plt.ylabel('Number of Billionaires')
plt.title('Distribution of Final Worth')
plt.show()

#  chart of final worth by category
plt.scatter(billionaires['finalWorth'], billionaires['category'])
plt.xlabel('Category')
plt.ylabel('Final Worth (USD)')
plt.title('Final Worth by Category')
plt.show()

# Scatter plot of final worth vs. age
plt.scatter(billionaires['finalWorth'], billionaires['age'])
plt.ylabel('Age')
plt.xlabel('Final Worth (USD)')
plt.title('Final Worth vs. Age')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:50:37.883938Z","iopub.execute_input":"2023-10-29T08:50:37.8843Z","iopub.status.idle":"2023-10-29T08:50:39.550958Z","shell.execute_reply.started":"2023-10-29T08:50:37.884269Z","shell.execute_reply":"2023-10-29T08:50:39.549799Z"}}
# Heatmap of final worth by country
fig, ax = plt.subplots(figsize=(20, 40))
sns.heatmap(billionaires.pivot_table(index='country', values='finalWorth', aggfunc='sum'), annot=True, fmt='.2f', ax=ax)
plt.title('Final Worth by Country')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:50:39.552364Z","iopub.execute_input":"2023-10-29T08:50:39.552807Z","iopub.status.idle":"2023-10-29T08:50:42.575502Z","shell.execute_reply.started":"2023-10-29T08:50:39.552776Z","shell.execute_reply":"2023-10-29T08:50:42.574304Z"}}
import plotly.express as px

form_age = FormFinalWorthByCountryDisplay()
form_age.set_data(billionaires)
form_age.show()

# Bubble chart of final worth vs. age
fig = px.scatter(billionaires, x='age', y='finalWorth', size='finalWorth', color='category')
fig.update_layout(title='Final Worth vs. Age')
fig.show()

# 3D scatter plot of final worth vs. age and category
fig = px.scatter_3d(billionaires, x='age', y='finalWorth', z='category', color='category')
fig.update_layout(title='Final Worth vs. Age and Category')
fig.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:50:42.576872Z","iopub.execute_input":"2023-10-29T08:50:42.577545Z","iopub.status.idle":"2023-10-29T08:52:41.940573Z","shell.execute_reply.started":"2023-10-29T08:50:42.57751Z","shell.execute_reply":"2023-10-29T08:52:41.939313Z"}}
sns.pairplot(billionaires)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:52:41.94197Z","iopub.execute_input":"2023-10-29T08:52:41.942428Z","iopub.status.idle":"2023-10-29T08:52:42.055305Z","shell.execute_reply.started":"2023-10-29T08:52:41.942393Z","shell.execute_reply":"2023-10-29T08:52:42.054143Z"}}
# Create a parallel coordinates plot to show the relationship between different features of billionaires
fig = px.parallel_coordinates(billionaires,
                              dimensions=['age', 'finalWorth', 'rank', 'cpi_country',
                                          'gdp_country', 'gross_tertiary_education_enrollment',
                                          'gross_primary_education_enrollment_country', 'life_expectancy_country',
                                          'tax_revenue_country_country', 'total_tax_rate_country',
                                          'population_country'])
fig.update_layout(title='Relationship Between Different Features of Billionaires')
fig.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:54:27.891468Z","iopub.execute_input":"2023-10-29T08:54:27.891893Z","iopub.status.idle":"2023-10-29T08:54:28.118653Z","shell.execute_reply.started":"2023-10-29T08:54:27.891856Z","shell.execute_reply":"2023-10-29T08:54:28.117492Z"}}
import plotly.express as px

# Create a bar chart of the number of billionaires in each category
fig = px.bar(billionaires, x='category', y='rank', color='finalWorth')
fig.update_layout(title='Number of Billionaires in Each Category')
fig.show()

# Create a scatter plot of final worth vs. age
fig = px.scatter(billionaires, x='age', y='finalWorth')
fig.update_layout(title='Final Worth vs. Age')
fig.show()

# Create a choropleth map of final worth by country
fig = px.choropleth(billionaires, locations='country', color='finalWorth', scope='world',
                    color_continuous_scale='Viridis')
fig.update_layout(title='Final Worth by Country')
fig.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:54:46.813705Z","iopub.execute_input":"2023-10-29T08:54:46.814108Z","iopub.status.idle":"2023-10-29T08:54:46.888201Z","shell.execute_reply.started":"2023-10-29T08:54:46.814076Z","shell.execute_reply":"2023-10-29T08:54:46.887111Z"}}

# Create a parallel coordinates plot of the relationship between different features of billionaires
pxBuilder = PxBuilder()
fig = pxBuilder.with_data(
    billionaires
).with_dimensions(
    ['age', 'finalWorth', 'rank', 'cpi_country', 'gdp_country', 'gross_tertiary_education_enrollment',
     'gross_primary_education_enrollment_country', 'life_expectancy_country', 'tax_revenue_country_country',
     'total_tax_rate_country', 'population_country']
).with_title(
    'Relationship Between Different Features of Billionaires'
).build()
fig.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:55:06.22448Z","iopub.execute_input":"2023-10-29T08:55:06.225652Z","iopub.status.idle":"2023-10-29T08:55:06.396135Z","shell.execute_reply.started":"2023-10-29T08:55:06.225602Z","shell.execute_reply":"2023-10-29T08:55:06.395025Z"}}

# Create a bubble chart of the relationship between age and final worth
fig = px.scatter(billionaires, x='age', y='finalWorth', size='finalWorth', color='category')
fig.update_layout(title='Relationship Between Age and Final Worth')
fig.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T08:59:21.201672Z","iopub.execute_input":"2023-10-29T08:59:21.20214Z","iopub.status.idle":"2023-10-29T08:59:21.267886Z","shell.execute_reply.started":"2023-10-29T08:59:21.202096Z","shell.execute_reply":"2023-10-29T08:59:21.266691Z"}}
# Create a pie chart of billionaires by country
fig = px.pie(billionaires.head(100), values='finalWorth', names='country', title='TOp-100 Billionaires by Country')
fig.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T09:00:50.504636Z","iopub.execute_input":"2023-10-29T09:00:50.50504Z","iopub.status.idle":"2023-10-29T09:00:50.608195Z","shell.execute_reply.started":"2023-10-29T09:00:50.505008Z","shell.execute_reply":"2023-10-29T09:00:50.606761Z"}}

# Create a bar chart of billionaires by country and industry
fig = px.bar(billionaires, x='country', y='rank', title='Billionaires by Country and Industry')
fig.show()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-29T09:01:22.642907Z","iopub.execute_input":"2023-10-29T09:01:22.6434Z","iopub.status.idle":"2023-10-29T09:01:22.717953Z","shell.execute_reply.started":"2023-10-29T09:01:22.643364Z","shell.execute_reply":"2023-10-29T09:01:22.716642Z"}}

# Create a scatter plot of final worth vs. age
fig = px.scatter(billionaires, x='age', y='finalWorth', title='Final Worth vs. Age')
fig.show()

# %% [markdown]
# # Work in progress
