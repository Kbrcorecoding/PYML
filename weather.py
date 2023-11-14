import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

data = pd.read_csv("DailyDelhiClimateTrain.csv")



#changing the format to access month and year

data["date"] = pd.to_datetime(data["date"], format= '%Y-%m-%d')
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
print(data.head())

#scatter graph, with circle size reflecting mean temp
figure = px.scatter(data_frame = data, x="humidity",
                    y="meantemp", size="meantemp", 
                    trendline="ols", 
                    title = "Relationship Between Temperature and Humidity")
#figure.show()


plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.title("Temperature Change in Delhi Over the Years")
sns.lineplot(data = data, x='month', y='meantemp', hue='year')


#using matplotlib to plot more complex, less html/interactive side graphs;with seaborn which is built on top of matplotlib
#plt.show()





forecast_data = data.rename(columns={"date": "ds", "meantemp":"y"})
#The prophet model accepts time data named as “ds”, and labels as “y”. 


model=Prophet()
model.fit(forecast_data)

forecasts = model.make_future_dataframe(periods=365)
predictions = model.predict(forecasts)

fig=plot_plotly(model,predictions)
fig.show()