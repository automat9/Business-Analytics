```python
# Orders handled by carrier (pie chart)


import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

# Load the dataset
file_path = "https://github.com/automat9/Business-Analytics/raw/master/Semester%201/Analytics%20and%20Visualisation%20for%20Managers%20and%20Consultants/datasets/data.xlsx"
data = pd.read_excel(file_path)

# Calculate the number of orders handled by each carrier
carrier_order_counts = data.groupby('Carrier').size().reset_index(name='Order Count')

# Create a pie chart
fig = px.pie(
    carrier_order_counts,
    names='Carrier',
    values='Order Count',
    title="Number of Orders Handled by Each Carrier"
)

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Carrier Order Distribution"),
    dcc.Graph(id='pie-chart', figure=fig)
])

# Run app
if __name__ == '__main__':
    app.run_server(debug=True, port=8026)

```



<iframe
    width="100%"
    height="650"
    src="http://127.0.0.1:8026/"
    frameborder="0"
    allowfullscreen

></iframe>




```python
# Chance of late delivery by carrier (bar chart) 

import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

# Load the dataset
file_path = "https://github.com/automat9/Business-Analytics/raw/master/Semester%201/Analytics%20and%20Visualisation%20for%20Managers%20and%20Consultants/datasets/data.xlsx"
data = pd.read_excel(file_path)

# Calculate the number of total deliveries and late deliveries for each carrier
carrier_totals = data.groupby('Carrier').size().reset_index(name='Total Deliveries')
carrier_late = data[data['Ship Late Day count'] > 0].groupby('Carrier').size().reset_index(name='Late Deliveries')

# Merge the two dataframes to calculate the late delivery likelihood
carrier_stats = pd.merge(carrier_totals, carrier_late, on='Carrier', how='left')
carrier_stats['Late Deliveries'] = carrier_stats['Late Deliveries'].fillna(0)  # Fill NaN for carriers with no late deliveries
carrier_stats['Late Delivery Likelihood (%)'] = (carrier_stats['Late Deliveries'] / carrier_stats['Total Deliveries']) * 100

# Create a bar chart for late delivery likelihood by carrier
fig = px.bar(
    carrier_stats,
    x='Carrier',
    y='Late Delivery Likelihood (%)',
    title="Likelihood of Late Delivery by Carrier",
    labels={'Late Delivery Likelihood (%)': 'Likelihood of Late Delivery (%)'},
    text='Late Delivery Likelihood (%)'
)

# Set bar color and add data labels for clarity
fig.update_traces(marker_color='coral', texttemplate='%{text:.2f}%', textposition='outside')

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Carrier Efficiency Analysis"),
    dcc.Graph(id='late-delivery-bar-chart', figure=fig)
])

# Run app
if __name__ == '__main__':
    app.run_server(debug=True, port=8032)


```



<iframe
    width="100%"
    height="650"
    src="http://127.0.0.1:8032/"
    frameborder="0"
    allowfullscreen

></iframe>




```python
# Top 20 products by late dispatch (bar chart)


# Import libraries
from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd

# File location
url = "https://github.com/automat9/Business-Analytics/raw/master/Semester%201/Analytics%20and%20Visualisation%20for%20Managers%20and%20Consultants/datasets/data.xlsx"
# Load and prepare data
top_10_products_df = (
    pd.read_excel(url, sheet_name='OrderList')
    .query('`Ship Late Day count` > 0')
    .assign(**{'Product ID': lambda df: df['Product ID'].astype(str)})
    ['Product ID']
    .value_counts()
    .nlargest(20)
    .reset_index(name='Late Order Count')
    .rename(columns={'index': 'Product ID'})
)

# Initialise the Dash app
app = Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(
        id='late-orders-bar-chart',
        figure=px.bar(
            top_10_products_df,
            x='Product ID',
            y='Late Order Count',
            title='Products by Most Late Orders',
            labels={'Product ID': 'Product ID', 'Late Order Count': 'Number of Late Orders'}
        ).update_traces(marker_color='steelblue')
    )
])

# Run the app
app.run_server(mode='inline', port=8091, dev_tools_ui=True, dev_tools_props_check=True)
```



<iframe
    width="100%"
    height="650"
    src="http://127.0.0.1:8091/"
    frameborder="0"
    allowfullscreen

></iframe>




```python
# Orders handled by port (bar chart)

import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

# Load the dataset
file_path = "https://github.com/automat9/Business-Analytics/raw/master/Semester%201/Analytics%20and%20Visualisation%20for%20Managers%20and%20Consultants/datasets/data.xlsx"
data = pd.read_excel(file_path)

# Group data by Origin Port and count the number of orders
orders_by_origin_port = data.groupby('Origin Port').size().reset_index(name='Number of Orders')

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Number of Orders per Origin Port"),
    dcc.Graph(
        id='origin-port-orders-bar-chart',
        figure=px.bar(
            orders_by_origin_port,
            x='Origin Port',
            y='Number of Orders',
            title="Number of Orders per Origin Port",
            labels={'Number of Orders': 'Orders'},
            text='Number of Orders'
        ).update_traces(texttemplate='%{text}', textposition='outside')
    )
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8036)
```



<iframe
    width="100%"
    height="650"
    src="http://127.0.0.1:8036/"
    frameborder="0"
    allowfullscreen

></iframe>




```python
# Ship on-time/ahead of schedule (gauge chart)

import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objects as go

# Load the dataset
file_path = "https://github.com/automat9/Business-Analytics/raw/master/Semester%201/Analytics%20and%20Visualisation%20for%20Managers%20and%20Consultants/datasets/data.xlsx"
data = pd.read_excel(file_path)

# Calculate total deliveries and on-time/ahead deliveries
total_deliveries = len(data)
on_time_or_ahead_deliveries = len(data[(data['Ship Late Day count'] <= 0)])

# Calculate the percentage of on-time or ahead deliveries
on_time_or_ahead_percentage = (on_time_or_ahead_deliveries / total_deliveries) * 100

# Create the gauge chart
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=on_time_or_ahead_percentage,
    title={'text': "Percentage of On-Time or Ahead Deliveries"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "blue"},
        'steps': [
            {'range': [0, 50], 'color': "lightcoral"},
            {'range': [50, 100], 'color': "lightgreen"}
        ]
    }
))

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("On-Time or Ahead Delivery Performance"),
    dcc.Graph(id='gauge-chart', figure=fig)
])

# Run app
if __name__ == '__main__':
    app.run_server(debug=True, port=8011)
```



<iframe
    width="100%"
    height="650"
    src="http://127.0.0.1:8011/"
    frameborder="0"
    allowfullscreen

></iframe>


