import plotly.graph_objects as go  # or plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html

from Code.DataFilter import get_subset_data


# Monthly graph.
def generate_monthly_graph(data):
    data['Date'] = data['Date'].apply(lambda x: x.strftime('%Y-%m'))
    return data


# Yearly graph.
def generate_yearly_graph(data):
    data['Date'] = data['Date'].apply(lambda x: x.strftime('%Y'))
    return data


# Generate graph based on the given input
def generate_graph(filepath, frequency, gender, start_date, end_date):
    # Get subset of data
    data = get_subset_data(filepath, gender,
                           start_date, end_date)
    if frequency == "Month":
        data = generate_monthly_graph(data)
    elif frequency == "Year":
        data = generate_yearly_graph(data)

    # Calculate mean for each topic based on date
    df_mean = (data.groupby('Date', as_index=False)
        .agg({'0': 'mean', '1': 'mean', '2': 'mean', '3': 'mean', '4': 'mean', '5': 'mean', '6': 'mean', '7': 'mean',
              '8': 'mean', '9': 'mean',
              '10': 'mean', '11': 'mean',
              '12': 'mean', '13': 'mean',
              '14': 'mean'}).rename(
        columns={'0': 'Topic#1', '1': 'Topic#2', '2': 'Topic#3', '3': 'Topic#4', '4': 'Topic#5', '5': 'Topic#6',
                 '6': 'Topic#7', '7': 'Topic#8', '8': 'Topic#9',
                 '9': 'Topic#10',
                 '10': 'Topic#11',
                 '11': 'Topic#12',
                 '12': 'Topic#13',
                 '13': 'Topic#14',
                 '14': 'Topic#15'}))

    # X-axis is the dates
    x = df_mean['Date']

    # Add y axis data
    fig = go.Figure(go.Bar(x=x, y=df_mean['Topic#1'].tolist(), name='Topic#1'))
    # Add further y axis data
    for i in range(2, 16):
        col_name = "Topic#" + str(i)
        y_cur = df_mean[col_name].tolist()
        fig.add_trace(go.Bar(x=x, y=y_cur, name=col_name))

    # Update the graph layout
    fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})

    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])

    # Run the server
    app.run_server(debug=True, use_reloader=False)

# Generate the graph (filename, aggregae_by, gender, start_date, end_date)
generate_graph("./../Output/Tweet_Data_With_Topic_Proportion_15.csv", "Month", "", "", "")
