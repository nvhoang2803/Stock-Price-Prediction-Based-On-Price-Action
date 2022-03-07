import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import talib
pd.core.common.is_list_like = pd.api.types.is_list_like
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yahoo_finance
yahoo_finance.pdr_override()
from mpl_finance import candlestick2_ohlc

from argparse import ArgumentParser

def createZigZagPoints(dfSeries, minSegSize=0.1, sizeInDevs=0.5):
    minRetrace = minSegSize
    print('dfSeries')
    print(type(dfSeries))
    print(dfSeries)
    curVal = dfSeries[0]
    curPos = dfSeries.index[0]
    curDir = 1
    dfRes = pd.DataFrame(index=dfSeries.index, columns=["Dir", "Value"])
    for ln in dfSeries.index:
        if ((dfSeries[ln] - curVal) * curDir >= 0):
            curVal = dfSeries[ln]
            curPos = ln
        else:
            retracePrc = abs((dfSeries[ln] - curVal) / curVal * 100)
            if (retracePrc >= minRetrace):
                dfRes.loc[curPos, 'Value'] = curVal
                dfRes.loc[curPos, 'Dir'] = curDir
                curVal = dfSeries[ln]
                curPos = ln
                curDir = -1 * curDir
    dfRes[['Value']] = dfRes[['Value']].astype(float)
    return (dfRes)

parser = ArgumentParser(description='Algorithmic Support and Resistance')
parser.add_argument('-t', '--tickers', default='SPY500', type=str, required=False, help='Used to look up a specific tickers. Commma seperated. Example: MSFT,AAPL,AMZN default: List of S&P 500 companies')
parser.add_argument('-p', '--period', default='1d', type=str, required=False, help='Period to look back. valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max. default: 1d')
parser.add_argument('-i', '--interval', default='1m', type=str, required=False, help='Interval of each bar. valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo. default: 1m')
parser.add_argument('-d', '--dif', default='0.05', type=float, required=False, help='Max %% difference between two points to group them together. Default: 0.05')
parser.add_argument('--time', default='150', type=int, required=False, help='Max time measured in number of bars between two points to be grouped together. Default: 150')
parser.add_argument('-n', '--number', default='3', type=int, required=False, help='Min number of points in price range to draw a support/resistance line. Default: 3')
parser.add_argument('-m', '--min', default='150', type=int, required=False, help='Min number of bars from the start the support/resistance line has to be at to display chart. Default: 150')
args = parser.parse_args()
def drawResistance(ticker_df):
    print('Log data')
    print(ticker_df)
    # try:
    x_max = 0
    fig, ax = plt.subplots()
    dfRes = createZigZagPoints(ticker_df.Close).dropna()
    candlestick2_ohlc(ax, ticker_df['Open'], ticker_df['High'], ticker_df['Low'], ticker_df['Close'], width=0.6,
                      colorup='g', colordown='r')

    plt.plot(dfRes['Value'])
    plt.show()
    removed_indexes = []
    for index, row in dfRes.iterrows():
        if (not (index in removed_indexes)):
            dropindexes = []
            dropindexes.append(index)
            counter = 0
            values = []
            values.append(row.Value)
            startx = index
            endx = index
            dir = row.Dir
            print('dir')
            print(dir)
            for index2, row2 in dfRes.iterrows():
                if (not (index2 in removed_indexes)):
                    if (index != index2 and abs(index2 - index) < args.time and row2.Dir == dir):
                        if (abs((row.Value / row2.Value) - 1) < (args.dif / 100)):
                            dropindexes.append(index2)
                            values.append(row2.Value)
                            if (index2 < startx):
                                startx = index2
                            elif (index2 > endx):
                                endx = index2
                            counter = counter + 1
            print('counter')
            print(counter)
            print(args.number)
            if (counter > args.number):
                sum = 0
                print("Support at ", end='')
                for i in range(len(values) - 1):
                    print("{:0.2f} and ".format(values[i]), end='')
                print("{:0.2f} \n".format(values[len(values) - 1]), end='')
                removed_indexes.extend(dropindexes)
                for value in values:
                    sum = sum + value
                if (endx > x_max):
                    x_max = endx
                plt.hlines(y=sum / len(values), xmin=startx, xmax=endx, linewidth=1, color='r')
    if (x_max > args.min):
        # plt.title(ticker)
        plt.show()
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    # except Exception as e:
    #     print('EROORRRRRR')
    #     print(e)

def get_rsi_plotly_data(df_date, df_close):
    df_rsi = talib.RSI(df_close)

    plotly_rsi_data = []
    plotly_rsi_data.append(go.Scatter(
        x=df_date,
        y=[70] * len(df_date),
        name='overbought'
    ))
    plotly_rsi_data.append(go.Scatter(
        x=df_date,
        y=[30] * len(df_date),
        name='oversold'
    ))
    plotly_rsi_data.append(go.Scatter(
        x=df_date,
        y=df_rsi,
        name='rsi'
    ))
    rsi_layout = go.Layout(title='RSI',
                           xaxis=dict(title='Date',
                                      rangeslider=dict(visible=False)),
                           yaxis=dict(title='%',
                                      showgrid=True,
                                      color='black',
                                      gridwidth=1,
                                      gridcolor='LightPink'),
                           plot_bgcolor='white',
                           paper_bgcolor='white',

                           )
    return plotly_rsi_data, rsi_layout

def get_moving_average_graph(df_date, df_close, timeperiod=10):
    return go.Scatter(
        x=df_date,
        y=talib.MA(df_close, timeperiod),
        name=f'MA{timeperiod}'
    )

def predict_stock_from_model(ticker, model_name, folder):
    path = './'
    if folder:
        path = path + folder + "/"
    # Convert value into range(0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_ticker = pd.read_csv(f"{path}{ticker}.csv")

    df_ticker["Date"] = pd.to_datetime(df_ticker.Date, format="%Y-%m-%d")
    df_ticker.index = df_ticker['Date']

    data = df_ticker.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df_ticker)), columns=['Date', 'Close'])

    for i in range(0, len(data)):
        new_data["Date"][i] = data['Date'][i]
        new_data["Close"][i] = data["Close"][i]

    new_data.index = new_data.Date
    new_data.drop("Date", axis=1, inplace=True)

    dataset = new_data.values

    train = dataset[0:987, :]
    valid = dataset[987:, :]

    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []

    for i in range(60, len(train)):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = load_model(model_name)

    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    train = new_data[:987]
    valid = new_data[987:]
    valid['Predictions'] = closing_price

    mean_valid = valid['Close'].mean()
    mean_pred = valid['Predictions'].mean()
    accuracy = round((mean_valid / mean_pred) * 100, 2)
    print("Accuracy: ", accuracy)
    return valid

def get_candlestick_graph(ticker, folder, start_date):
    path= './'
    if folder:
        path = path + folder + "/"
    data_df = pd.read_csv(f"{path}{ticker}.csv")
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    data_df = data_df.sort_values('Date', ascending=True)
    data_df.head()
    if start_date:
        data_df = data_df[data_df['Date'] > start_date]

    return data_df, go.Candlestick(x=data_df['Date'],
                                      open=data_df['Open'],
                                      high=data_df['High'],
                                      low=data_df['Low'],
                                      close=data_df['Close'],
                                      name=ticker)

def main():
    ###### Compare predicted data vs actual data
    result = predict_stock_from_model(ticker='AAPL', model_name='saved_lstm_model.h5',folder='stock_data')

    ###### Draw candlestick graph and RSI
    plotly_data = []
    layout = []
    data_df, candlestick_graph = get_candlestick_graph(ticker='AAPL', folder='stock_data', start_date='2020-01-01')
    plotly_data.append(candlestick_graph)
    # drawResistance(data_df.reset_index())
    layout = go.Layout(title='AAPL',
                       xaxis=dict(title='Date',
                                  rangeslider=dict(visible=False)),
                       yaxis=dict(title='Price',
                                  showgrid=True,
                                  color='black',
                                  gridwidth=1,
                                  gridcolor='LightPink'),
                       plot_bgcolor='white',
                       paper_bgcolor='white',
                       )
    plotly_data.append(get_moving_average_graph(data_df['Date'], data_df['Close'], 10))
    plotly_data.append(get_moving_average_graph(data_df['Date'], data_df['Close'], 30))
    rsi_plotly_data, rsi_layout = get_rsi_plotly_data(data_df['Date'], data_df['Close'])

    ###### Compare Stockdata
    df = pd.read_csv("./stock_data.csv")

    ###### Render web
    app.layout = html.Div([

        html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
        dcc.Tabs(id="tabs", children=[
            dcc.Tab(label='Candlestick', children=[
                html.Div([
                    dcc.Graph(
                        id="Candlestick",
                        figure={
                            "data": plotly_data,
                            "layout": layout
                        }
                    )
                ]),
                html.Div([
                    dcc.Graph(
                        id="RSI",
                        figure={
                            "data": rsi_plotly_data,
                            "layout": rsi_layout
                        }
                    )
                ])
            ]),
            dcc.Tab(label='AAPL', children=[
                html.Div([
                    html.H2("Actual closing price", style={"textAlign": "center"}),
                    dcc.Graph(
                        id="Actual Data",
                        figure={
                            "data": [
                                go.Scatter(
                                    x=result.index,
                                    y=result["Close"],
                                    mode='markers'
                                )

                            ],
                            "layout": go.Layout(
                                title='scatter plot',
                                xaxis={'title': 'Date'},
                                yaxis={'title': 'Closing Rate'}
                            )
                        }

                    ),
                    html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
                    dcc.Graph(
                        id="Predicted Data",
                        figure={
                            "data": [
                                go.Scatter(
                                    x=result.index,
                                    y=result["Predictions"],
                                    mode='markers'
                                )

                            ],
                            "layout": go.Layout(
                                title='scatter plot',
                                xaxis={'title': 'Date'},
                                yaxis={'title': 'Closing Rate'}
                            )
                        }

                    )
                ])

            ]),
            dcc.Tab(label='Facebook Stock Data', children=[
                html.Div([
                    html.H1("Stocks High vs Lows",
                            style={'textAlign': 'center'}),

                    dcc.Dropdown(id='my-dropdown',
                                 options=[{'label': 'Tesla', 'value': 'TSLA'},
                                          {'label': 'Apple', 'value': 'AAPL'},
                                          {'label': 'Facebook', 'value': 'FB'},
                                          {'label': 'Microsoft', 'value': 'MSFT'}],
                                 multi=True, value=['FB'],
                                 style={"display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "60%"}),
                    dcc.Graph(id='highlow'),
                    html.H1("Stocks Market Volume", style={'textAlign': 'center'}),

                    dcc.Dropdown(id='my-dropdown2',
                                 options=[{'label': 'Tesla', 'value': 'TSLA'},
                                          {'label': 'Apple', 'value': 'AAPL'},
                                          {'label': 'Facebook', 'value': 'FB'},
                                          {'label': 'Microsoft', 'value': 'MSFT'}],
                                 multi=True, value=['FB'],
                                 style={"display": "block", "margin-left": "auto",
                                        "margin-right": "auto", "width": "60%"}),
                    dcc.Graph(id='volume')
                ], className="container"),
            ])

        ])
    ])

    ###### Handle action
    @app.callback(Output('highlow', 'figure'),
                  [Input('my-dropdown', 'value')])
    def update_graph(selected_dropdown):
        dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
        trace1 = []
        trace2 = []
        for stock in selected_dropdown:
            trace1.append(
                go.Scatter(x=df[df["Stock"] == stock]["Date"],
                           y=df[df["Stock"] == stock]["High"],
                           mode='lines', opacity=0.7,
                           name=f'High {dropdown[stock]}', textposition='bottom center'))
            trace2.append(
                go.Scatter(x=df[df["Stock"] == stock]["Date"],
                           y=df[df["Stock"] == stock]["Low"],
                           mode='lines', opacity=0.6,
                           name=f'Low {dropdown[stock]}', textposition='bottom center'))
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                                '#FF7400', '#FFF400', '#FF0056'],
                                      height=600,
                                      title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
                                      xaxis={"title": "Date",
                                             'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                                 'step': 'month',
                                                                                 'stepmode': 'backward'},
                                                                                {'count': 6, 'label': '6M',
                                                                                 'step': 'month',
                                                                                 'stepmode': 'backward'},
                                                                                {'step': 'all'}])},
                                             'rangeslider': {'visible': True}, 'type': 'date'},
                                      yaxis={"title": "Price (USD)"})}
        return figure

    @app.callback(Output('volume', 'figure'),
                  [Input('my-dropdown2', 'value')])
    def update_graph(selected_dropdown_value):
        dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
        trace1 = []
        for stock in selected_dropdown_value:
            trace1.append(
                go.Scatter(x=df[df["Stock"] == stock]["Date"],
                           y=df[df["Stock"] == stock]["Volume"],
                           mode='lines', opacity=0.7,
                           name=f'Volume {dropdown[stock]}', textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                                '#FF7400', '#FFF400', '#FF0056'],
                                      height=600,
                                      title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
                                      xaxis={"title": "Date",
                                             'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                                 'step': 'month',
                                                                                 'stepmode': 'backward'},
                                                                                {'count': 6, 'label': '6M',
                                                                                 'step': 'month',
                                                                                 'stepmode': 'backward'},
                                                                                {'step': 'all'}])},
                                             'rangeslider': {'visible': True}, 'type': 'date'},
                                      yaxis={"title": "Transactions Volume"})}
        return figure

app = dash.Dash()
server = app.server
if __name__=='__main__':
    main()
    app.run_server(debug=True)