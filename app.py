from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import io
import base64
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

app = Flask(__name__)
model = load_model('stock.keras')

@app.route('/', methods=['GET', 'POST'])
def index():
    ticker = "SBIN.BO"  # Default ticker
    plots = []
    metrics = {}

    if request.method == 'POST':
        ticker = request.form.get('ticker', ticker)

    # Fetch stock data
    df = yf.download(ticker, start='2010-01-01', end=pd.to_datetime('today').strftime('%Y-%m-%d'))

    if df.empty:
        return render_template('index.html', error="Invalid stock ticker or no data available.")

    # Plot 1: Closing Time vs Time chart
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Close Price')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Closing Time vs Time")
    plt.grid(True)
    plots.append(figure_to_base64(plt))

    # Plot 2: Closing Time with 100MA
    ma100 = df['Close'].rolling(100).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(ma100, 'r', label='100MA')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Closing Time with 100MA")
    plt.legend()
    plt.grid(True)
    plots.append(figure_to_base64(plt))

    # Plot 3: Closing Time with 100MA and 200MA
    ma200 = df['Close'].rolling(200).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(ma100, 'r', label='100MA')
    plt.plot(ma200, 'g', label='200MA')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Closing Time with 100MA and 200MA")
    plt.legend()
    plt.grid(True)
    plots.append(figure_to_base64(plt))

    # Train-test split
    training_data = pd.DataFrame(df['Close'][:int(len(df) * 0.70)])
    testing_data = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(training_data)

    # Prepare test data
    past_100_days = training_data.tail(100)
    final_df = pd.concat([past_100_days, testing_data], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    xtest = []
    ytest = []
    for i in range(100, input_data.shape[0]):
        xtest.append(input_data[i - 100:i])
        ytest.append(input_data[i, 0])

    xtest, ytest = np.array(xtest), np.array(ytest)

    # Predictions
    ypred = model.predict(xtest)
    scale_factor = 1 / scaler.scale_[0]
    ypred = ypred * scale_factor
    ytest = ytest * scale_factor

    # Plot 4: Predictions vs Original
    plt.figure(figsize=(12, 6))
    plt.plot(ytest, 'b', label='Original Price')
    plt.plot(ypred, 'r', label='Predicted Price')
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.title("Predictions vs Original")
    plt.legend()
    plt.grid(True)
    plots.append(figure_to_base64(plt))

    # Metrics
    metrics['r2_score'] = r2_score(ytest, ypred)
    metrics['mse'] = mean_squared_error(ytest, ypred)

    return render_template('index.html', plots=plots, metrics=metrics, ticker=ticker)

def figure_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    return plot_url

if __name__ == '__main__':
    app.run(debug=True)
