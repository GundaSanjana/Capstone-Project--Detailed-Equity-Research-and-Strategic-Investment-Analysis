import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set Streamlit Page Configuration
st.set_page_config(page_title="Detailed Equity Research and Strategic Investment Analysis", layout="wide")

# App Title
st.title("Detailed Equity Research and Strategic Investment Analysis App")

# First Input: Stock Ticker
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker")  # Ticker input first, no default

# If ticker is provided, show the rest of the inputs
if ticker:
    # Ask the user for the number of days to predict in the future first
    future_days_input = st.sidebar.number_input("Enter number of days to Predict from today in Future", min_value=1, max_value=365, value=10)  # User input for future prediction

    # Other Inputs: Start and End Date
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.now())

    # Fetch Data
    @st.cache_data
    def fetch_data(ticker, start_date, end_date):
        return yf.download(ticker, start=start_date, end=end_date)

    # Fetch the data after ticker is provided
    data = fetch_data(ticker, start_date, end_date)

    if data.empty:
        st.error("No data found. Check the stock ticker or date range.")
    else:
        # Display the head of the data (first few rows)
        st.subheader(f"Head of {ticker} Data")
        st.write(data.head())  # Display the first 5 rows of the stock data

        # Data Preparation
        data['Date'] = data.index
        data['Date'] = pd.to_datetime(data['Date'])
        data['Days'] = (data['Date'] - data['Date'].min()).dt.days
        X = data[['Days']]
        y = data['Close']

        # Set Polynomial Degree (fixed degree 4)
        degree = 4
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Model Evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        #st.sidebar.write(f"Mean Squared Error: {mse:.2f}")

        # **Step 1: Plot Historical Prices First**
        st.subheader("Historical Prices")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(data['Date'], data['Close'], label="Historical Prices", color='blue')
        ax1.set_title(f"{ticker} Stock Price History")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Close Price")
        ax1.legend()
        st.pyplot(fig1)

        # **Step 2: Future Price Prediction based on user input for future days**
        last_day = data['Days'].iloc[-1]
        future_day = last_day + future_days_input
        future_day_poly = poly.transform([[future_day]])
        future_price = model.predict(future_day_poly)[0]

        st.sidebar.write(f"Predicted Price in {future_days_input} days: ${float(future_price):.2f}")

        # Visualization for Future Price Prediction
        st.subheader("Future Price Prediction")
        future_date = data['Date'].iloc[-1] + pd.Timedelta(days=future_days_input)
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(data['Date'], data['Close'], label="Historical Prices", color='blue')
        ax3.scatter(future_date, future_price, color='red', label=f"Predicted Price on {future_date.date()} (${float(future_price):.2f})")
        ax3.set_title(f"{ticker} Future Price Prediction")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Close Price")
        ax3.legend()
        st.pyplot(fig3)

        # **Step 3: Predict Today's Price**
        today_day_poly = poly.transform([[last_day]])
        today_price = model.predict(today_day_poly)[0]

        # **Step 4: Predict Tomorrow's Price**
        tomorrow_day = last_day + 1
        tomorrow_day_poly = poly.transform([[tomorrow_day]])
        tomorrow_price = model.predict(tomorrow_day_poly)[0]

        # Display Predicted Prices for Today and Tomorrow in Sidebar
        st.sidebar.write(f"Predicted Price for Today: ${float(today_price):.2f}")
        st.sidebar.write(f"Predicted Price for Tomorrow: ${float(tomorrow_price):.2f}")

        # Visualization for Today's and Tomorrow's Price Predictions
        st.header("Today's and Tomorrow's Price Predictions")
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        # Plot Historical Prices
        ax2.plot(data['Date'], data['Close'], label="Historical Prices", color='blue')

        # Add Predicted Prices
        today_date = data['Date'].iloc[-1]  # Date corresponding to today
        tomorrow_date = today_date + pd.Timedelta(days=1)  # Date corresponding to tomorrow
        ax2.scatter(today_date, today_price, color='orange', label=f"Predicted Price for Today (${float(today_price):.2f})")
        ax2.scatter(tomorrow_date, tomorrow_price, color='purple', label=f"Predicted Price for Tomorrow (${float(tomorrow_price):.2f})")

        # Customize Graph
        ax2.set_title(f"{ticker} Predicted Prices for Today and Tomorrow")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Close Price")
        ax2.legend()

        # Display the Graph
        st.pyplot(fig2)

        # **Graph 3: Predict Future Prices with Dates**
        future_days_input_for_dates = future_days_input  # Use the same input value from the sidebar

        future_day_numbers_for_dates = np.arange(last_day + 1, last_day + future_days_input_for_dates + 1)
        future_day_numbers_for_dates_poly = poly.transform(future_day_numbers_for_dates.reshape(-1, 1))

        future_prices_for_dates = model.predict(future_day_numbers_for_dates_poly)
        future_dates_for_predictions = pd.date_range(start=today_date, periods=future_days_input_for_dates, freq='D')

        fig4, ax4 = plt.subplots(figsize=(14, 7))
        ax4.plot(future_dates_for_predictions, future_prices_for_dates, label=f"Predicted Prices for Next {future_days_input_for_dates} Days", color='purple', linestyle='-.')

        ax4.set_title(f"Future Predicted Prices for {ticker} with Dates")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Predicted Close Price")
        ax4.legend()
        ax4.grid(True)
        st.pyplot(fig4)
        
        # Graph 4: Predict Past and Future Prices with Dates (n days in past and future)
        n_days_input_for_dates = st.sidebar.number_input("Enter the number of days (both past and future) to predict:", min_value=1, max_value=365, value=5)

        # Predict past days (before today)
        past_day_numbers_for_dates = np.arange(last_day - n_days_input_for_dates, last_day)
        past_day_numbers_for_dates_poly = poly.transform(past_day_numbers_for_dates.reshape(-1, 1))
        past_prices_for_dates = model.predict(past_day_numbers_for_dates_poly)

        # Predict future days (after today)
        future_day_numbers_for_dates = np.arange(last_day + 1, last_day + n_days_input_for_dates + 1)
        future_day_numbers_for_dates_poly = poly.transform(future_day_numbers_for_dates.reshape(-1, 1))
        future_prices_for_dates = model.predict(future_day_numbers_for_dates_poly)

        # Combine past and future predictions (excluding today's price)
        all_day_numbers_for_predictions = np.concatenate([past_day_numbers_for_dates, future_day_numbers_for_dates])

        # Concatenate the past and future prices only (no need to include today's price)
        all_prices_for_predictions = np.concatenate([past_prices_for_dates, future_prices_for_dates])

        # Combine the dates for visualization (for both past and future)
        all_dates_for_predictions = pd.date_range(start=today_date - pd.Timedelta(days=n_days_input_for_dates), periods=(2 * n_days_input_for_dates), freq='D')

        # Create a graph with predicted past and future prices and their corresponding dates
        fig5, ax5 = plt.subplots(figsize=(14, 7))

        # Plot the continuous line connecting past and future prices
        ax5.plot(all_dates_for_predictions, all_prices_for_predictions, label=f"Predicted Prices (Past + Future)", color='purple', linestyle='-')

        # Add title and labels
        ax5.set_title(f"Predicted Prices for {ticker} (Past and Future {n_days_input_for_dates} Days)")
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Predicted Close Price")

        # Add grid and legend
        ax5.legend()
        ax5.grid(True)

        # Add grid and legend
        ax5.legend()
        ax5.grid(True)

        # Rotate x-axis labels for better readability and display each date
        ax5.set_xticklabels(all_dates_for_predictions.date, rotation=45)  # .date will exclude time
        st.pyplot(fig5)


        # **Graph 5: Actual vs Predicted Prices (on Test Data)**
        fig6, ax6 = plt.subplots(figsize=(14, 7))
        ax6.plot(y_test.index, y_test, label="Actual Prices", color='blue')
        ax6.plot(y_test.index, y_pred, label="Predicted Prices", color='red', linestyle='--')

        ax6.set_title(f"Actual vs Predicted Prices for {ticker}")
        ax6.set_xlabel("Date")
        ax6.set_ylabel("Close Price")
        ax6.legend()
        ax6.grid(True)
        st.pyplot(fig6)
