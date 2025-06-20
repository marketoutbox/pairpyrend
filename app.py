from flask import Flask, request, jsonify
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from flask_cors import CORS # Import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

@app.route('/adf-test', methods=['POST'])
def adf_test_api():
    data = request.get_json()
    if not data or 'time_series' not in data:
        return jsonify({"error": "Missing 'time_series' in request body"}), 400

    time_series_list = data['time_series']
    if not isinstance(time_series_list, list):
        return jsonify({"error": "'time_series' must be a list of numbers"}), 400

    # Convert list to pandas Series, dropping NaNs
    clean_series = pd.Series(time_series_list).dropna()

    if clean_series.empty:
        return jsonify({"error": "Input time series is empty after dropping NaN values."}), 400
    if len(clean_series) < 5: # adfuller typically needs at least 5 observations
        return jsonify({"error": f"Not enough observations ({len(clean_series)}) for ADF test. Minimum required is 5."}), 400

    try:
        # Perform the ADF test
        # autolag='AIC': Automatically selects the optimal number of lags based on AIC.
        # regression='c': Includes a constant (intercept) in the test regression.
        adf_result = adfuller(clean_series, autolag='AIC', regression='c')

        # Extract results
        test_statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4] # Dictionary of critical values

        # Determine stationarity based on p-value (common significance level 0.05)
        is_stationary = bool(p_value < 0.05) # Explicitly cast to Python bool

        return jsonify({
            "statistic": test_statistic,
            "pValue": p_value,
            "criticalValues": critical_values,
            "isStationary": is_stationary # This will now be a standard Python bool
        }), 200

    except Exception as e:
        app.logger.error(f"An error occurred during ADF test calculation: {e}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) # Use 0.0.0.0 for Render deployment
