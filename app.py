from flask import Flask, request, jsonify
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/adf-test', methods=['POST'])
def adf_test_api():
    data = request.get_json()
    if not data or 'time_series' not in data:
        return jsonify({"error": "Missing 'time_series' in request body"}), 400

    time_series_list = data['time_series']
    if not isinstance(time_series_list, list):
        return jsonify({"error": "'time_series' must be a list of numbers"}), 400

    clean_series = pd.Series(time_series_list).dropna()

    if clean_series.empty:
        return jsonify({"error": "Input time series is empty after dropping NaN values."}), 400
    if len(clean_series) < 5:
        return jsonify({"error": f"Not enough observations ({len(clean_series)}) for ADF test. Minimum required is 5."}), 400

    try:
        adf_result = adfuller(clean_series, autolag='AIC', regression='c')

        # Explicitly convert all NumPy types to standard Python types
        test_statistic = float(adf_result[0])
        p_value = float(adf_result[1])

        # Convert critical_values dictionary values to standard floats
        critical_values = {k: float(v) for k, v in adf_result[4].items()}

        is_stationary = bool(p_value < 0.05) # Ensure it's a standard Python bool

        return jsonify({
            "statistic": test_statistic,
            "pValue": p_value,
            "criticalValues": critical_values,
            "isStationary": is_stationary
        }), 200

    except Exception as e:
        # Log the full exception for better debugging on Render
        app.logger.error(f"An error occurred during ADF test calculation: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
