from flask import Flask, request, jsonify, render_template_string
import yfinance as yf
import pandas as pd 
import numpy as np

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Trading-AI</title>
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #0b1020;
            color: white;
        }
        .box {
            max-width: 700px;
            margin: 40px auto;
            padding: 25px;
        }
        h1 {
            text-align: center;
            font-size: 34px;
        }
        .card {
            background: #151c31;
            border-radius: 16px;
            padding: 20px;
            margin-top: 20px;
        }
        input, select, button {
            width: 100%;
            box-sizing: border-box;
            padding: 14px;
            margin-top: 10px;
            border-radius: 10px;
            border: 1px solid #34405e;
            background: #0d1426;
            color: white;
            font-size: 16px;
        }
        button {
            background: #2563eb;
            border: none;
            font-weight: bold;
            cursor: pointer;
        }
        #result {
            line-height: 1.7;
        }
        .signal {
            font-size: 30px;
            font-weight: bold;
            text-align: center;
            padding: 15px;
        }
    </style>
</head>

<body>
<div class="box">
    <h1>Trading-AI</h1>

    <div class="card">
        <label>Market / Symbol</label>
        <input id="symbol" value="CL=F" placeholder="Example: CL=F">

        <label>Timeframe</label>
        <select id="interval">
            <option value="1m">1 Minute</option>
            <option value="5m" selected>5 Minutes</option>
            <option value="15m">15 Minutes</option>
            <option value="1h">1 Hour</option>
        </select>

        <button onclick="analyze()">ANALYZE MARKET</button>
    </div>

    <div class="card" id="result">
        Enter a symbol and press Analyze Market.
    </div>
</div>

<script>
async function analyze() {
    const symbol = document.getElementById("symbol").value;
    const interval = document.getElementById("interval").value;

    document.getElementById("result").innerHTML =
        "Analyzing market...";

    try {
        const response = await fetch(
            "/api/analyze?symbol=" +
            encodeURIComponent(symbol) +
            "&interval=" +
            encodeURIComponent(interval)
        );

        const data = await response.json();

        if (!response.ok) {
            document.getElementById("result").innerHTML =
                "<b>Error:</b> " + data.error;
            return;
        }

        document.getElementById("result").innerHTML = `
            <div class="signal">${data.signal}</div>
            <hr>
            <b>Symbol:</b> ${data.symbol}<br>
            <b>Price:</b> ${data.price}<br>
            <b>RSI:</b> ${data.rsi}<br>
            <b>Trend:</b> ${data.trend}<br>
            <b>Support:</b> ${data.support}<br>
            <b>Resistance:</b> ${data.resistance}<br>
            <b>Confidence:</b> ${data.confidence}%<br>
            <hr>
            <b>Analysis:</b><br>
            ${data.analysis}
        `;

    } catch (error) {
        document.getElementById("result").innerHTML =
            "<b>Connection error:</b> " + error;
    }
}
</script>
</body>
</html>
"""


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/api/analyze")
def analyze():
    symbol = request.args.get("symbol", "CL=F")
    interval = request.args.get("interval", "5m")

    allowed = ["1m", "5m", "15m", "1h"]

    if interval not in allowed:
        return jsonify({"error": "Invalid timeframe"}), 400

    try:
        period = "1d" if interval in ["1m", "5m", "15m"] else "5d"

        data = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False
        )

        if data.empty:
            return jsonify({
                "error": "Market data not available for this symbol."
            }), 404

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        close = data["Close"].dropna()

        if len(close) < 30:
            return jsonify({
                "error": "Not enough market data for analysis."
            }), 400

        price = float(close.iloc[-1])

        ema20 = float(close.ewm(span=20).mean().iloc[-1])
        ema50 = float(close.ewm(span=50).mean().iloc[-1])

        rsi_series = calculate_rsi(close)
        rsi = float(rsi_series.iloc[-1])

        recent = close.tail(30)

        support = float(recent.min())
        resistance = float(recent.max())

        bullish = 0
        bearish = 0

        if price > ema20:
            bullish += 1
        else:
            bearish += 1

        if ema20 > ema50:
            bullish += 1
        else:
            bearish += 1

        if rsi < 35:
            bullish += 1
        elif rsi > 65:
            bearish += 1

        if bullish >= 2 and bearish == 0:
            signal = "BUY"
        elif bearish >= 2 and bullish == 0:
            signal = "SELL"
        else:
            signal = "WAIT"

        confidence = int(
            min(95, 50 + abs(bullish - bearish) * 15)
        )

        if ema20 > ema50:
            trend = "Bullish"
        elif ema20 < ema50:
            trend = "Bearish"
        else:
            trend = "Sideways"

        analysis = (
            f"Price is {price:.2f}. "
            f"EMA20 is {ema20:.2f} and EMA50 is {ema50:.2f}. "
            f"RSI is {rsi:.1f}. "
            f"The current structure is {trend.lower()}. "
            "Use support/resistance confirmation before entering a trade."
        )

        return jsonify({
            "symbol": symbol,
            "price": round(price, 4),
            "rsi": round(rsi, 2),
            "trend": trend,
            "support": round(support, 4),
            "resistance": round(resistance, 4),
            "signal": signal,
            "confidence": confidence,
            "analysis": analysis
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000,
        debug=False
    )
