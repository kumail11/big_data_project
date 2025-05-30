from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    # Load the DataFrame (you can load directly instead of CSV if you embed it)
    df = pd.read_csv('analysis_data.csv')

    # Clean NaN for display
    df = df.fillna('N/A')

    # Convert to list of dicts for rendering
    records = df.to_dict(orient='records')

    return render_template('index.html', data=records)

if __name__ == '__main__':
    app.run(debug=True)