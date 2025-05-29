from flask import Flask, render_template
import plotly.express as px
import pandas as pd
import pickle
import os
from src.utils.logging import setup_logger

app = Flask(__name__)
logger = setup_logger('dashboard', 'logs/dashboard.log')

@app.route('/')
def dashboard():
    logger.info('Loading metrics for dashboard')
    try:
        with open('logs/metrics_history.pkl', 'rb') as f:
            metrics_history = pickle.load(f)
        metrics_df = pd.DataFrame([
            {'Round': i + 1, 'Accuracy': m['sparse_categorical_accuracy'], 'Loss': m['loss']}
            for i, m in enumerate(metrics_history)
        ])
        fig = px.line(metrics_df, x='Round', y=['Accuracy', 'Loss'], title='Federated Learning Metrics')
        graph_html = fig.to_html(full_html=False)
        logger.info('Dashboard rendered successfully')
        return render_template('dashboard.html', graph_html=graph_html)
    except Exception as e:
        logger.error(f'Error rendering dashboard: {str(e)}')
        return f'Error: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)