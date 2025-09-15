import os
import sys
import time
import json
import sqlite3
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Data APIs
import yfinance as yf
import requests

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates

# Web Dashboard
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConfigManager:
    """Configuration management for the analytics dashboard"""
    
    def __init__(self):
        self.config = {
            'database': {
                'name': 'stock_analytics.db',
                'tables': {
                    'stock_prices': 'stock_prices',
                    'predictions': 'predictions',
                    'alerts': 'alerts'
                }
            },
            'stocks': {
                'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA'],
                'update_interval': 300,  # 5 minutes
                'prediction_horizon': 5  # 5 days
            },
            'ml_models': {
                'retrain_interval': 86400,  # 24 hours
                'min_training_days': 60,
                'features': ['sma_5', 'sma_20', 'rsi', 'volatility', 'volume_ratio']
            }
        }
    
    def get(self, key_path: str):
        """Get configuration value by dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self, db_name: str = 'stock_analytics.db'):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Stock prices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                sma_5 REAL,
                s ma_20 REAL,
                rsi REAL,
                volatility REAL,
                volume_ratio REAL
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                prediction_date DATETIME NOT NULL,
                target_date DATETIME NOT NULL,
                predicted_price REAL,
                actual_price REAL,
                model_name TEXT,
                confidence REAL
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                alert_type TEXT,
                message TEXT,
                value REAL,
                threshold REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        logging.info("Database initialized successfully")
    
    def insert_stock_data(self, data: pd.DataFrame, symbol: str):
        """Insert stock data with technical indicators"""
        conn = sqlite3.connect(self.db_name)
        
        # Reset index and prepare data for insertion
        data_copy = data.copy()
        data_copy.reset_index(inplace=True)
        data_copy['symbol'] = symbol
        
        # Rename columns to match database
        data_copy.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Date': 'timestamp'
        }, inplace=True)
        
        # Ensure all required columns exist
        for col in ['open', 'high', 'low', 'close', 'volume', 'sma_5', 'sma_20', 'rsi', 'volatility', 'volume_ratio']:
            if col not in data_copy.columns:
                data_copy[col] = None
        
        # Insert data
        data_copy[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                  'sma_5', 'sma_20', 'rsi', 'volatility', 'volume_ratio']].to_sql(
            'stock_prices', conn, if_exists='append', index=False
        )
        
        conn.close()
    
    def get_stock_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Retrieve stock data from database"""
        conn = sqlite3.connect(self.db_name)
        query = '''
            SELECT * FROM stock_prices 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(symbol, days))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def insert_prediction(self, symbol: str, target_date: datetime, 
                         predicted_price: float, model_name: str, confidence: float):
        """Insert prediction data"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (symbol, prediction_date, target_date, predicted_price, model_name, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (symbol, datetime.now(), target_date, predicted_price, model_name, confidence))
        conn.commit()
        conn.close()

class DataCollector:
    """Collects real-time stock data from various APIs"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df_copy = df.copy()
        
        # Simple Moving Averages
        df_copy['sma_5'] = df_copy['Close'].rolling(window=5).mean()
        df_copy['sma_20'] = df_copy['Close'].rolling(window=20).mean()
        
        # RSI
        delta = df_copy['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_copy['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility (20-day rolling std)
        df_copy['volatility'] = df_copy['Close'].rolling(window=20).std()
        
        # Volume ratio
        df_copy['volume_ratio'] = df_copy['Volume'] / df_copy['Volume'].rolling(window=20).mean()
        
        return df_copy
    
    def fetch_stock_data(self, symbol: str, period: str = '1mo') -> pd.DataFrame:
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                logging.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Store in database
            self.db_manager.insert_stock_data(df, symbol)
            
            logging.info(f"Successfully fetched data for {symbol}")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_stocks(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks"""
        stock_data = {}
        for symbol in symbols:
            data = self.fetch_stock_data(symbol)
            if not data.empty:
                stock_data[symbol] = data
            time.sleep(1)  # Rate limiting
        return stock_data

class MLPredictor:
    """Machine Learning models for stock price prediction"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.models = {}
        self.scalers = {}
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for ML models"""
        # Feature columns
        feature_cols = ['sma_5', 'sma_20', 'rsi', 'volatility', 'volume_ratio']
        
        # Check if required columns exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logging.warning(f"Missing columns: {missing_cols}")
            return pd.DataFrame(), pd.Series()
        
        # Drop rows with NaN values
        df_clean = df.dropna(subset=feature_cols + ['close'])
        
        if len(df_clean) < 30:
            return pd.DataFrame(), pd.Series()
        
        X = df_clean[feature_cols]
        y = df_clean['close'].shift(-1).dropna()  # Next day's closing price
        
        # Align X and y
        X = X.iloc[:-1]
        
        return X, y
    
    def train_model(self, symbol: str, model_type: str = 'random_forest') -> Dict:
        """Train ML model for a specific symbol"""
        # Get historical data
        df = self.db_manager.get_stock_data(symbol, days=100)
        
        if df.empty:
            return {'status': 'error', 'message': 'No data available'}
        
        X, y = self.prepare_features(df)
        
        if X.empty or len(X) < 30:
            return {'status': 'error', 'message': 'Insufficient data for training'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Store model and scaler
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        
        return {
            'status': 'success',
            'mae': mae,
            'rmse': rmse,
            'model_type': model_type,
            'training_samples': len(X_train)
        }
    
    def predict_price(self, symbol: str, days_ahead: int = 1) -> Dict:
        """Predict stock price"""
        if symbol not in self.models:
            return {'status': 'error', 'message': 'Model not trained for this symbol'}
        
        # Get recent data
        df = self.db_manager.get_stock_data(symbol, days=30)
        
        if df.empty:
            return {'status': 'error', 'message': 'No recent data available'}
        
        # Prepare features
        feature_cols = ['sma_5', 'sma_20', 'rsi', 'volatility', 'volume_ratio']
        
        # Check if columns exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            return {'status': 'error', 'message': f'Missing features: {missing_cols}'}
        
        latest_features = df[feature_cols].iloc[-1:].dropna()
        
        if latest_features.empty:
            return {'status': 'error', 'message': 'Cannot calculate features'}
        
        # Scale and predict
        features_scaled = self.scalers[symbol].transform(latest_features)
        prediction = self.models[symbol].predict(features_scaled)[0]
        
        # Calculate confidence (simplified)
        recent_volatility = df['volatility'].iloc[-1] if not pd.isna(df['volatility'].iloc[-1]) else 0.02
        confidence = max(0.1, min(0.9, 1 - (recent_volatility / df['close'].iloc[-1])))
        
        return {
            'status': 'success',
            'predicted_price': prediction,
            'confidence': confidence,
            'current_price': df['close'].iloc[-1],
            'prediction_date': datetime.now() + timedelta(days=days_ahead)
        }

class AlertSystem:
    """Alert system for significant market events"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def check_price_alerts(self, symbol: str, current_price: float, 
                          previous_price: float) -> List[Dict]:
        """Check for price-based alerts"""
        alerts = []
        
        if previous_price == 0:
            return alerts
        
        # Significant price movement (>5%)
        price_change_pct = abs(current_price - previous_price) / previous_price
        if price_change_pct > 0.05:
            direction = "surge" if current_price > previous_price else "drop"
            
            alerts.append({
                'type': 'price_movement',
                'message': f'{symbol} price {direction} of {price_change_pct:.1%}',
                'value': current_price,
                'threshold': 0.05
            })
        
        return alerts
    
    def check_volume_alerts(self, symbol: str, current_volume: int, 
                           avg_volume: float) -> List[Dict]:
        """Check for volume-based alerts"""
        alerts = []
        
        if avg_volume == 0:
            return alerts
        
        # Unusual volume (>2x average)
        if current_volume > avg_volume * 2:
            ratio = current_volume / avg_volume
            alerts.append({
                'type': 'high_volume',
                'message': f'{symbol} volume {ratio:.1f}x above average',
                'value': current_volume,
                'threshold': avg_volume * 2
            })
        
        return alerts

class DashboardApp:
    """Streamlit dashboard application"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.db_manager = DatabaseManager()
        self.data_collector = DataCollector(self.db_manager)
        self.ml_predictor = MLPredictor(self.db_manager)
        self.alert_system = AlertSystem(self.db_manager)
    
    def run_data_collection(self):
        """Background data collection"""
        symbols = self.config.get('stocks.symbols')
        
        while True:
            try:
                logging.info("Starting data collection cycle...")
                stock_data = self.data_collector.fetch_multiple_stocks(symbols)
                
                # Train/update models
                for symbol in stock_data.keys():
                    result = self.ml_predictor.train_model(symbol)
                    logging.info(f"Model training for {symbol}: {result}")
                
                logging.info("Data collection cycle completed")
                time.sleep(self.config.get('stocks.update_interval'))
                
            except Exception as e:
                logging.error(f"Error in data collection: {str(e)}")
                time.sleep(60)  # Wait before retrying
    
    def create_price_chart(self, symbol: str) -> go.Figure:
        """Create interactive price chart"""
        df = self.db_manager.get_stock_data(symbol, days=30)
        
        if df.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        ))
        
        # Moving averages
        if 'sma_5' in df.columns and not pd.isna(df['sma_5'].iloc[-1]):
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['sma_5'],
                name='SMA 5',
                line=dict(color='orange', width=1)
            ))
        
        if 'sma_20' in df.columns and not pd.isna(df['sma_20'].iloc[-1]):
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['sma_20'],
                name='SMA 20',
                line=dict(color='blue', width=1)
            ))
        
        fig.update_layout(
            title=f'{symbol} Price Chart',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def create_prediction_chart(self, symbol: str) -> go.Figure:
        """Create prediction visualization"""
        # Get historical data
        df = self.db_manager.get_stock_data(symbol, days=30)
        
        if df.empty:
            return go.Figure()
        
        # Get prediction
        prediction = self.ml_predictor.predict_price(symbol)
        
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='white')
        ))
        
        if prediction['status'] == 'success':
            # Prediction point
            pred_date = prediction['prediction_date']
            pred_price = prediction['predicted_price']
            
            fig.add_trace(go.Scatter(
                x=[pred_date],
                y=[pred_price],
                mode='markers',
                name='Prediction',
                marker=dict(color='red', size=10)
            ))
            
            # Confidence interval
            confidence = prediction['confidence']
            error_margin = pred_price * (1 - confidence)
            
            fig.add_trace(go.Scatter(
                x=[pred_date, pred_date],
                y=[pred_price - error_margin, pred_price + error_margin],
                mode='lines',
                name='Confidence Interval',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title=f'{symbol} Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def create_technical_indicators_chart(self, symbol: str) -> go.Figure:
        """Create technical indicators chart"""
        df = self.db_manager.get_stock_data(symbol, days=30)
        
        if df.empty:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['RSI', 'Volume Ratio', 'Volatility', 'Moving Averages'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi'], name='RSI'),
                row=1, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        
        # Volume ratio
        if 'volume_ratio' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['volume_ratio'], name='Volume Ratio'),
                row=1, col=2
            )
            fig.add_hline(y=1, line_dash="dash", line_color="white", row=1, col=2)
        
        # Volatility
        if 'volatility' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['volatility'], name='Volatility'),
                row=2, col=1
            )
        
        # Moving averages comparison
        if 'sma_5' in df.columns and 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['sma_5'], name='SMA 5'),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20'),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f'{symbol} Technical Indicators',
            template='plotly_dark',
            height=600
        )
        
        return fig

# Streamlit Dashboard Interface
def run_streamlit_dashboard():
    """Main Streamlit dashboard"""
    st.set_page_config(
        page_title="Stock Market Analytics Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize dashboard
    dashboard = DashboardApp()
    
    # Initialize session state
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    # Sidebar
    st.sidebar.title("🏦 Stock Analytics")
    st.sidebar.markdown("---")
    
    # Stock selection
    symbols = dashboard.config.get('stocks.symbols')
    selected_symbol = st.sidebar.selectbox("Select Stock", symbols, key='symbol_select')
    
    # Dashboard controls
    st.sidebar.markdown("### Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False, key='auto_refresh')
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60, key='refresh_interval')
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data", key='refresh_button'):
        with st.spinner("Fetching data..."):
            dashboard.data_collector.fetch_stock_data(selected_symbol)
            st.session_state.refresh_counter += 1
            st.session_state.last_refresh = time.time()
        st.sidebar.success("Data refreshed!")
    
    # Auto-refresh logic
    if auto_refresh:
        current_time = time.time()
        if current_time - st.session_state.last_refresh > refresh_interval:
            st.session_state.refresh_counter += 1
            st.session_state.last_refresh = current_time
            st.rerun()
    
    # Main dashboard
    st.title(f"📈 {selected_symbol} Stock Analytics Dashboard")
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    # Get current data
    current_data = dashboard.db_manager.get_stock_data(selected_symbol, days=2)
    
    if not current_data.empty and len(current_data) >= 2:
        current_price = current_data['close'].iloc[-1]
        prev_price = current_data['close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}")
        
        with col2:
            st.metric("Change %", f"{price_change_pct:.2f}%")
        
        with col3:
            volume = current_data['volume'].iloc[-1] if 'volume' in current_data.columns else 0
            st.metric("Volume", f"{volume:,}")
        
        with col4:
            volatility = current_data['volatility'].iloc[-1] if 'volatility' in current_data.columns else 0
            st.metric("Volatility", f"{volatility:.4f}")
    else:
        st.warning("Insufficient data for metrics. Please refresh data.")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Price Chart")
        price_chart = dashboard.create_price_chart(selected_symbol)
        st.plotly_chart(price_chart, use_container_width=True, key=f'price_chart_{st.session_state.refresh_counter}')
    
    with col2:
        st.subheader("🔮 Price Prediction")
        prediction_chart = dashboard.create_prediction_chart(selected_symbol)
        st.plotly_chart(prediction_chart, use_container_width=True, key=f'prediction_chart_{st.session_state.refresh_counter}')
    
    # Technical indicators
    st.subheader("📈 Technical Indicators")
    tech_chart = dashboard.create_technical_indicators_chart(selected_symbol)
    st.plotly_chart(tech_chart, use_container_width=True, key=f'tech_chart_{st.session_state.refresh_counter}')
    
    # ML Model Performance
    st.subheader("🤖 ML Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Train Model", key='train_button'):
            with st.spinner("Training model..."):
                result = dashboard.ml_predictor.train_model(selected_symbol)
                if result['status'] == 'success':
                    st.success("Model trained successfully!")
                    st.write(f"MAE: ${result['mae']:.2f}")
                    st.write(f"RMSE: ${result['rmse']:.2f}")
                    st.write(f"Training samples: {result['training_samples']}")
                else:
                    st.error(result['message'])
    
    with col2:
        if st.button("Get Prediction", key='predict_button'):
            prediction = dashboard.ml_predictor.predict_price(selected_symbol)
            if prediction['status'] == 'success':
                st.success("Prediction generated!")
                st.write(f"Predicted Price: ${prediction['predicted_price']:.2f}")
                st.write(f"Confidence: {prediction['confidence']:.2%}")
                st.write(f"Current Price: ${prediction['current_price']:.2f}")
                st.write(f"Prediction Date: {prediction['prediction_date'].strftime('%Y-%m-%d')}")
            else:
                st.error(prediction['message'])

# Main execution
if __name__ == "__main__":
    # Setup logging
    logging.info("Starting Stock Market Analytics Dashboard")
    
    # Check if running in Streamlit
    try:
        import streamlit as st
        # Start background data collection in a separate thread
        dashboard = DashboardApp()
        
        # Start data collection thread
        data_thread = threading.Thread(target=dashboard.run_data_collection, daemon=True)
        data_thread.start()
        
        # Run Streamlit dashboard
        run_streamlit_dashboard()
        
    except ImportError:
        # Run as standalone script
        print("Streamlit not available. Running data collection only...")
        dashboard = DashboardApp()
        
        # Collect initial data
        symbols = dashboard.config.get('stocks.symbols')
        stock_data = dashboard.data_collector.fetch_multiple_stocks(symbols)
        
        # Train models
        for symbol in stock_data.keys():
            result = dashboard.ml_predictor.train_model(symbol)
            print(f"Model training for {symbol}: {result}")
            
            # Get prediction
            prediction = dashboard.ml_predictor.predict_price(symbol)
            print(f"Prediction for {symbol}: {prediction}")
        
        print("✅ Stock Market Analytics Dashboard completed!")
        print("\nTo run the web dashboard, install Streamlit:")
        print("pip install streamlit")
        print("Then run: streamlit run this_script.py")