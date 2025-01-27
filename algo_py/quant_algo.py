import os
import time
import pyotp
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
from SmartApi import SmartConnect
from functools import wraps
from nsepython import nse_optionchain_scrapper
import json
import sys
import os

# Import algo_ai functionality
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Rate limiting decorator
def rate_limited(max_per_second):
    min_interval = 1.0 / float(max_per_second)
    def decorator(func):
        last_time_called = [0.0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_time_called[0] = time.time()
            return ret
        return wrapper
    return decorator

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_portfolio_metrics(trades):
    """Calculate portfolio performance metrics"""
    metrics = {
        'Total Trades': len(trades),
        'Closed Trades': len([t for t in trades if t.status == "CLOSED"]),
        'Open Trades': len([t for t in trades if t.status == "OPEN"]),
        'Total P&L': sum(t.pnl for t in trades if t.status == "CLOSED"),
        'Best Trade': max((t.pnl for t in trades if t.status == "CLOSED"), default=0),
        'Worst Trade': min((t.pnl for t in trades if t.status == "CLOSED"), default=0),
        'Win Rate': 0,
        'Average P&L': 0,
        'Average Duration': 0
    }
    
    closed_trades = [t for t in trades if t.status == "CLOSED"]
    if closed_trades:
        winning_trades = len([t for t in closed_trades if t.pnl > 0])
        metrics['Win Rate'] = (winning_trades / len(closed_trades)) * 100
        metrics['Average P&L'] = metrics['Total P&L'] / len(closed_trades)
        metrics['Average Duration'] = sum((t.exit_time - t.entry_time).total_seconds() for t in closed_trades) / len(closed_trades)
    
    return metrics

def initialize_angel_broker():
    try:
        api_key = os.getenv('ANGEL_API_KEY')
        client_id = os.getenv('ANGEL_CLIENT_ID')
        password = os.getenv('ANGEL_PASSWORD')
        totp_key = os.getenv('ANGEL_TOTP_KEY')
        
        smart_api = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(totp_key)
        current_totp = totp.now()
        
        login_response = smart_api.generateSession(client_id, password, current_totp)
        
        if login_response.get('status'):
            refresh_token = login_response['data']['refreshToken']
            smart_api.generateToken(refresh_token)
            profile = smart_api.getProfile(refresh_token)
            
            if profile.get('status'):
                st.success(f"Login successful for client: {client_id}")
                return smart_api
        
        st.error("Login failed. Please check your credentials.")
        return None
            
    except Exception as e:
        st.error(f"Error connecting to AngelOne: {e}")
        return None


@rate_limited(1)
def fetch_option_chain(symbol):
    """Fetch option chain data for a given symbol using nsepython."""
    try:
        # Fetch option chain data
        option_chain_data = nse_optionchain_scrapper(symbol)
        
        # Extract relevant data
        ce_data = option_chain_data['records']['data']
        pe_data = option_chain_data['records']['data']
        
        # Convert to DataFrame for display
        ce_df = pd.DataFrame([item['CE'] for item in ce_data if 'CE' in item])
        pe_df = pd.DataFrame([item['PE'] for item in pe_data if 'PE' in item])
        
        return ce_df, pe_df
    except Exception as e:
        st.error(f"Error fetching option chain data: {e}")
        return None, None

def main():
    st.set_page_config(page_title="Trading Dashboard", layout="wide")
    st.title("Trading Dashboard")

    # Create tabs
    tabs = st.tabs(["Index Analysis", "Options Chain", "Portfolio", "Settings"])

    # Index Analysis Tab
    with tabs[0]:
        st.header("Index Analysis")
        # Add your index analysis code here
        st.write("Index analysis content goes here.")

    # Options Chain Tab
    with tabs[1]:
        st.header("Options Chain")
        symbol = st.text_input("Enter the symbol (e.g., NIFTY, BANKNIFTY):", "NIFTY")

        if st.button("Fetch Option Chain"):
            ce_df, pe_df = fetch_option_chain(symbol)

            if ce_df is not None and pe_df is not None:
                st.subheader("Call Options")
                st.dataframe(ce_df.style.format({
                    'strikePrice': '₹{:.2f}',
                    'lastPrice': '₹{:.2f}',
                    'change': '{:.2f}%',
                    'totalTradedVolume': '{:,.0f}',
                    'openInterest': '{:,.0f}'
                }))

                st.subheader("Put Options")
                st.dataframe(pe_df.style.format({
                    'strikePrice': '₹{:.2f}',
                    'lastPrice': '₹{:.2f}',
                    'change': '{:.2f}%',
                    'totalTradedVolume': '{:,.0f}',
                    'openInterest': '{:,.0f}'
                }))

    # Portfolio Tab
    with tabs[2]:
        st.header("Portfolio")
        if 'paper_trades' not in st.session_state:
            st.session_state.paper_trades = []
        
        # Portfolio Summary
        st.subheader("Portfolio Summary")
        col1, col2, col3 = st.columns(3)
        
        # Calculate daily P&L
        today = pd.Timestamp.now(tz='Asia/Kolkata').date()
        today_trades = [t for t in st.session_state.paper_trades 
                        if t.status == "CLOSED" and t.exit_time.date() == today]
        daily_pnl = sum(t.pnl for t in today_trades)
        
        with col1:
            st.metric("Today's P&L", f"₹{daily_pnl:,.2f}")
            
        with col2:
            total_pnl = sum(t.pnl for t in st.session_state.paper_trades if t.status == "CLOSED")
            st.metric("Total P&L", f"₹{total_pnl:,.2f}")
            
        with col3:
            open_positions = len([t for t in st.session_state.paper_trades if t.status == "OPEN"])
            st.metric("Open Positions", open_positions)
        
        # Performance Metrics
        st.subheader("Performance Metrics")
        metrics = calculate_portfolio_metrics(st.session_state.paper_trades)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Win Rate", f"{metrics.get('Win Rate', 0):.1f}%")
            st.metric("Total Trades", metrics.get('Total Trades', 0))
        with col2:
            st.metric("Average P&L", f"₹{metrics.get('Average P&L', 0):,.2f}")
            st.metric("Best Trade", f"₹{metrics.get('Best Trade', 0):,.2f}")
        with col3:
            avg_duration_mins = metrics.get('Average Duration', 0) / 60
            st.metric("Avg Trade Duration", f"{avg_duration_mins:.1f} mins")
            st.metric("Worst Trade", f"₹{metrics.get('Worst Trade', 0):,.2f}")
        with col4:
            st.metric("Closed Trades", metrics.get('Closed Trades', 0))
            st.metric("Open Trades", metrics.get('Open Trades', 0))
        
        # Trade History
        st.subheader("Trade History")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            date_filter = st.date_input(
                "Date Range",
                value=(today, today),
                key='portfolio_date_filter'
            )
        with col2:
            status_filter = st.multiselect(
                "Status",
                options=["OPEN", "CLOSED"],
                default=["OPEN", "CLOSED"],
                key='portfolio_status_filter'
            )
        with col3:
            type_filter = st.multiselect(
                "Trade Type",
                options=["BUY", "SELL"],
                default=["BUY", "SELL"],
                key='portfolio_type_filter'
            )
        
        # Create trade history dataframe
        trade_history = []
        for trade in st.session_state.paper_trades:
            trade_dict = {
                'Entry Time': trade.entry_time,
                'Exit Time': trade.exit_time if trade.status == "CLOSED" else None,
                'Symbol': trade.option_symbol,
                'Type': trade.trade_type,
                'Quantity': trade.quantity,
                'Entry Price': trade.entry_price,
                'Exit Price': trade.exit_price,
                'P&L': trade.pnl,
                'Status': trade.status,
                'Exit Reason': trade.exit_reason
            }
            trade_history.append(trade_dict)
        
        trade_df = pd.DataFrame(trade_history)
        if not trade_df.empty:
            # Apply filters
            if date_filter:
                trade_df = trade_df[
                    (trade_df['Entry Time'].dt.date >= date_filter[0]) &
                    (trade_df['Entry Time'].dt.date <= date_filter[1])
                ]
            if status_filter:
                trade_df = trade_df[trade_df['Status'].isin(status_filter)]
            if type_filter:
                trade_df = trade_df[trade_df['Type'].isin(type_filter)]
            
            # Display trade history
            st.dataframe(trade_df, use_container_width=True)
        
        # Performance Charts
        st.subheader("Performance Analysis")
        
        if not trade_df.empty:
            tab1, tab2, tab3 = st.tabs(["P&L Analysis", "Trade Distribution", "Time Analysis"])
            
            with tab1:
                # Cumulative P&L chart
                closed_trades = trade_df[trade_df['Status'] == "CLOSED"].copy()
                if not closed_trades.empty:
                    closed_trades['Cumulative P&L'] = closed_trades['P&L'].cumsum()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=closed_trades['Exit Time'],
                        y=closed_trades['Cumulative P&L'],
                        mode='lines+markers',
                        name='Cumulative P&L'
                    ))
                    fig.update_layout(
                        title='Cumulative P&L Over Time',
                        xaxis_title='Date',
                        yaxis_title='P&L (₹)',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    # Win/Loss Distribution
                    win_loss = trade_df[trade_df['Status'] == "CLOSED"]['P&L'].apply(
                        lambda x: 'Win' if x > 0 else 'Loss'
                    ).value_counts()
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=win_loss.index,
                        values=win_loss.values,
                        hole=.3
                    )])
                    fig.update_layout(title='Win/Loss Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Trade Type Distribution
                    trade_types = trade_df['Type'].value_counts()
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=trade_types.index,
                        values=trade_types.values,
                        hole=.3
                    )])
                    fig.update_layout(title='Trade Type Distribution')
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Trade Duration Analysis
                closed_trades['Duration'] = (
                    closed_trades['Exit Time'] - closed_trades['Entry Time']
                ).dt.total_seconds() / 60  # Convert to minutes
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=closed_trades['Duration'],
                    nbinsx=20,
                    name='Trade Duration'
                ))
                fig.update_layout(
                    title='Trade Duration Distribution',
                    xaxis_title='Duration (minutes)',
                    yaxis_title='Number of Trades',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Export Options
        st.subheader("Export Data")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export to CSV", key='portfolio_export_csv'):
                if not trade_df.empty:
                    csv = trade_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"trade_history_{today}.csv",
                        mime="text/csv",
                        key='portfolio_download_csv'
                    )
        with col2:
            if st.button("Generate Report", key='portfolio_generate_report'):
                # Generate and display summary report
                report = f"""
                # Trading Performance Report
                
                ## Summary Metrics
                - Total P&L: ₹{total_pnl:,.2f}
                - Win Rate: {metrics.get('Win Rate', 0):.1f}%
                - Total Trades: {metrics.get('Total Trades', 0)}
                - Average P&L per Trade: ₹{metrics.get('Average P&L', 0):,.2f}
                
                ## Today's Performance
                - P&L: ₹{daily_pnl:,.2f}
                - Number of Trades: {len(today_trades)}
                
                ## Risk Metrics
                - Best Trade: ₹{metrics.get('Best Trade', 0):,.2f}
                - Worst Trade: ₹{metrics.get('Worst Trade', 0):,.2f}
                - Average Trade Duration: {avg_duration_mins:.1f} minutes
                """
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"trading_report_{today}.txt",
                    mime="text/plain",
                    key='portfolio_download_report'
                )

    # Settings Tab
    with tabs[3]:
        st.header("Settings")
        # Add your settings code here
        st.write("Settings content goes here.")

if __name__ == "__main__":
    main()

# from trading_strategy import TradingStrategy
from algo_ai import broker
import algo_ai
from strategy_manager import StrategyManager  # Import the StrategyManager

def main():
    # Custom CSS
    st.markdown("""
        <style>
        .main { padding: 0rem 1rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        .stTabs [data-baseweb="tab"] { padding: 10px 20px; background-color: #f0f2f6; }
        .stTabs [aria-selected="true"] { background-color: #4CAF50; color: white; }
        </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("📊 Quant Algo ")

    # Initialize broker connection
    if not broker.connect():
        st.stop()

    
    # Create tabs
    tabs = st.tabs([
        "🏠 Overview", 
        "💼 Portfolio", 
        "📈 Market Data", 
        "🔄 Orders", 
        "📊 Analysis",
        "🔍 Symbol Search",
        "📋 Option Chain",
        "⚙️ Settings",
        "🤖 Algo AI",  # New tab
        "📈 Strategy Manager"  # New tab
    ])

    # Overview Tab
    with tabs[0]:
        st.header("Account Overview")
        try:
            rms_response = broker.smart_api.rmsLimit()
            if rms_response['status']:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Available Balance", 
                             f"₹{float(rms_response['data'].get('net', 0)):,.2f}")
                with col2:
                    st.metric("Used Margin", 
                             f"₹{float(rms_response['data'].get('utilised', {}).get('grossUtilization', 0)):,.2f}")
                with col3:
                    st.metric("Available Margin", 
                             f"₹{float(rms_response['data'].get('net', 0)):,.2f}")
        except Exception as e:
            st.error("Error fetching account details")

    # Symbol Search Tab
    with tabs[5]:
        st.header("Symbol Search")
        
        col1, col2 = st.columns(2)
        with col1:
            exchange = st.selectbox(
                "Select Exchange",
                ["NSE", "BSE", "NFO", "MCX"],
                key="search_exchange"
            )
        with col2:
            search_query = st.text_input("Search Symbol", 
                placeholder="Enter name like RELIANCE, NIFTY, BANKNIFTY, etc.").upper()
            
        if search_query:
            try:
                search_result = broker.smart_api.searchScrip(exchange, search_query)
                if search_result['status']:
                    df = pd.DataFrame(search_result['data'])
                    if len(df) > 0:
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info(f"No symbols found matching '{search_query}' in {exchange}")
            except Exception as e:
                st.error(f"Error searching symbols: {e}")

        with st.expander("Search Tips"):
            st.markdown("""
            - For **Equity**, simply enter the company name or symbol (e.g., RELIANCE, TCS)
            - For **Index**, enter the index name (e.g., NIFTY, BANKNIFTY)
            - For **F&O**, search the underlying (e.g., RELIANCE for Reliance futures/options)
            - For **Commodities**, enter commodity name (e.g., GOLD, SILVER)
            """)

    # Option Chain Tab
    with tabs[6]:
        st.header("Option Chain")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            index = st.selectbox(
                "Select Index",
                ["NIFTY", "BANKNIFTY", "FINNIFTY"],
                key="option_index"
            )
        
        with col2:
            try:
                search_params = {"exchange": "NFO", "searchscrip": index}
                chain_response = fetch_option_chain(broker.smart_api, search_params)
                
                if chain_response['status']:
                    df = pd.DataFrame(chain_response['data'])
                    if 'expiry' in df.columns:
                        expiry_dates = sorted(pd.to_datetime(df['expiry'].unique()))
                        selected_expiry = st.selectbox(
                            "Select Expiry",
                            expiry_dates,
                            key="option_expiry"
                        )
            except Exception as e:
                st.error(f"Error fetching expiry dates: {e}")
        
        with col3:
            spot_price = st.number_input("Spot Price", min_value=0.0, value=0.0, step=0.05)

    # Market Data Tab
    with tabs[2]:
        st.header("Market Data")
        
        # Create subtabs for Live and Historical data
        market_subtabs = st.tabs(["📊 Live Data", "📈 Historical Data"])
        
        # Live Data Subtab
        with market_subtabs[0]:
            st.subheader("Live Market Data")
            
            col1, col2 = st.columns(2)
            with col1:
                live_index = st.selectbox(
                    "Select Index",
                    ["NIFTY 50", "BANK NIFTY", "FIN NIFTY", "MIDCAP NIFTY"],
                    key="live_market_index"
                )
            
            with col2:
                auto_refresh = st.checkbox("Auto Refresh (5s)", value=False)
            
            # Map index names to tokens for live data
            live_index_token_map = {
                "NIFTY 50": "99926000",
                "BANK NIFTY": "99926009",
                "FIN NIFTY": "99926037",
                "MIDCAP NIFTY": "99926012"
            }
            
            try:
                # Get current market date (only on weekdays)
                current_date = datetime.now()
                if current_date.weekday() >= 5:  # Saturday or Sunday
                    # Get last Friday
                    days_to_subtract = current_date.weekday() - 4
                    current_date = current_date - timedelta(days=days_to_subtract)
                
                # Format dates for market hours (9:15 AM to 3:30 PM)
                from_date = current_date.strftime("%Y-%m-%d 09:15")
                to_date = current_date.strftime("%Y-%m-%d 15:30")
                
                # Get live market data
                live_params = {
                    "exchange": "NSE",
                    "symboltoken": live_index_token_map[live_index],
                    "interval": "ONE_MINUTE",
                    "fromdate": from_date,
                    "todate": to_date
                }
                
                if st.checkbox("Show Debug Info"):
                    st.write("Request Parameters:", live_params)
                
                live_data = broker.smart_api.getCandleData(live_params)
                
                if live_data and live_data.get('status'):
                    live_df = pd.DataFrame(
                        live_data['data'],
                        columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
                    )
                    live_df['Datetime'] = pd.to_datetime(live_df['Datetime'])
                    
                    # Display live statistics
                    col1, col2, col3, col4 = st.columns(4)
                    current_price = live_df['Close'].iloc[-1]
                    prev_price = live_df['Close'].iloc[-2]
                    price_change = ((current_price - prev_price)/prev_price*100)
                    
                    with col1:
                        st.metric(
                            "Live Price",
                            f"₹{current_price:,.2f}",
                            f"{price_change:,.2f}%",
                            delta_color="normal"
                        )
                    
                    with col2:
                        st.metric(
                            "Day High",
                            f"₹{live_df['High'].max():,.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Day Low",
                            f"₹{live_df['Low'].min():,.2f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Volume",
                            f"{live_df['Volume'].iloc[-1]:,.0f}"
                        )
                    
                    # Live chart
                    fig = go.Figure()
                    
                    # Add price line
                    fig.add_trace(go.Scatter(
                        x=live_df['Datetime'],
                        y=live_df['Close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='#00ff00')
                    ))
                    
                    fig.update_layout(
                        title=f'{live_index} Live Price Movement',
                        yaxis_title='Price',
                        xaxis_title='Time',
                        template='plotly_dark',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if auto_refresh:
                        time.sleep(5)
                        st.experimental_rerun()
                
                else:
                    error_msg = live_data.get('message', 'Unknown error') if live_data else 'No response from API'
                    st.error(f"Failed to fetch live data: {error_msg}")
                    
            except Exception as e:
                st.error(f"Error fetching live data: {e}")
                if st.checkbox("Show Error Details"):
                    st.write("Error Type:", type(e).__name__)
                    st.write("Error Message:", str(e))
        
        # Historical Data Subtab
        with market_subtabs[1]:
            st.subheader("Historical Market Data")
            
            # Market data filters
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                market_index = st.selectbox(
                    "Select Index",
                    ["NIFTY 50", "BANK NIFTY", "FIN NIFTY", "MIDCAP NIFTY"],
                    key="market_index"
                )
                
            with col2:
                market_timeframe = st.selectbox(
                    "Select Timeframe",
                    ["1 Minute", "5 Minutes", "15 Minutes", "30 Minutes", "1 Hour", "1 Day"],
                    key="market_timeframe"
                )
                
            with col3:
                market_from_date = st.date_input(
                    "From Date",
                    datetime.now() - timedelta(days=7),
                    key="market_from_date"
                )
                
            with col4:
                market_to_date = st.date_input(
                    "To Date",
                    datetime.now(),
                    key="market_to_date"
                )
                
            if st.button("Fetch Historical Data"):
                try:
                    # Validate date range based on timeframe
                    if market_timeframe in ["1 Minute", "5 Minutes"]:
                        if (market_to_date - market_from_date).days > 7:
                            st.warning("For 1 and 5 minute timeframes, maximum period is 7 days. Adjusting date range.")
                            market_from_date = market_to_date - timedelta(days=7)
                    elif market_timeframe in ["15 Minutes", "30 Minutes"]:
                        if (market_to_date - market_from_date).days > 15:
                            st.warning("For 15 and 30 minute timeframes, maximum period is 15 days. Adjusting date range.")
                            market_from_date = market_to_date - timedelta(days=15)
                    
                    # Map index names to tokens
                    index_token_map = {
                        "NIFTY 50": "99926000",
                        "BANK NIFTY": "99926009",
                        "FIN NIFTY": "99926037",
                        "MIDCAP NIFTY": "99926012"
                    }
                    
                    # Map timeframe to API parameters
                    timeframe_map = {
                        "1 Minute": "ONE_MINUTE",
                        "5 Minutes": "FIVE_MINUTE",
                        "15 Minutes": "FIFTEEN_MINUTE",
                        "30 Minutes": "THIRTY_MINUTE",
                        "1 Hour": "ONE_HOUR",
                        "1 Day": "ONE_DAY"
                    }
                    
                    # Prepare parameters for API call
                    params = {
                        "exchange": "NSE",
                        "symboltoken": index_token_map[market_index],
                        "interval": timeframe_map[market_timeframe],
                        "fromdate": market_from_date.strftime("%Y-%m-%d 09:15"),
                        "todate": market_to_date.strftime("%Y-%m-%d 15:30")
                    }
                    
                    # Fetch market data
                    market_data = broker.smart_api.getCandleData(params)
                    
                    if market_data['status']:
                        # Convert to DataFrame
                        df = pd.DataFrame(
                            market_data['data'],
                            columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
                        )
                        df['Datetime'] = pd.to_datetime(df['Datetime'])
                        
                        # Display candlestick chart
                        fig = go.Figure(data=[go.Candlestick(
                            x=df['Datetime'],
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close']
                        )])
                        
                        fig.update_layout(
                            title=f'{market_index} Price Movement',
                            yaxis_title='Price',
                            xaxis_title='Date',
                            template='plotly_dark'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Current Price",
                                f"₹{df['Close'].iloc[-1]:,.2f}",
                                f"{((df['Close'].iloc[-1] - df['Close'].iloc[-2])/df['Close'].iloc[-2]*100):,.2f}%"
                            )
                        
                        with col2:
                            st.metric(
                                "Day High",
                                f"₹{df['High'].max():,.2f}"
                            )
                        
                        with col3:
                            st.metric(
                                "Day Low",
                                f"₹{df['Low'].min():,.2f}"
                            )
                        
                        with col4:
                            st.metric(
                                "Volume",
                                f"{df['Volume'].iloc[-1]:,.0f}"
                            )
                        
                        # Display data table
                        st.subheader("Market Data")
                        st.dataframe(
                            df.sort_values('Datetime', ascending=False),
                            use_container_width=True
                        )
                        
                    else:
                        st.error("Failed to fetch market data. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error fetching market data: {e}")

    # Analysis Tab
    with tabs[4]:
        st.header("Technical Analysis")
        
        # First row - Index, Timeframe, and Date Selection
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            index_mapping = {
                "NIFTY 50": "99926000",
                "BANK NIFTY": "99926009",
                "FIN NIFTY": "99926037",
                "MIDCAP NIFTY": "99926012"
            }
            selected_index = st.selectbox(
                "Select Index",
                list(index_mapping.keys()),
                index=0
            )
        with col2:
            timeframe_mapping = {
                "1 Minute": "ONE_MINUTE",
                "5 Minutes": "FIVE_MINUTE",
                "15 Minutes": "FIFTEEN_MINUTE",
                "30 Minutes": "THIRTY_MINUTE",
                "1 Hour": "ONE_HOUR",
                "1 Day": "ONE_DAY"
            }
            selected_timeframe = st.selectbox(
                "Select Timeframe",
                list(timeframe_mapping.keys()),
                index=5
            )
        with col3:
            from_date = st.date_input("From Date", datetime.now() - timedelta(days=30))
        with col4:
            to_date = st.date_input("To Date", datetime.now())
            
        # Second row - Moving Average Parameters
        st.subheader("Moving Average Parameters")
        ma_col1, ma_col2, ma_col3, ma_col4 = st.columns(4)
        
        with ma_col1:
            fast_ma = st.number_input("Fast MA Length", min_value=1, value=9)
        with ma_col2:
            slow_ma = st.number_input("Slow MA Length", min_value=1, value=21)
        with ma_col3:
            fast_ema = st.number_input("Fast EMA Length", min_value=1, value=12)
        with ma_col4:
            slow_ema = st.number_input("Slow EMA Length", min_value=1, value=26)
            
        if st.button("Get Historical Data"):
            try:
                # Adjust date range based on timeframe
                if selected_timeframe in ["1 Minute", "5 Minutes"]:
                    if (to_date - from_date).days > 7:
                        st.warning("For 1 and 5 minute timeframes, maximum period is 7 days. Adjusting date range.")
                        from_date = to_date - timedelta(days=7)
                elif selected_timeframe in ["15 Minutes", "30 Minutes"]:
                    if (to_date - from_date).days > 15:
                        st.warning("For 15 and 30 minute timeframes, maximum period is 15 days. Adjusting date range.")
                        from_date = to_date - timedelta(days=15)
                
                params = {
                    "exchange": "NSE",
                    "symboltoken": index_mapping[selected_index],
                    "interval": timeframe_mapping[selected_timeframe],
                    "fromdate": from_date.strftime("%Y-%m-%d 09:15"),
                    "todate": to_date.strftime("%Y-%m-%d 15:30")
                }
                
                hist_data = broker.smart_api.getCandleData(params)
                if hist_data['status']:
                    df = pd.DataFrame(hist_data['data'], 
                                    columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    df['Datetime'] = pd.to_datetime(df['Datetime'])
                    
                    # Calculate indicators
                    df[f'MA_{fast_ma}'] = df['Close'].rolling(window=fast_ma).mean()
                    df[f'MA_{slow_ma}'] = df['Close'].rolling(window=slow_ma).mean()
                    df[f'EMA_{fast_ema}'] = df['Close'].ewm(span=fast_ema, adjust=False).mean()
                    df[f'EMA_{slow_ema}'] = df['Close'].ewm(span=slow_ema, adjust=False).mean()
                    df['RSI'] = calculate_rsi(df['Close'])
                    df['MACD'], df['Signal'] = calculate_macd(df['Close'])
                    df['MACD_Hist'] = df['MACD'] - df['Signal']
                    
                    # Generate crossover signals
                    df['MA_Cross'] = np.where(df[f'MA_{fast_ma}'] > df[f'MA_{slow_ma}'], 1, -1)
                    df['MA_Signal'] = df['MA_Cross'].diff()
                    df['EMA_Cross'] = np.where(df[f'EMA_{fast_ema}'] > df[f'EMA_{slow_ema}'], 1, -1)
                    df['EMA_Signal'] = df['EMA_Cross'].diff()
                    
                    # Current values
                    current_price = df['Close'].iloc[-1]
                    current_rsi = df['RSI'].iloc[-1]
                    current_macd_hist = df['MACD_Hist'].iloc[-1]
                    prev_macd_hist = df['MACD_Hist'].iloc[-2]
                    
                    # Display summary and signals
                    st.subheader(f"{selected_index} Analysis & Signals")
                    
                    # Market metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"₹{current_price:,.2f}", 
                                f"{((current_price - df['Close'].iloc[-2])/df['Close'].iloc[-2]*100):,.2f}%")
                    with col2:
                        st.metric("RSI", f"{current_rsi:.2f}")
                    with col3:
                        st.metric("MACD Histogram", f"{current_macd_hist:.2f}")
                    
                    # Signal Analysis
                    signal_col1, signal_col2 = st.columns(2)
                    
                    with signal_col1:
                        st.markdown("### Trading Signals")
                        
                        # Combined MA/EMA Crossover Analysis
                        bullish_crossover = (df['MA_Signal'].iloc[-1] == 2 or df['EMA_Signal'].iloc[-1] == 2)
                        bearish_crossover = (df['MA_Signal'].iloc[-1] == -2 or df['EMA_Signal'].iloc[-1] == -2)
                        
                        # RSI Confirmation
                        rsi_bullish = current_rsi < 40
                        rsi_bearish = current_rsi > 60
                        
                        # MACD Confirmation
                        macd_bullish = current_macd_hist > 0 and prev_macd_hist < 0
                        macd_bearish = current_macd_hist < 0 and prev_macd_hist > 0
                        
                        # Generate Strong Buy Signals
                        if bullish_crossover and rsi_bullish and current_macd_hist > 0:
                            st.success("🟢 Strong Buy Signal - Consider Call Options")
                            st.write("- Bullish MA/EMA Crossover")
                            st.write("- RSI showing oversold/bullish")
                            st.write("- MACD confirms upward momentum")
                            
                        # Generate Strong Sell Signals
                        elif bearish_crossover and rsi_bearish and current_macd_hist < 0:
                            st.error("🔴 Strong Sell Signal - Consider Put Options")
                            st.write("- Bearish MA/EMA Crossover")
                            st.write("- RSI showing overbought/bearish")
                            st.write("- MACD confirms downward momentum")
                            
                        # Moderate Signals
                        elif bullish_crossover or (rsi_bullish and macd_bullish):
                            st.info("🔵 Moderate Buy Signal - Watch for confirmation")
                            st.write("- Some bullish indicators present")
                            st.write("- Wait for additional confirmation")
                            
                        elif bearish_crossover or (rsi_bearish and macd_bearish):
                            st.warning("🟡 Moderate Sell Signal - Watch for confirmation")
                            st.write("- Some bearish indicators present")
                            st.write("- Wait for additional confirmation")
                            
                        else:
                            st.write("No clear trading signals at the moment")
                    
                    with signal_col2:
                        st.markdown("### Technical Levels")
                        st.write(f"Current RSI: {current_rsi:.2f}")
                        st.write(f"MACD Histogram: {current_macd_hist:.2f}")
                        st.write(f"Fast MA ({fast_ma}): ₹{df[f'MA_{fast_ma}'].iloc[-1]:,.2f}")
                        st.write(f"Slow MA ({slow_ma}): ₹{df[f'MA_{slow_ma}'].iloc[-1]:,.2f}")
                        st.write(f"Fast EMA ({fast_ema}): ₹{df[f'EMA_{fast_ema}'].iloc[-1]:,.2f}")
                        st.write(f"Slow EMA ({slow_ema}): ₹{df[f'EMA_{slow_ema}'].iloc[-1]:,.2f}")
                    
                    # Candlestick chart with indicators
                    fig = go.Figure()
                    
                    # Add candlestick
                    fig.add_trace(go.Candlestick(
                        x=df['Datetime'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Price'
                    ))
                    
                    # Add MAs and EMAs
                    fig.add_trace(go.Scatter(
                        x=df['Datetime'], 
                        y=df[f'MA_{fast_ma}'],
                        name=f'MA {fast_ma}',
                        line=dict(color='orange')
                    ))
                    fig.add_trace(go.Scatter(
                        x=df['Datetime'], 
                        y=df[f'MA_{slow_ma}'],
                        name=f'MA {slow_ma}',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=df['Datetime'], 
                        y=df[f'EMA_{fast_ema}'],
                        name=f'EMA {fast_ema}',
                        line=dict(color='yellow', dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=df['Datetime'], 
                        y=df[f'EMA_{slow_ema}'],
                        name=f'EMA {slow_ema}',
                        line=dict(color='purple', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title=f'{selected_index} Price Movement with Indicators',
                        yaxis_title='Price',
                        xaxis_title='Date',
                        template='plotly_dark',
                        yaxis=dict(tickformat=".2f"),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display OHLC data with indicators
                    display_df = df.tail().copy()
                    st.dataframe(display_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error fetching historical data: {e}")

    # Settings Tab
    with tabs[7]:
        st.header("Settings")
        
        # Create sections for different settings
        st.subheader("API Configuration")
        api_col1, api_col2 = st.columns(2)
        
        with api_col1:
            st.text_input(
                "API Key",
                value=os.getenv('ANGEL_API_KEY', ''),
                type="password",
                key="api_key_input"
            )
            st.text_input(
                "Client ID",
                value=os.getenv('ANGEL_CLIENT_ID', ''),
                key="client_id_input"
            )
        
        with api_col2:
            st.text_input(
                "Password",
                value=os.getenv('ANGEL_PASSWORD', ''),
                type="password",
                key="password_input"
            )
            st.text_input(
                "TOTP Key",
                value=os.getenv('ANGEL_TOTP_KEY', ''),
                type="password",
                key="totp_key_input"
            )

        if st.button("Save API Settings"):
            try:
                # Create or update .env file
                env_file = ".env"
                env_data = {
                    "ANGEL_API_KEY": st.session_state.api_key_input,
                    "ANGEL_CLIENT_ID": st.session_state.client_id_input,
                    "ANGEL_PASSWORD": st.session_state.password_input,
                    "ANGEL_TOTP_KEY": st.session_state.totp_key_input
                }
                
                with open(env_file, "w") as f:
                    for key, value in env_data.items():
                        f.write(f"{key}={value}\n")
                
                st.success("API settings saved successfully! Please restart the application.")
            except Exception as e:
                st.error(f"Error saving settings: {e}")

        # Trading Settings
        st.subheader("Trading Settings")
        trading_col1, trading_col2 = st.columns(2)
        
        with trading_col1:
            default_quantity = st.number_input(
                "Default Trading Quantity",
                min_value=1,
                value=1,
                help="Default quantity for trading orders"
            )
            
            risk_percentage = st.slider(
                "Risk Percentage per Trade",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Maximum risk percentage per trade"
            )

        with trading_col2:
            default_stoploss = st.number_input(
                "Default Stop Loss (%)",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="Default stop loss percentage"
            )
            
            default_target = st.number_input(
                "Default Target (%)",
                min_value=0.1,
                max_value=20.0,
                value=4.0,
                step=0.1,
                help="Default target percentage"
            )

        if st.button("Save Trading Settings"):
            try:
                # Save trading settings to a JSON file
                trading_settings = {
                    "default_quantity": default_quantity,
                    "risk_percentage": risk_percentage,
                    "default_stoploss": default_stoploss,
                    "default_target": default_target
                }
                
                with open("trading_settings.json", "w") as f:
                    json.dump(trading_settings, f, indent=4)
                
                st.success("Trading settings saved successfully!")
            except Exception as e:
                st.error(f"Error saving trading settings: {e}")

        # Display Settings
        st.subheader("Display Settings")
        display_col1, display_col2 = st.columns(2)
        
        with display_col1:
            chart_theme = st.selectbox(
                "Chart Theme",
                ["plotly_dark", "plotly", "plotly_white"],
                index=0,
                help="Select the theme for charts"
            )
            
            show_indicators = st.multiselect(
                "Default Indicators",
                ["RSI", "MACD", "Moving Averages", "Bollinger Bands"],
                default=["RSI", "MACD"],
                help="Select default indicators to show on charts"
            )

        with display_col2:
            auto_refresh_interval = st.number_input(
                "Auto Refresh Interval (seconds)",
                min_value=1,
                max_value=60,
                value=5,
                help="Interval for auto-refreshing data"
            )
            
            show_debug_info = st.checkbox(
                "Show Debug Information",
                value=False,
                help="Show additional debugging information"
            )

        if st.button("Save Display Settings"):
            try:
                # Save display settings to a JSON file
                display_settings = {
                    "chart_theme": chart_theme,
                    "show_indicators": show_indicators,
                    "auto_refresh_interval": auto_refresh_interval,
                    "show_debug_info": show_debug_info
                }
                
                with open("display_settings.json", "w") as f:
                    json.dump(display_settings, f, indent=4)
                
                st.success("Display settings saved successfully!")
            except Exception as e:
                st.error(f"Error saving display settings: {e}")

        # Add import/export settings functionality
        st.subheader("Backup & Restore")
        backup_col1, backup_col2 = st.columns(2)
        
        with backup_col1:
            if st.button("Export Settings"):
                try:
                    # Combine all settings
                    all_settings = {
                        "trading_settings": trading_settings,
                        "display_settings": display_settings
                    }
                    
                    # Convert to JSON string
                    settings_json = json.dumps(all_settings, indent=4)
                    
                    # Create download button
                    st.download_button(
                        "Download Settings",
                        settings_json,
                        "trading_bot_settings.json",
                        "application/json"
                    )
                except Exception as e:
                    st.error(f"Error exporting settings: {e}")

        with backup_col2:
            uploaded_file = st.file_uploader(
                "Import Settings",
                type="json",
                help="Upload a previously exported settings file"
            )
            
            if uploaded_file is not None:
                try:
                    imported_settings = json.load(uploaded_file)
                    
                    # Save the imported settings
                    with open("trading_settings.json", "w") as f:
                        json.dump(imported_settings["trading_settings"], f, indent=4)
                    
                    with open("display_settings.json", "w") as f:
                        json.dump(imported_settings["display_settings"], f, indent=4)
                    
                    st.success("Settings imported successfully! Please refresh the page.")
                except Exception as e:
                    st.error(f"Error importing settings: {e}")

    # Orders Tab
    with tabs[3]:
        st.header("Orders")
        
        # Create subtabs for different order types
        order_tabs = st.tabs(["🔄 Active Orders", "📝 Place Order", "📜 Order History"])
        
        # Active Orders Tab
        with order_tabs[0]:
            st.subheader("Active Orders")
            try:
                # Fetch active orders
                active_orders = broker.smart_api.orderBook()
                if active_orders and active_orders.get('status'):
                    if active_orders.get('data'):
                        # Convert to DataFrame
                        orders_df = pd.DataFrame(active_orders['data'])
                        
                        # Format and display orders
                        display_cols = [
                            'orderid', 'tradingsymbol', 'transactiontype',
                            'producttype', 'quantity', 'price', 'orderstatus',
                            'ordertype', 'exchtime'
                        ]
                        
                        st.dataframe(
                            orders_df[display_cols].sort_values('exchtime', ascending=False),
                            use_container_width=True
                        )
                        
                        # Add cancel order functionality
                        order_to_cancel = st.selectbox(
                            "Select Order to Cancel",
                            orders_df['orderid'].tolist()
                        )
                        
                        if st.button("Cancel Selected Order"):
                            try:
                                cancel_response = broker.smart_api.cancelOrder(
                                    order_to_cancel,
                                    orders_df[orders_df['orderid'] == order_to_cancel]['variety'].iloc[0]
                                )
                                if cancel_response and cancel_response.get('status'):
                                    st.success("Order cancelled successfully!")
                                else:
                                    st.error("Failed to cancel order")
                            except Exception as e:
                                st.error(f"Error cancelling order: {e}")
                    else:
                        st.info("No active orders found")
                else:
                    st.error("Failed to fetch active orders")
            except Exception as e:
                st.error(f"Error fetching orders: {e}")
        
        # Place Order Tab
        with order_tabs[1]:
            st.subheader("Place New Order")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Symbol search
                exchange = st.selectbox(
                    "Exchange",
                    ["NSE", "BSE", "NFO"],
                    key="order_exchange"
                )
                
                symbol = st.text_input(
                    "Symbol",
                    placeholder="Enter trading symbol",
                    key="order_symbol"
                ).upper()
                
                selected_symbol = None
                token = None
                
                if symbol:
                    try:
                        search_result = broker.smart_api.searchScrip(exchange, symbol)
                        if search_result and search_result.get('status'):
                            symbols_df = pd.DataFrame(search_result['data'])
                            selected_symbol = st.selectbox(
                                "Select Trading Symbol",
                                symbols_df['tradingsymbol'].tolist()
                            )
                            
                            if selected_symbol:
                                symbol_data = symbols_df[symbols_df['tradingsymbol'] == selected_symbol].iloc[0]
                                token = symbol_data['symboltoken']
                                
                                # Display symbol details
                                st.write("Symbol Details:")
                                st.write(f"Symbol: {selected_symbol}")
                                st.write(f"Token: {token}")
                                st.write(f"Exchange: {exchange}")
                    except Exception as e:
                        st.error(f"Error searching symbol: {e}")
            
            with col2:
                # Order details
                transaction_type = st.selectbox(
                    "Transaction Type",
                    ["BUY", "SELL"]
                )
                
                product_type = st.selectbox(
                    "Product Type",
                    ["DELIVERY", "INTRADAY", "MARGIN"]
                )
                
                order_type = st.selectbox(
                    "Order Type",
                    ["MARKET", "LIMIT", "SL", "SL-M"]
                )
            
            col3, col4 = st.columns(2)
            
            with col3:
                quantity = st.number_input(
                    "Quantity",
                    min_value=1,
                    value=1
                )
                
                if order_type in ["LIMIT", "SL"]:
                    price = st.number_input(
                        "Price",
                        min_value=0.05,
                        value=0.0,
                        step=0.05
                    )
                else:
                    price = 0
                
                if order_type in ["SL", "SL-M"]:
                    trigger_price = st.number_input(
                        "Trigger Price",
                        min_value=0.05,
                        value=0.0,
                        step=0.05
                    )
                else:
                    trigger_price = 0
            
            with col4:
                validity = st.selectbox(
                    "Validity",
                    ["DAY", "IOC"]
                )
                
                variety = st.selectbox(
                    "Variety",
                    ["NORMAL", "STOPLOSS", "AMO"]
                )
            
            if st.button("Place Order"):
                if not (selected_symbol and token):
                    st.error("Please select a valid symbol first")
                else:
                    try:
                        order_params = {
                            "variety": variety,
                            "tradingsymbol": selected_symbol,
                            "symboltoken": token,
                            "transactiontype": transaction_type,
                            "exchange": exchange,
                            "ordertype": order_type,
                            "producttype": product_type,
                            "duration": validity,
                            "quantity": quantity
                        }
                        
                        if order_type in ["LIMIT", "SL"]:
                            order_params["price"] = price
                            
                        if order_type in ["SL", "SL-M"]:
                            order_params["triggerprice"] = trigger_price
                        
                        order_response = broker.smart_api.placeOrder(order_params)
                        
                        if order_response and order_response.get('status'):
                            st.success(f"Order placed successfully! Order ID: {order_response['data']['orderid']}")
                        else:
                            st.error("Failed to place order")
                    except Exception as e:
                        st.error(f"Error placing order: {e}")
        
        # Order History Tab
        with order_tabs[2]:
            st.subheader("Order History")
            
            try:
                # Fetch order history
                order_history = broker.smart_api.orderBook()
                if order_history and order_history.get('status'):
                    if order_history.get('data'):
                        history_df = pd.DataFrame(order_history['data'])
                        
                        # Format and display history
                        display_cols = [
                            'orderid', 'tradingsymbol', 'transactiontype',
                            'producttype', 'quantity', 'price', 'orderstatus',
                            'ordertype', 'exchtime'
                        ]
                        
                        st.dataframe(
                            history_df[display_cols].sort_values('exchtime', ascending=False),
                            use_container_width=True
                        )
                        
                        # Add export functionality
                        if st.button("Export Order History"):
                            csv = history_df.to_csv(index=False)
                            st.download_button(
                                "Download CSV",
                                csv,
                                "order_history.csv",
                                "text/csv",
                                key='download-csv'
                            )
                    else:
                        st.info("No orders in history")
                else:
                    st.error("Failed to fetch order history")
                    
            except Exception as e:
                st.error(f"Error fetching order history: {e}")

    # Algo AI Tab
    with tabs[8]:
        st.header("Algorithmic Trading AI")
        algo_container = st.container()
        
        with algo_container:
            if 'ma_length' not in st.session_state:
                st.session_state.ma_length = 10
            if 'ema_length' not in st.session_state:
                st.session_state.ema_length = 10
            
            # Use algo_ai functionality
            symbol = st.selectbox(
                "Select Index",
                options=list(algo_ai.INDICES.keys()),
                key='algo_ai_symbol'
            )
            symbol = algo_ai.INDICES[symbol]
            
            # Get data and calculate indicators
            df = algo_ai.get_data(symbol)
            if df is not None:
                df = algo_ai.calculate_indicators(df)
                df = algo_ai.generate_signals(df)
                
                if df is not None:
                    # Process historical signals
                    algo_ai.process_historical_signals(df, symbol)
                    
                    # Create sub-tabs for Algo AI
                    ai_tabs = st.tabs([
                        "Chart & Signals",
                        "Paper Trading",
                        "Portfolio",
                        "Options Chain"
                    ])
                    
                    with ai_tabs[0]:
                        algo_ai.display_metrics(df)
                        fig = algo_ai.plot_chart(df, f"{symbol} - 5min")
                        st.plotly_chart(fig, use_container_width=True)
                        algo_ai.display_strategy_dashboard(df, symbol)
                    
                    with ai_tabs[1]:
                        algo_ai.display_paper_trading(df, symbol)
                    
                    with ai_tabs[2]:
                        algo_ai.display_portfolio_dashboard(df, symbol)
                    
                    with ai_tabs[3]:
                        algo_ai.display_options_analysis(df, symbol)

    # Strategy Manager Tab
    with tabs[9]:
        st.header("Strategy Manager")
        strategy_manager = StrategyManager()
        
        # Add strategy form
        with st.form("add_strategy_form"):
            st.subheader("Add New Strategy")
            strategy_name = st.text_input("Strategy Name")
            strategy_type = st.selectbox("Strategy Type", ["moving_average", "rsi", "combined"])
            fast_period = st.number_input("Fast Period", min_value=1, value=10)
            slow_period = st.number_input("Slow Period", min_value=1, value=30)
            rsi_period = st.number_input("RSI Period", min_value=1, value=14)
            oversold = st.number_input("RSI Oversold Level", min_value=1, value=30)
            overbought = st.number_input("RSI Overbought Level", min_value=1, value=70)
            ma_weight = st.number_input("MA Weight", min_value=0.0, max_value=1.0, value=0.5)
            rsi_weight = st.number_input("RSI Weight", min_value=0.0, max_value=1.0, value=0.5)
            submit_button = st.form_submit_button("Add Strategy")
            
            if submit_button:
                strategy_config = {
                    "type": strategy_type,
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "rsi": {
                        "period": rsi_period,
                        "oversold": oversold,
                        "overbought": overbought
                    },
                    "weights": {
                        "ma": ma_weight,
                        "rsi": rsi_weight
                    }
                }
                strategy_manager.add_strategy(strategy_name, strategy_config)
                st.success(f"Strategy '{strategy_name}' added successfully!")
        
        # Display available strategies
        st.subheader("Available Strategies")
        strategies = strategy_manager.get_strategies()
        if strategies:
            for strategy in strategies:
                st.write(strategy)
        else:
            st.write("No strategies available.")
        
        # Activate strategy
        st.subheader("Activate Strategy")
        strategy_to_activate = st.selectbox("Select Strategy to Activate", strategies)
        if st.button("Activate Strategy"):
            if strategy_manager.activate_strategy(strategy_to_activate):
                st.success(f"Strategy '{strategy_to_activate}' activated successfully!")
            else:
                st.error(f"Failed to activate strategy '{strategy_to_activate}'.")

if __name__ == "__main__":
    main()