import streamlit as st
from SmartApi import SmartConnect
import pyotp
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class BrokerService:
    def __init__(self):
        self.api_key = os.getenv('ANGEL_API_KEY')
        self.client_id = os.getenv('ANGEL_CLIENT_ID')
        self.password = os.getenv('ANGEL_PASSWORD')
        self.totp_key = os.getenv('ANGEL_TOTP_KEY')
        self.smart_api = None
        
    def authenticate(self):
        """Connect to broker API"""
        try:
            self.smart_api = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_key)
            current_totp = totp.now()
            
            login_response = self.smart_api.generateSession(
                self.client_id, 
                self.password, 
                current_totp
            )
            
            if login_response.get('status'):
                refresh_token = login_response['data']['refreshToken']
                self.smart_api.generateToken(refresh_token)
                return True
                
            st.error("Login failed. Please check your credentials.")
            return False
            
        except Exception as e:
            st.error(f"Error connecting to broker: {e}")
            return False
            
    def get_candle_data(self, params):
        """Get market data"""
        if not self.smart_api:
            st.error("Not connected to broker")
            return None
            
        try:
            return self.smart_api.getCandleData(params)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
            
    def place_order(self, params):
        """Place trading order"""
        if not self.smart_api:
            st.error("Not connected to broker")
            return None
            
        try:
            return self.smart_api.placeOrder(params)
        except Exception as e:
            st.error(f"Error placing order: {e}")
            return None
