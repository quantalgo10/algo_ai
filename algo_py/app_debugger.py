import streamlit as st
from market_data_pipeline import MarketDataPipeline

class AppDebugger:
    def __init__(self):
        st.title("🔍 System Diagnostics")
        self.market_data = MarketDataPipeline()
        
    def run_diagnostics(self):
        with st.expander("System Status"):
            # Test Market Data
            try:
                data = self.market_data.fetch_data("NIFTY")
                if data is not None:
                    st.success("✅ Market Data Pipeline")
                else:
                    st.error("❌ Market Data Pipeline")
            except Exception as e:
                st.error(f"❌ Market Data Error: {str(e)}")
                
def main():
    debugger = AppDebugger()
    debugger.run_diagnostics()

if __name__ == "__main__":
    main()