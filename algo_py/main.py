import streamlit as st
from quant_algo import main as quant_algo_main

st.set_page_config(page_title="Quant Algo Trading", page_icon="📈", layout="wide")

def main():
    quant_algo_main()

if __name__ == "__main__":
    main()
