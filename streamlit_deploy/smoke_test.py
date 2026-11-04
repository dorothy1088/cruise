import sys
import streamlit as st
st.set_page_config(page_title="Streamlit Cloud startup test")
st.title("Cloud startup test: OK")
st.success("If you can see this page, Streamlit itself has started.")
st.write("Python:", sys.version)
st.write("Streamlit:", st.__version__)
