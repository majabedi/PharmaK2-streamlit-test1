import streamlit as st


if "n" == 'y':
    from src.components import sidebar
    sidebar.show_sidebar()

st.title('pharmak2')
st.write('This is an initial version')