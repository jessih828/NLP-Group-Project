import streamlit as st
from streamlit_pages import page_dataset_comparison, page_single_comparison, page_enhance_cv, page_interview_questions

# Streamlit app
def main():
    # Add custom CSS for styling
    st.markdown('''
        <style>
            .css-1egvi7u {margin-top: -4rem;}
            .css-znku1x a {color: #9d03fc;}
            .css-qrbaxs {min-height: 0.0rem;}
            .stSpinner > div > div {border-top-color: #9d03fc;}
            .css-15tx938{min-height: 0.0rem;}
            header {visibility: hidden;}
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .sidebar-content {
                display: flex;
                flex-direction: column;
                align-items: center;
                height: 100%;
                justify-content: center;
            }
            .stButton>button {
                width: 100%;
                padding: 0.5rem 1rem;
                margin-bottom: 0.5rem;
                border-radius: 4px;
                text-align: center;
                background-color: #ffffff;
                color: #000000;
                box-sizing: border-box;
            }
            .stButton>button:hover {
                background-color: #9d03fc;
                color: white;
            }
            .stButton>button:active {
                background-color: #9d03fc;
                color: white;
            }
        </style>
    ''', unsafe_allow_html=True)

    if 'page' not in st.session_state:
        st.session_state.page = 'Dataset Comparison'

    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

    if st.sidebar.button('Dataset Comparison'):
        st.session_state.page = 'Dataset Comparison'
    if st.sidebar.button('1-vs-1 Comparison'):
        st.session_state.page = '1-vs-1 Comparison'
    if st.sidebar.button('Enhance my CV'):
        st.session_state.page = 'Enhance my CV'
    if st.sidebar.button('Interview Question Generator'):
        st.session_state.page = 'Interview Question Generator'

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.page == "Dataset Comparison":
        page_dataset_comparison()
    elif st.session_state.page == "1-vs-1 Comparison":
        page_single_comparison()
    elif st.session_state.page == "Enhance my CV":
        page_enhance_cv()
    elif st.session_state.page == "Interview Question Generator":
        page_interview_questions()

 


if __name__ == "__main__":
    main()
