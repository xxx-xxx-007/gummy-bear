import streamlit as st
import joblib
import predict_text as predict

st.markdown("""
<style>
    .stMetricValue-positif {
        background-color: green;
        color: white;
        border-radius: 10px;
        padding: 5px;
        text-align: center;
        font-size: 20px;
    }
    .stMetricValue-negatif {
        background-color: red;
        color: white;
        border-radius: 10px;
        padding: 5px;
        text-align: center;
        font-size: 20px;
    }
    .stMetricValue-netral {
        background-color: yellow;
        color: white;
        border-radius: 10px;
        padding: 5px;
        text-align: center;
        font-size: 20px;
    }
    .stMetricLabel {
        font-size: 16px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource  # Ensures model is only loaded once
def load_model():
    return joblib.load('model.pkl')

@st.cache_resource  # Ensures vectorizer is only loaded once
def load_vectorizer():
    return joblib.load('vectorizer.pkl')

model = load_model()
vectorizer = load_vectorizer()

if 'positif' not in st.session_state:
    st.session_state.positif = 0
if 'negatif' not in st.session_state:
    st.session_state.negatif = 0
if 'netral' not in st.session_state:
    st.session_state.netral = 0

st.title("Form Kritik dan Saran")
with st.container(height=150, border=False):
    st.write("Silakan isi form di bawah ini untuk memberikan kritik dan saran Anda. " \
    "Kami sangat menghargai masukan Anda untuk meningkatkan pelayanan kami. Terima kasih atas partisipasi Anda!")
    st.container(height=5, border=False)
    user_input = st.chat_input("Say something")
    if user_input:
        st.toast("Terima kasih atas kritik dan saran Anda!")
        prediction = predict.predict(model, vectorizer, user_input).lower()
        if prediction == "positif":
            st.session_state.positif += 1
        elif prediction == "negatif":
            st.session_state.negatif += 1
        elif prediction == "netral":
            st.session_state.netral += 1

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Positif", value=st.session_state.positif, delta=None, help="Positive feedback")
    st.markdown('<div class="stMetricValue-positif"></div>', unsafe_allow_html=True)
with col2:  
    st.metric(label="Netral", value=st.session_state.netral, delta=None, help="Netral feedback")
    st.markdown('<div class="stMetricValue-netral"></div>', unsafe_allow_html=True)
with col3:
    st.metric(label="Negatif", value=st.session_state.negatif, delta=None, help="Negative feedback")
    st.markdown('<div class="stMetricValue-negatif"></div>', unsafe_allow_html=True)
