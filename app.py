import pickle
import numpy as np
import pandas as pd
import streamlit as st

model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))

st.title("Sarcasm Detection")
text = st.text_area("Text")

if st.button("Detect"):
	test = cv.transform([text]).toarray()
	res = model.predict(test)
	print(res)
	st.success("Detected: " + str(res[0]))
