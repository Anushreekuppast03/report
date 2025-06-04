import torchvision
torchvision.disable_beta_transforms_warning()


import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


import streamlit as st
from transformers import pipeline
import asyncio
import torch

# Fix for asyncio and Torch issues
if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

# Load the model
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

generator = load_model()

# Function to get medical advice
def get_medical_advice(symptoms):
    prompt = f"What could be the possible illness for symptoms: {symptoms}?"
    response = generator(prompt, max_length=100)
    return response[0]["generated_text"]

# Streamlit UI
def main():
    st.title("🤖 AI-Powered Symptom Checker")
    symptoms = st.text_input("Enter your symptoms (comma-separated):")
    
    if st.button("Get Medical Advice"):
        if symptoms:
            result = get_medical_advice(symptoms)
            st.success(f"🔍 Possible condition: {result}")
        else:
            st.error("❌ Please enter symptoms before submitting.")

if __name__ == "__main__":
    main()
