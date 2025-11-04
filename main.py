import gradio as gr
import requests
import os
from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables if needed
load_dotenv()

# Local Ollama endpoint for text generation
LOCAL_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

def generate_text(prompt: str, model_name: str):
    if not prompt:
        return "Please enter a prompt."
    try:
        # Local model processing for the thinking model
        data = { "model": model_name, "prompt": prompt, "stream": False }
        response = requests.post(LOCAL_OLLAMA_ENDPOINT, json=data)
        response.raise_for_status()
        return response.json().get("response", "No response found.")
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Dropdown with only the thinking model
model_dropdown = gr.Dropdown(
    choices=["codellama:7b-instruct"],  # Only thinking model
    value="codellama:7b-instruct",  # Default value
    label="Select Model"
)

# Set up the Gradio interface
gui = gr.Interface(
    fn=generate_text,
    inputs=[gr.Textbox(lines=3, label="Your Prompt", placeholder="Enter a starting phrase..."), model_dropdown],
    outputs=gr.Textbox(label="Generated Text"),
    title="Local Ollama Thinking Model",
    description="The model 'codellama:7b-instruct' runs locally for general conversation or instruction-following tasks."
)

# FastAPI setup to host the Gradio app
app = FastAPI(title="Ollama Thinking Model API")
app = gr.mount_gradio_app(app, gui, path="/")
