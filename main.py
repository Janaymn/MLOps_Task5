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
        # Local model processing
        data = { "model": model_name, "prompt": prompt, "stream": False }
        response = requests.post(LOCAL_OLLAMA_ENDPOINT, json=data)
        response.raise_for_status()
        return response.json().get("response", "No response found.")
    except requests.exceptions.RequestException as e:
        return f"An error occurred connecting to Ollama: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Dropdown with two thinking models (local)
model_dropdown = gr.Dropdown(
    choices=["qwen:0.5b", "codellama:7b-instruct"],  # Both local models
    value="qwen:0.5b",  # Default to lighter model
    label="Select Model"
)

# Set up the Gradio interface
gui = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=3, label="Your Prompt", placeholder="Enter a question or idea..."),
        model_dropdown
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Local Thinking AI Assistant",
    description="Choose between 'qwen:0.5b' (lightweight) and 'codellama:7b-instruct' (larger model) to generate responses locally."
)

# FastAPI setup to host the Gradio app
app = FastAPI(title="Local Thinking AI Assistant API")
app = gr.mount_gradio_app(app, gui, path="/")
