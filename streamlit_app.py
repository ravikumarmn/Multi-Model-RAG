import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Set up Streamlit page configuration for a wide layout
st.set_page_config(
    page_title="Image Question Answering with Gemini AI", layout="centered")
st.title("🖼️ Vision-Language Integration for RAG Systems")
st.divider()
# Initialize session state to maintain conversation history and image
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None

# Sidebar for File Upload
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Upload an image (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

    # Load and display uploaded image, if available
    if uploaded_file:
        st.session_state["uploaded_image"] = Image.open(uploaded_file)
        st.image(st.session_state["uploaded_image"],
                 caption="Uploaded Image")

# Display chat history, if any
if st.session_state["messages"]:
    for msg in st.session_state["messages"]:
        role = "Assistant" if msg["role"] == "assistant" else "You"
        st.chat_message(role).write(msg["content"])

# Handle chat input from the user
if prompt := st.chat_input(placeholder="Ask a question about the uploaded image, e.g., 'What is the restaurant name?'"):
    # Append user message to the session state
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Check if an image is uploaded
    if st.session_state["uploaded_image"] is None:
        # If no image is uploaded, answer as a text-only interaction
        with st.spinner("Generating response... Please wait"):
            try:
                # Set up configuration for Google Generative AI
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                # Generate response using the prompt
                response = model.generate_content([prompt])
                response_text = response.text

                # Store response in session state
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response_text})

                # Display the assistant response
                st.chat_message("assistant").write(response_text)

            except Exception as e:
                st.error(f"An error occurred: {e}")

    else:
        # Show a spinner while processing
        with st.spinner("Generating response... Please wait"):
            try:
                # Prepare the prompt with the user query
                prompt_template_str = f"""\
                You have been given an image along with a specific question from the user. \
                The user's question is: "{prompt}". \
                Carefully analyze the provided image to extract any relevant information needed to answer the question. \
                If the information needed to answer the question is not available in the image, respond accordingly without making up an answer. \
                Ask the user for more context or indicate that the information is not present in the image. \
                Your response should be clear and concise, relying solely on the context provided by the image. \
                Please return the answer formatted in markdown, and do not include any additional information outside of this answer format.\
                """
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model = genai.GenerativeModel("gemini-1.5-flash")
                # Generate content using Gemini API
                response = model.generate_content(
                    [prompt_template_str, st.session_state["uploaded_image"]])

                # Store response in session state
                response_text = response.text
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response_text})

                # Display the assistant response
                st.chat_message("assistant").write(response_text)

            except Exception as e:
                st.error(f"An error occurred: {e}")
