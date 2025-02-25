import os
import streamlit as st
from PIL import Image
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from florence_caption import generate_caption
from florence_objects import generate_labels
from vosk_stt import transcribe_audio

# ✅ Initialize LangChain Groq Model
chat_model = ChatGroq(
    temperature=0.7,  # Adjust creativity
    api_key="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"  # Replace with actual API key
)

# ✅ Maintain chat history
history = [
    SystemMessage(content="You are an AI assistant analyzing multiple images and audio.")
]

# ✅ Store processed frame data (Dictionary: {filename → {caption, objects}})
frame_data = {}

frames_folder = "/home/vmukti/multi_aii/frames/"
audio_path = "/home/vmukti/multi_aii/harvard.wav"

# Ensure the folder exists
if not os.path.exists(frames_folder):
    st.error("❌ Error: Frames folder not found!")
    st.stop()

# Ensure audio file exists
if not os.path.exists(audio_path):
    st.error("❌ Error: Audio file not found!")
    st.stop()

def process_frames():
    """Processes all images in the frames folder and stores captions & objects."""
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith((".jpg", ".png"))])
    
    if not frame_files:
        st.error("❌ No frames found in the folder.")
        st.stop()
    
    for frame_file in frame_files:
        image_path = os.path.join(frames_folder, frame_file)
        image = Image.open(image_path).convert("RGB")
        caption = generate_caption(image)
        objects = generate_labels(image)
        frame_data[frame_file] = {"caption": caption, "objects": objects}

    st.success("✅ All frames processed!")

def chat_with_groq(user_query, transcription):
    """Uses LangChain to interact with Groq's chatbot with history."""
    frame_name = next((frame for frame in frame_data if frame in user_query), None)
    
    if not frame_name:
        return "❌ No valid frame name found in your query. Please mention a frame like 'frame_1.jpg'."
    
    caption = frame_data[frame_name]["caption"]
    objects = frame_data[frame_name]["objects"]
    
    prompt = f"""
    Frame Name: {frame_name}
    Image Caption: {caption}
    Detected Objects: {objects}
    Transcribed Audio: {transcription}
    User Query: {user_query}
    
    Based on the above information, please provide a meaningful response.
    """
    
    history.append(HumanMessage(content=prompt))
    
    try:
        response = chat_model.invoke(history)
        history.append(response)
        return response.content.strip()
    except Exception as e:
        return f"❌ Groq API request failed: {str(e)}"

# Process all frames
process_frames()

# Process Audio (Speech-to-Text)
transcription = transcribe_audio(audio_path)

# Streamlit UI
st.title("🖼️ Multi-Modal AI Chatbot")
st.sidebar.header("Available Frames")
st.sidebar.write("Select a frame from the list:")

frame_list = list(frame_data.keys())
selected_frame = st.sidebar.selectbox("Choose a frame", frame_list)

st.image(os.path.join(frames_folder, selected_frame), caption=selected_frame)
st.write("### Image Caption:", frame_data[selected_frame]["caption"])
st.write("### Detected Objects:", ", ".join(frame_data[selected_frame]["objects"]))
st.write("### Transcribed Audio:", transcription)

user_query = st.text_input("💬 Ask a question about the selected frame:")

if st.button("Get AI Response"):
    if user_query:
        ai_response = chat_with_groq(user_query, transcription)
        st.write("### 🤖 AI Response:")
        st.success(ai_response)
    else:
        st.warning("Please enter a query before submitting.")
