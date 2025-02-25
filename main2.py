import os
import json
import streamlit as st
from multiprocessing import Pool
from PIL import Image
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from florence_caption import generate_caption
from florence_objects import generate_labels
from vosk_stt import transcribe_audio


# ✅ Initialize LangChain Groq Model
chat_model = ChatGroq(
    temperature=0.7,  # Adjust creativity
    api_key="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"
)

# ✅ Chat history
history = [
    SystemMessage(content="You are an AI assistant analyzing multiple images and audio.")
]

# ✅ Paths
frames_folder = "/home/vmukti/multi_aii/frames/"
audio_path = "/home/vmukti/multi_aii/harvard.wav"
json_cache = "frame_data.json"

# ✅ Process frames using multiprocessing
def process_single_frame(frame_file):
    """Processes a single frame (Caption + Objects)."""
    image_path = os.path.join(frames_folder, frame_file)
    image = Image.open(image_path).convert("RGB")
    caption = generate_caption(image)
    objects = generate_labels(image)
    return frame_file, {"caption": caption, "objects": objects}

def process_frames():
    """Processes frames using multiprocessing for faster execution."""
    if os.path.exists(json_cache):
        with open(json_cache, "r") as f:
            return json.load(f)

    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith((".jpg", ".png"))])
    if not frame_files:
        st.error("❌ No frames found in the folder.")
        st.stop()

    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_single_frame, frame_files)

    frame_data = dict(results)

    with open(json_cache, "w") as f:
        json.dump(frame_data, f)

    return frame_data

# ✅ Load frames data (cached)
frame_data = process_frames()

# ✅ Extract frame name from user query
def extract_frame_name(user_query):
    for frame_name in frame_data.keys():
        if frame_name in user_query:
            return frame_name
    return None

# ✅ Chat with AI
def chat_with_groq(user_query, transcription):
    """Interacts with Groq AI based on frame data & transcription."""
    frame_name = extract_frame_name(user_query)

    if not frame_name:
        return "❌ No valid frame name found in your query. Please mention a frame like 'frame_1.jpg'."

    caption = frame_data[frame_name]["caption"]
    objects = frame_data[frame_name]["objects"]

    # Construct AI input prompt
    prompt = f"""
    Frame Name: {frame_name}
    Image Caption: {caption}
    Detected Objects: {objects}
    Transcribed Audio: {transcription}
    User Query: {user_query}

    Based on the above information, please provide a meaningful response.
    """

    st.info(f"🔄 Processing query about {frame_name}...")

    try:
        history.append(HumanMessage(content=prompt))
        response = chat_model.invoke(history)
        history.append(response)
        return response.content.strip()
    except Exception as e:
        return f"❌ Groq API request failed: {str(e)}"

# ✅ Streamlit UI
st.title("📷 Multi-AI Image & Audio Analyzer")
st.sidebar.header("⚙ Settings")
st.sidebar.write(f"Frames Folder: `{frames_folder}`")
st.sidebar.write(f"Audio File: `{audio_path}`")

# ✅ Frame selection
selected_frame = st.selectbox("🖼 Select a Frame", list(frame_data.keys()))

# ✅ Display Image, Caption, and Objects
if selected_frame:
    st.image(os.path.join(frames_folder, selected_frame), caption=selected_frame, use_column_width=True)
    st.subheader("📝 Caption")
    st.write(frame_data[selected_frame]["caption"])
    st.subheader("🎯 Detected Objects")
    st.write(", ".join(frame_data[selected_frame]["objects"]))

# ✅ Audio Transcription
st.subheader("🎙 Transcribed Audio")
transcription = transcribe_audio(audio_path)
st.write(transcription)

# ✅ Chat Input
user_query = st.text_input("💬 Ask about this frame (e.g., 'What is in frame_1.jpg?')")

# ✅ Chat Output
if st.button("🤖 Get AI Response"):
    if user_query:
        response = chat_with_groq(user_query, transcription)
        st.subheader("🤖 AI Response")
        st.write(response)
    else:
        st.warning("⚠ Please enter a question.")
