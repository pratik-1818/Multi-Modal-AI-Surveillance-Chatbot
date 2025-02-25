import os
from PIL import Image
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from florence_caption import generate_caption
from florence_objects import generate_labels
from vosk_stt import transcribe_audio
import re  # For extracting frame names from user queries

# ✅ Initialize LangChain Groq Model
chat_model = ChatGroq(
    temperature=0.7,  # Adjust creativity
    api_key="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"
)

# ✅ Maintain chat history
history = [
    SystemMessage(content="You are an AI assistant analyzing multiple images and audio.")
]

# ✅ Store processed frame data (Dictionary: {filename → {caption, objectsS}})
frame_data = {}

def process_frames(frames_folder):
    """Processes all images in the frames folder and stores captions & objects."""
    
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith((".jpg", ".png"))])
    
    if not frame_files:
        print("❌ No frames found in the folder.")
        exit()

    print("🔄 Processing all frames...")

    for frame_file in frame_files:
        image_path = os.path.join(frames_folder, frame_file)

        print(f"🖼️ Processing Image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        caption = generate_caption(image)
        objects = generate_labels(image)

        # Store data using filename as key
        frame_data[frame_file] = {"caption": caption, "objects": objects}

    print("\n✅ All frames processed! You can now ask about any frame in your question.")

def extract_frame_name(user_query):
    """Extracts frame name from user query if present."""
    for frame_name in frame_data.keys():
        if frame_name in user_query:
            return frame_name
    return None

def chat_with_groq(user_query, transcription):
    """Uses LangChain to interact with Groq's chatbot with history."""

    frame_name = extract_frame_name(user_query)

    if not frame_name:
        return "❌ No valid frame name found in your query. Please mention a frame like 'frame_1.jpg'."

    caption = frame_data[frame_name]["caption"]
    objects = frame_data[frame_name]["objects"]

    # Constructing the user prompt
    prompt = f"""
    Frame Name: {frame_name}
    Image Caption: {caption}
    Detected Objects: {objects}
    Transcribed Audio: {transcription}
    User Query: {user_query}

    Based on the above information, please provide a meaningful response.
    """

    print(f"🔄 Processing query about {frame_name}...")

    try:
        # Append user input to history
        history.append(HumanMessage(content=prompt))

        # Get response
        response = chat_model.invoke(history)

        # Append AI response to history
        history.append(response)

        return response.content.strip()
    
    except Exception as e:
        return f"❌ Groq API request failed: {str(e)}"

if __name__ == "__main__":
    # Folder containing all frames
    frames_folder = "/home/vmukti/multi_aii/frames/"
    audio_path = "/home/vmukti/multi_aii/harvard.wav"

    # Ensure the folder exists
    if not os.path.exists(frames_folder):
        print("❌ Error: Frames folder not found!")
        exit()

    # Ensure audio file exists
    if not os.path.exists(audio_path):
        print("❌ Error: Audio file not found!")
        exit()

    # Process all frames first and store data
    process_frames(frames_folder)

    # Process Audio (Speech-to-Text)
    transcription = transcribe_audio(audio_path)

    while True:
        # User Query (Mention Frame Name Directly)
        user_query = input("\n💬 Ask a question about any frame (e.g., 'What is in frame_1.jpg?') or type 'exit': ").strip()

        if user_query.lower() == "exit":
            print("👋 Exiting chat. Goodbye!")
            break

        # Get AI Response
        ai_response = chat_with_groq(user_query, transcription)

        print("\n🤖 Groq AI Response:", ai_response)

        print("\n🔄 You can ask about another frame!")