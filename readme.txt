Multi-Modal AI Surveillance Chatbot

Overview

This project is an AI-powered multi-modal surveillance chatbot that analyzes images and audio data to provide intelligent insights. It processes frames from a video, generates captions and object labels for each frame, transcribes audio, and enables users to ask queries about specific frames using natural language.

Features

Image Processing: Extracts captions and objects from images.

Speech-to-Text (STT): Converts spoken audio into text.

AI Chatbot: Uses LangChain with Groq's AI model to answer questions about the processed frames and transcriptions.

Interactive Querying: Users can ask questions about specific frames using their filenames.

Technologies Used

Python

LangChain (Groq Model)

PIL (Pillow) (Image Processing)

Florence AI (Captioning & Object Detection)

Vosk (Speech Recognition)

Installation

Prerequisites

Ensure you have the following installed:

Python 3.8+

pip (Python Package Manager)

Install Dependencies

pip install langchain_groq PIL florence_caption florence_objects vosk

Usage

1. Prepare Data

Place your image frames inside the frames folder.

Ensure you have an audio file (e.g., harvard.wav).

2. Run the System

python main.py

3. Interact with the AI Chatbot

Enter a query about a specific frame (e.g., "What is in frame_1.jpg?")

The AI chatbot will respond with relevant information based on the processed data.

Type exit to terminate the session.

Code Structure

|-- frames/                 # Folder containing extracted video frames
|-- main.py                 # Main script for processing images and audio
|-- florence_caption.py      # Generates captions for images
|-- florence_objects.py      # Detects objects in images
|-- vosk_stt.py             # Transcribes speech from audio

How It Works

Processes all images in the given folder: Extracts captions and detected objects.

Transcribes speech from the given audio file.

Waits for user queries: Accepts questions referencing frame filenames.

Generates AI responses: Uses Groq’s AI chatbot to provide meaningful insights based on the processed data.

Example Query

💬 User: What objects are in frame_1.jpg?
🤖 Groq AI Chatbot Response: The image contains a person, a bicycle, and a traffic light.

Notes

Make sure frame names are correctly referenced in queries.

The API key for Groq must be valid for the chatbot to function.

Future Improvements

Enhance object detection accuracy.

Support multiple video sources.

Implement a web-based UI for easier interaction.