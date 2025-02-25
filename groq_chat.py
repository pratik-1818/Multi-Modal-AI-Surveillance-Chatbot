import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage


chat_model = ChatGroq(
    temperature=0.7,  
    api_key = "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"
)

def chat_with_groq(user_query, caption, objects, transcription):
    """Uses LangChain to interact with Groq's chatbot."""
    
    prompt = f"""
    Image Caption: {caption}
    Detected Objects: {objects}
    Transcribed Audio: {transcription}
    User Query: {user_query}

    Based on the above information, please provide a meaningful response.
    """
    
    print("🔄 Sending request to Groq API via LangChain...")

    try:
        response = chat_model.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    
    except Exception as e:
        return f"❌ Groq API request failed: {str(e)}"
