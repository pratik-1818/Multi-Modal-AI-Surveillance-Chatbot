from vosk import Model, KaldiRecognizer
import wave
import json

def transcribe_audio(audio_path):
    model = Model("/home/vmukti/multi_aii/vosk-model-small-en-us-0.15")  
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())

    transcript = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            transcript += result["text"] + " "

    return transcript.strip()

if __name__ == "__main__":
    audio_file = "/home/vmukti/multi_aii/harvard.wav"  
    transcript = transcribe_audio(audio_file)
    print("Transcription:", transcript)
