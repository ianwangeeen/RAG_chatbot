from helper_functions import query_handler
from faster_whisper import WhisperModel
import pyaudio
import os
import wave
import numpy as np
import time
import re
import pyttsx3 

GREEN =  '\033[32m' # Green Text
WHITE = '\033[37m' # white text 

# List of placeholder phrases to filter
PLACEHOLDER_STRINGS = [
    "Thanks for watching!", 
    "Thank you.", 
    "Thank you very much."
]

def is_silence(audio_chunk, threshold=500):
    """
    Determines if an audio chunk is silent based on a threshold.

    Args:
        audio_chunk (bytes): The audio data chunk.
        threshold (int): Amplitude threshold to classify silence.

    Returns:
        bool: True if the chunk is silent, False otherwise.
    """
    audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
    return np.abs(audio_array).mean() < threshold



def record_chunk(p, stream, file_path, silence_duration=2):
    """
    Records audio until a specified silence duration is detected.

    Args:
        p (PyAudio): The PyAudio instance.
        stream (Stream): The audio input stream.
        file_path (str): Path to save the recorded file.
        silence_duration (int): Time in seconds to stop recording on silence.

    Returns:
        bool: True if a valid chunk is recorded, False otherwise.
    """
    frames = []
    silence_start = None
    print(GREEN + "Recording..." + WHITE)

    while True:
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

        if is_silence(data):
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > silence_duration:
                break
        else:
            silence_start = None

    if len(frames) > 0:
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        wf.close()
        return True

    return False



def filter_transcription(text):
    """
    Filters out placeholder text and meaningless filler phrases.

    Args:
        text (str): Raw transcription from the model.

    Returns:
        str: Cleaned transcription.
    """
    # Remove placeholder strings
    for placeholder in PLACEHOLDER_STRINGS:
        text = text.replace(placeholder, "")

    # Optional: Remove any non-alphanumeric sequences like double spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Return cleaned text
    return text



def transcribe_chunk(model, file_path):
    """
    Transcribes a chunk of audio to text using the WhisperModel.

    Args:
        model (WhisperModel): An instance of the WhisperModel.
        file_path (str): Path to the audio file to transcribe.

    Returns:
        str: The transcribed text.
    """
    try:
        # Transcribe the audio file
        segments, _ = model.transcribe(file_path)

        # Concatenate transcribed text from all segments
        transcription = " ".join([segment.text for segment in segments])

        filtered_text = filter_transcription(transcription)
        return filtered_text
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "[Transcription failed]"
    


def record_until_trigger(p, stream, model, trigger_phrase="alpha"):
    """
    Continuously records audio and transcribes it, sending data to LLM when the trigger phrase is spoken.
    """
    frames = []
    accumulated_transcript = ""  # Accumulates text between responses

    print(GREEN + "Recording... (say '" + trigger_phrase + "' to process)" + WHITE)

    while True:
        try:
            # Continuously read audio data
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)

            # Check if data is silent; process audio if sufficient length
            if len(frames) > 50:  # Adjust this buffer size as needed
                file_path = "temp_chunk.wav"
                with wave.open(file_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(16000)
                    wf.writeframes(b''.join(frames))
                frames = []  # Reset the audio frame buffer

                # Transcribe the audio chunk
                transcription = transcribe_chunk(model, file_path)
                os.remove(file_path)  # Clean up temp file

                if transcription.strip():
                    print(GREEN + "You said: " + transcription + WHITE)
                    accumulated_transcript += transcription + " "

                    # Check for trigger phrase
                    if trigger_phrase.lower() in transcription.lower():
                        print(GREEN + "Processing accumulated input..." + WHITE)
                        response = query_handler.generate_response(accumulated_transcript.replace(trigger_phrase, "").strip())
                        print(GREEN + "Llama3 Response: " + response + WHITE)
                        speak_text(response)
                        accumulated_transcript = ""  # Reset transcript after processing

        except KeyboardInterrupt:
            print("\nStopping recording...")
            break



def speak_text(text):
    """Converts text to speech and plays it."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()



def main():
    model_size = "medium.en"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    accumulated_transcript = ""

    # try:
    #     while True:
    #         chunk_file = "temp_chunk.wav"
    #         if record_chunk(p, stream, chunk_file, silence_duration=2):
    #             transcription = transcribe_chunk(model, chunk_file)
    #             os.remove(chunk_file)

    #             if transcription.strip():
    #                 print(GREEN + "You said: " + transcription + WHITE)

    #                 if "You can answer now." in transcription:
    #                     response = query_handler.generate_response(transcription)
    #                     print(GREEN + "Llama3 Response: " + response + WHITE)
    #                     accumulated_transcript = ""

    #                 else:
    #                     accumulated_transcript += transcription + " "

    # except KeyboardInterrupt:
    #     print("Stopping...")
    #     with open("log.txt", "w") as log_file:
    #         log_file.write(accumulated_transcript)

    try:
        record_until_trigger(p, stream, model)

    finally:
        print("Cleaning up resources...")
        stream.stop_stream()
        stream.close()
        p.terminate

if __name__ == "__main__":
    main()