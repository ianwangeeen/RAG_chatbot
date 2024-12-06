from helper_functions import query_handler
from faster_whisper import WhisperModel
import pyaudio
import os
import wave
import numpy as np
import time
import re
import pyttsx3 
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch


GREEN =  '\033[32m' # Green Text
WHITE = '\033[37m' # white text 
model_path = "D:\\PersonalProjs\\NLP With LLM\\fine tuning speech to text\\results"

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
    
    # Remove any non-alphanumeric sequences like double spaces
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)

    # Remove placeholder strings
    for placeholder in PLACEHOLDER_STRINGS:
        text = text.replace(placeholder, "")
    
    # Return cleaned text
    return re.sub(r'\s+', ' ', text).strip()



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
    

def process_audio(frames, processor, model, device):
    """
    Process and transcribe audio frames using WhisperProcessor and WhisperForConditionalGeneration.

    Args:
        frames (list): List of audio frames (NumPy arrays).
        processor (WhisperProcessor): Whisper processor for feature extraction.
        model (WhisperForConditionalGeneration): Whisper model for transcription.
        device (torch.device): Device to run the model on.

    Returns:
        str: Transcribed text from the audio.
    """
    # Merge and normalize audio frames
    audio_data = np.concatenate(frames, axis=0).astype(np.float32)
    audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize to -1 to 1

    # Extract mel spectrogram features
    inputs = processor.feature_extractor(
        audio_data,
        sampling_rate=16000,
        return_tensors="pt"
    )

    # Extract mel features and pad explicitly to 3000
    input_features = inputs.input_features.to(device)
    mel_length = input_features.size(-1)
    
    if mel_length < 3000:
        pad_length = 3000 - mel_length
        input_features = torch.nn.functional.pad(
            input_features,
            (0, pad_length),
            value=processor.feature_extractor.padding_value
        )
    elif mel_length > 3000:
        input_features = input_features[:, :, :3000]  # Truncate to 3000 if larger
    attention_mask = (input_features != processor.feature_extractor.padding_value).long()


    # Generate transcription using the model
    with torch.no_grad():
        # forced_language_ids = processor.tokenizer.get_lang_id("en")
        if attention_mask is not None:
            predicted_ids = model.generate(input_features, attention_mask=attention_mask, pad_token_id=processor.tokenizer.pad_token_id)
        else:
            predicted_ids = model.generate(input_features, pad_token_id=processor.tokenizer.pad_token_id)
        transcription = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

    filtered_transcription = filter_transcription(transcription)
    
    return filtered_transcription



def record_until_trigger(p, stream, processor, model, trigger_phrase="finish"):
    """
    Continuously records audio and transcribes it, sending data to LLM when the trigger phrase is spoken.
    """
    frames = []
    accumulated_transcript = ""  
    device = model.device

    print(GREEN + "Recording... (say '" + trigger_phrase + "' to process)" + WHITE)

    while True:
        try:
            # Continuously read audio data
            data = stream.read(1024, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            frames.append(audio_data)

            # Check if data is silent; process audio if sufficient length
            if len(frames) > 50:  # Adjust this buffer size as needed
                transcription = process_audio(frames, processor, model, device)
                frames = []  # Reset the frame buffer

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
    # model_size = "medium.en"
    # model = WhisperModel(model_size, device="cuda", compute_type="float16")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)

    # Set the model to evaluation mode
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    # accumulated_transcript = ""

    try:
        record_until_trigger(p, stream, processor, model)

    finally:
        print("Cleaning up resources...")
        stream.stop_stream()
        stream.close()
        p.terminate

if __name__ == "__main__":
    main()