import gradio as gr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
from pydub.generators import Sine
import io
import ffmpeg
import subprocess
import torch
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
import tempfile
import numpy as np
import pandas as pd
import re
import scipy.io.wavfile

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=8,
    device=device,
)


arabic_bad_Words = pd.read_csv("arabic_bad_words_dataset.csv")
english_bad_Words = pd.read_csv("english_bad_words_dataset.csv")


# def load_audio(file: str, sr: int = 16000):
#     try:
#         # This reads the audio from the video file without creating a separate audio file
#         command = [
#             "ffmpeg",
#             "-i", file,
#             "-f", "s16le",
#             "-acodec", "pcm_s16le",
#             "-ar", str(sr),
#             "-ac", "1",
#             "-"
#         ]
        
#         out = subprocess.run(command, capture_output=True, check=True).stdout
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    
#     return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def clean_english_word(word):
    cleaned_text = re.sub(r'^[\s\W_]+|[\s\W_]+$', '', word)
    return cleaned_text.lower()

def clean_arabic_word(word):
    pattern = r'[^\u0600-\u06FF]'
    cleaned_word = re.sub(pattern, '', word)
    return cleaned_word

def classifier(word_list_with_timestamp, language):

    foul_words = []
    negative_timestamps = []

    if language == "English":
        list_to_search = set(english_bad_Words["words"])
        for item in word_list_with_timestamp:
            word = clean_english_word(item['text'])
            if word.lower() in list_to_search:
                foul_words.append(word)
                negative_timestamps.append(item['timestamp'])
    else:
        list_to_search = list(arabic_bad_Words["words"])
        for item in word_list_with_timestamp:
            word = clean_arabic_word(item['text'])
            for word_in_list in list_to_search:
                if word_in_list == word:
                    foul_words.append(word)
                    negative_timestamps.append(item['timestamp'])   
        
    return [foul_words, negative_timestamps]

def generate_bleep(duration_ms, frequency=1000):
    sine_wave = Sine(frequency)
    bleep = sine_wave.to_audio_segment(duration=duration_ms)
    return bleep

def mute_audio_range(audio_filepath, ranges, bleep_frequency=800):
    audio = AudioSegment.from_file(audio_filepath)
    for range in ranges:
        start_time = range[0]
        end_time = range[-1]
        start_ms = start_time * 1000  # pydub works with milliseconds
        end_ms = end_time * 1000
        duration_ms = end_ms - start_ms
        
        # Generate the bleep sound
        bleep_sound = generate_bleep(duration_ms, bleep_frequency)
        
        # Combine the original audio with the bleep sound
        audio = audio[:start_ms] + bleep_sound + audio[end_ms:]

    return audio

def resample_audio(audio_segment, target_sample_rate=16000):
    return audio_segment.set_frame_rate(target_sample_rate).set_channels(1).set_sample_width(2)

def format_output_to_list(data):
    formatted_list = "\n".join([f"{item['timestamp'][0]}s - {item['timestamp'][1]}s \t : {item['text']}" for item in data])
    return formatted_list

def transcribe_audio(input_audio, audio_language, task, timestamp_type):
    if input_audio is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    if timestamp_type == "sentence":
        timestamp_type = True
    else:
        timestamp_type = "word"

    output = pipe(input_audio, return_timestamps=timestamp_type, generate_kwargs={"task": task})
    text = output['text']

    timestamps = format_output_to_list(output['chunks'])

    foul_words, negative_timestamps = classifier(output['chunks'], audio_language)
    foul_words = ", ".join(foul_words)


    audio_output = mute_audio_range(input_audio, negative_timestamps)

    # Resample the output audio to 16kHz
    audio_output = resample_audio(audio_output, 16000)

    # Save the output audio to a BytesIO object
    output_buffer = io.BytesIO()
    audio_output.export(output_buffer, format="wav")
    output_buffer.seek(0)
    
    # Read the audio data from the BytesIO buffer
    sample_rate = audio_output.frame_rate
    audio_data = np.frombuffer(output_buffer.read(), dtype=np.int16)

    return  [text, timestamps, foul_words, (sample_rate, audio_data)]


def transcribe_video(input_video, video_language, task, timestamp_type):
    # Load the video file
    video = VideoFileClip(input_video)

    # Extract the audio
    audio = video.audio

    # Create a temporary file to save the audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        audio.write_audiofile(temp_audio_file.name, codec='pcm_s16le')

    # Load the audio file into an AudioSegment
    audio_segment = AudioSegment.from_file(temp_audio_file.name, format="wav")

    # Ensure the audio is mono
    if audio_segment.channels > 1:
        audio_segment = audio_segment.set_channels(1)

    # Save the mono audio to a temporary file
    mono_temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_segment.export(mono_temp_audio_file.name, format="wav")

    # Save the mono audio to a file
    extracted_audio_path = "extracted_audio_mono.mp3"
    audio_segment.export(extracted_audio_path, format="mp3")

    # Load the audio as a numpy array
    # sample_rate, audio_array = scipy.io.wavfile.read(mono_temp_audio_file.name)

    output = pipe(extracted_audio_path, return_timestamps=timestamp_type, generate_kwargs={"task": task})
    text = output['text']

    timestamps = format_output_to_list(output['chunks'])

    foul_words, negative_timestamps = classifier(output['chunks'], video_language)
    foul_words = ", ".join(foul_words)


    audio_output = mute_audio_range(extracted_audio_path, negative_timestamps)

    # Resample the output audio to 16kHz
    audio_output = resample_audio(audio_output, 16000)

    # Save the output audio to a BytesIO object
    output_buffer = io.BytesIO()
    audio_output.export(output_buffer, format="wav")
    output_buffer.seek(0)
    
    # Read the audio data from the BytesIO buffer
    sample_rate = audio_output.frame_rate
    audio_data = np.frombuffer(output_buffer.read(), dtype=np.int16)

    # Save the processed NumPy array to a WAV file
    processed_audio_path = "processed_audio.wav"
    scipy.io.wavfile.write(processed_audio_path, sample_rate, audio_data)

    # Load the processed audio into a moviepy AudioFileClip
    processed_audio = AudioFileClip(processed_audio_path)

    # Set the audio of the video to the processed audio
    final_video = video.set_audio(processed_audio)

    # Save the final video with the combined audio
    final_video_path = "final_video_with_processed_audio.mp4"
    final_video.write_videofile(final_video_path, codec="libx264", audio_codec="aac")


    return [text, timestamps, foul_words, final_video_path]


examples = [
        ["arabic_english_audios/audios/arabic_audio_11.mp3", 'Arabic', 'transcribe', 'word'],
        ["arabic_english_audios/audios/arabic_audio_12.mp3", 'Arabic', 'transcribe', 'word'],
        ["arabic_english_audios/audios/arabic_audio_13.mp3", 'Arabic', 'transcribe', 'word'],
    
        ["arabic_english_audios/audios/english_audio_18.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_19.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_20.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_21.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_22.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_23.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_24.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_25.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_26.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_27.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_28.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_29.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_30.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_31.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_32.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_33.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_34.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_35.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_36.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_37.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_38.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_39.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_40.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_41.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_42.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_43.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_44.mp3", 'English', 'transcribe', 'word'],
        ["arabic_english_audios/audios/english_audio_45.mp3", 'English', 'transcribe', 'word'],
    ]

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.HTML("<h2 style='text-align: center;'>Transcribing Audio with Timestamps using whisper-large-v3</h2>")
    # gr.Markdown("")
    with gr.Tab("Audio"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources=["upload", 'microphone'], type="filepath", label="Audio file")
                audio_language = gr.Radio(["Arabic", "English"], label="Audio Language")
                audio_task = gr.Radio(["transcribe", "translate"], label="Task")
                audio_timestamp_type = gr.Radio(["sentence", "word"], label="Timestamp Type")
                with gr.Row():
                    audio_clear_button = gr.ClearButton(value="Clear")
                    audio_submit_button = gr.Button("Submit", variant="primary", )

            with gr.Column():
                audio_transcript_output = gr.Text(label="Transcript")
                audio_timestamp_output = gr.Text(label="Timestamps")
                audio_foul_words = gr.Text(label="Foul Words")
                output_audio = gr.Audio(label="Output Audio", type="numpy")

        examples = gr.Examples(examples, inputs=[audio_input, audio_language, audio_task, audio_timestamp_type], outputs=[audio_transcript_output, audio_timestamp_output,  audio_foul_words, output_audio], fn=transcribe_audio, examples_per_page=50,  cache_examples=False)

        audio_submit_button.click(fn=transcribe_audio, inputs=[audio_input, audio_language, audio_task, audio_timestamp_type], outputs=[audio_transcript_output, audio_timestamp_output, audio_foul_words, output_audio])
        audio_clear_button.add([audio_input, audio_language, audio_task, audio_timestamp_type, audio_transcript_output, audio_timestamp_output, audio_foul_words, output_audio])


    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(sources=["upload", 'webcam'], label="Video file")
                video_language = gr.Radio(["Arabic", "English"], label="Video Language")
                video_task = gr.Radio(["transcribe", "translate"], label="Task")
                video_timestamp_type = gr.Radio(["sentence", "word"], label="Timestamp Type")
                with gr.Row():
                    video_clear_button = gr.ClearButton(value="Clear")
                    video_submit_button = gr.Button("Submit", variant="primary", )

            with gr.Column():
                video_transcript_output = gr.Text(label="Transcript")
                video_timestamp_output = gr.Text(label="Timestamps")
                video_foul_words = gr.Text(label="Foul Words")
                output_video = gr.Video(label="Output Video")
                # output_video = gr.Audio(label="Output Audio", type="numpy")


        video_submit_button.click(fn=transcribe_video, inputs=[video_input, video_language, video_task, video_timestamp_type], outputs=[video_transcript_output, video_timestamp_output, video_foul_words, output_video])
        video_clear_button.add([video_input, video_language, video_task, video_timestamp_type, video_transcript_output, video_timestamp_output, video_foul_words, output_video])



if __name__ == "__main__":
    demo.launch()
