import whisperx
import gc 

from whisperx.SubtitlesProcessor import SubtitlesProcessor

device = "cpu" 
audio_file = "/Users/ellekartchner/Desktop/WhisperX/IPHY4650-4-1.mp3"
batch_size = 16 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

language = "en"

print("why")

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("medium", device, compute_type=compute_type, language=language)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

print("help")

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, language=language, batch_size=batch_size)
print(result["segments"]) # before alignment

print("rah")

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

#### print(result["segments"]) # after alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

print("hi")

# All variable names below apart from `result` are settings that can be exposed to the user.
subtitles_proccessor = SubtitlesProcessor(
    result["segments"],
    "en", # str, two letter code to identify the language
    max_line_length=90, # int, around 100 has been working for me
    min_char_length_splitter=60, # int, around 70 has been working for me
    is_vtt= False, # bool, true for vtt, false for srt format
)
subtitles_proccessor.save("IPHY4650-4-1.srt", advanced_splitting=True) # output_path is a str with your desired filename


# # 3. Assign speaker labels
# diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

# # add min/max number of speakers if known
# diarize_segments = diarize_model(audio)
# # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

# result = whisperx.assign_word_speakers(diarize_segments, result)
# print(diarize_segments)
# print(result["segments"]) # segments are now assigned speaker IDs