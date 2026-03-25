import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding = 'utf-8')

import whisper
import pandas as pd
import librosa
import numpy as np

print("list Whisper medium mode...")
print("(First Time = doownloads - 1.5GB, be patient!)")
model = whisper.load_model("medium")
print("Model Loaded!")

tsv_path="cv-corpus-24.0-2025-12-05/ta/validated.tsv"
df= pd.read_csv(tsv_path, sep='\t', low_memory=False)

clips_folder = "cv-corpus-24.0-2025-12-05/ta/clips"

first_row = df.iloc[0]
audio_path = f"{clips_folder}/{first_row['path']}"
ground_truth = first_row['sentence']

print("\n" + "="*50)
print(f"Audio file : {first_row['path']}")
print(f"Ground Truth : {ground_truth}")
print('='*50)
print("\nLoading Audio")
y, sr = librosa.load(audio_path, sr=16000)
y=y.astype(np.float32)
print("Transcribing...")
result = model.transcribe(y, language="ta")
transcription = result["text"]
print("\n"+"="*50)
print(f"Ground Truth : {ground_truth}")
print(f"Whisper Said : {transcription}")
print("="*50)

def calculate_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    ref_len = len(ref_words)
    hyp_len = len(hyp_words)
    d = np.zeros((ref_len + 1, hyp_len + 1), dtype=int)
    for i in range(ref_len + 1):
        d[i][0]=i
    for j in range(hyp_len + 1):
        d[0][j]=j 
    for i in range(1,ref_len +1):
        for j in range(1,hyp_len +1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] +1 
                delete = d[i-1][j] +1
                insert = d[i][j-1] +1
                d[i][j] = min(substitute, delete, insert)
    errors = d[ref_len][hyp_len]
    wer = errors / ref_len *100
    return wer
wer = calculate_wer(ground_truth, transcription)
print("\n" + "=" * 50)
print("RESEARCH FINDING:")
if wer < 20:
    print(f"WER {wer:.1f}% → Good performance!")
elif wer < 50:
    print(f"WER {wer:.1f}% → Moderate performance")
else:
    print(f"WER {wer:.1f}% → Poor performance")
print("(English Whisper WER is typically 3-5%)")
print("=" * 50)
# ─── Test on 20 recordings ────────────────────────────────
print("\nTesting on 20 recordings...")
print("=" * 50)

results = []  # store each result here

for i in range(20):
    row = df.iloc[i]
    audio_path = f"{clips_folder}/{row['path']}"
    ground_truth = row['sentence']
    
    # Load and transcribe
    y, sr = librosa.load(audio_path, sr=16000)
    y = y.astype(np.float32)
    result = model.transcribe(y, language="ta")
    transcription = result["text"]
    
    # Calculate WER
    wer = calculate_wer(ground_truth, transcription)
    results.append(wer)
    
    print(f"Recording {i+1:2d}: WER = {wer:.1f}%")

# Average WER
avg_wer = sum(results) / len(results)
print("\n" + "=" * 50)
print(f"Average WER across 20 recordings: {avg_wer:.1f}%")
print(f"Best  WER: {min(results):.1f}%")
print(f"Worst WER: {max(results):.1f}%")
print("Compare: English Whisper WER ≈ 3-5%")
print("=" * 50) 
print("\nLooking at Recording 18:")
row = df.iloc[17]   # iloc[17] = 18th row (counting from 0)
print(f"Sentence : {row['sentence']}")
print(f"Words    : {len(row['sentence'].split())}")

# Compare with Recording 1
print("\nLooking at Recording 1:")
row = df.iloc[0]
print(f"Sentence : {row['sentence']}")
print(f"Words    : {len(row['sentence'].split())}")