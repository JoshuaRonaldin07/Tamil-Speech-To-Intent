import sys
import io
sys.stdout= io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
tsv_path= "cv-corpus-24.0-2025-12-05/ta/validated.tsv"
df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
first_row= df.iloc[0]
print("=" *50)
print (f"Audio file: {first_row['path']}")
print(f"Sentence :{first_row['sentence']}")
print(f"Up votes :{first_row['up_votes']}")
clips_folder="cv-corpus-24.0-2025-12-05/ta/clips"
audio_path=f"{clips_folder}/{first_row['path']}"
y,sr = librosa.load(audio_path, sr=16000)
print(f"\nSample Rate:{sr}Hz")
print(f"Total Samples:{len(y)}")
print(f"Duration:{len(y)/sr:.2f}seconds")
print(f"\nFirst 10 numbers of audio")
print (y[:10])
plt.figure(figsize=(12,4))
librosa.display.waveshow(y,sr=sr)
plt.title(f"Tamil Speech Waveform\n{first_row['sentence']}")
plt.xlabel("Time(seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("waveform.png")
plt.show()
print("\nWaveform saved as waveform.png")
fig,axes=plt.subplots(2,1, figsize=(12,8))
plt.sca(axes[0])
librosa.display.waveshow(y, sr=sr, ax=axes[0])
axes[0].set_title("waveform-Amplitude over Time")
axes[0].set_xlabel("time in seconds")
axes[0].set_ylabel("Amplitude")

D=librosa.amplitude_to_db(
    abs(librosa.stft(y)),
    ref=np.max
)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
axes[1].set_title("Spectrogram-Frequency over Time")
axes[1].set_xlabel("Time(seconds)")
axes[1].set_ylabel("frequency(Hz)")
plt.colorbar(axes[1].collections[0], ax=axes[1],format="%+2.0f dB")
plt.tight_layout()
plt.savefig("spectrogram.png")
plt.show()
print("Saved as spectrogram.png!")