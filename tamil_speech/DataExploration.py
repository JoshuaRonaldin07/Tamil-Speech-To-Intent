# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import os

# Load the dataset
tsv_path = "cv-corpus-24.0-2025-12-05/ta/validated.tsv"
df = pd.read_csv(tsv_path, sep='\t')

# Basic info
print("=" * 50)
print(f"Total recordings: {len(df)}")
print("=" * 50)

# Gender distribution
print("\nGender distribution:")
print(df['gender'].value_counts())

# Age distribution
print("\nAge distribution:")
print(df['age'].value_counts())

# Verified recordings
verified = df[df['up_votes'] > 0]
print(f"\nHighly verified recordings: {len(verified)}")

# Sentence length - CREATE column first, THEN use it
df['sentence_length'] = df['sentence'].apply(len)
print(f"\nAverage sentence length: {df['sentence_length'].mean():.1f} characters")
print(f"Shortest sentence: {df['sentence_length'].min()} characters")
print(f"Longest sentence: {df['sentence_length'].max()} characters")

# Look inside clips folder
clips_folder = "cv-corpus-24.0-2025-12-05/ta/clips"
all_clips = os.listdir(clips_folder)
print(f"\nTotal audio files: {len(all_clips)}")
print("\nFirst 5 filenames:")
for clip in all_clips[:5]:
    print(f"  {clip}")