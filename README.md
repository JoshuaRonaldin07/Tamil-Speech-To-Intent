#  Tamil Speech-to-Intent System
### Building an End-to-End Speech NLP Pipeline for an Underrepresented Language

---

##  Project Overview

Voice interfaces — Siri, Alexa, Google Assistant — work well for English, Mandarin, and Spanish. For the **80+ million Tamil speakers** in the world, the technology either does not exist or performs so poorly it is unusable.

This project builds a **Speech-to-Intent pipeline** for Tamil — a complete end-to-end system that takes raw audio and maps it to structured meaning. The core research question is:

> *"How does ASR performance degrade as we move from high-resource (English) to low-resource (Tamil) conditions, and where in the pipeline does the degradation happen?"*

---

##  Pipeline Architecture

```
 Raw Tamil Audio
        ↓
  [ Audio Processing ]     → waveform, spectrogram analysis
        ↓
  [ ASR — Whisper ]        → speech → Tamil text
        ↓
  [ Intent Detection ]     → text → structured intent
        ↓
  [ Evaluation ]           → WER, accuracy metrics
        ↓
Research Findings
```

---

##  Project Structure

```
tamil_speech_project/
│
├── lesson1.py          # Tamil Unicode & text exploration
├── lesson1_3.py        # Dataset loading & exploration
├── lesson1_4.py        # Audio visualization (waveform + spectrogram)
├── lesson2_1.py        # Whisper ASR + WER measurement
│
├── outputs/
│   ├── waveform.png        # Tamil speech waveform visualization
│   └── spectrogram.png     # Tamil speech spectrogram visualization
│
└── README.md
```

---

##  Dataset

**Mozilla Common Voice — Tamil (CV Corpus 24.0)**
- 🔗 https://commonvoice.mozilla.org/en/datasets
- **117,289** validated Tamil recordings
- **Real speakers** from Tamil Nadu, Sri Lanka, and diaspora communities
- Includes speaker metadata: age, gender, accent

| Split | Recordings |
|-------|-----------|
| Train | ~80,000 |
| Test  | ~14,000 |
| Dev   | ~14,000 |

---

##  Key Research Findings (So Far)

### Finding 1 — Encoding Problem
Standard Windows tools use `cp1252` encoding which has no Tamil support. This is one systemic reason Tamil voice AI is underdeveloped — even basic infrastructure assumes Western languages.

### Finding 2 — Script Complexity
Tamil uses an **agglutinative script** with invisible modifier characters (pulli ்). The word **வணக்கம்** (hello) appears as 4 visual shapes but contains 7 Unicode code points. Standard `len()` functions are unreliable for Tamil NLP.

### Finding 3 — Waveform Analysis
Tamil speech has a **syllable-timed rhythm** — each syllable gets roughly equal duration. This contrasts with stress-timed languages like English and has implications for acoustic modelling.

### Finding 4 — ASR Performance Gap

| Metric | English (Whisper medium) | Tamil (Whisper medium) |
|--------|--------------------------|------------------------|
| Average WER | ~3-5% | **66.8%** |
| Best case WER | ~1% | **0.0%** |
| Worst case WER | ~10% | **111.1%** |

> Tamil is approximately **13x harder** for Whisper than English.

### Finding 5 — Error Type Analysis
The dominant error type for Tamil is **word splitting** — Whisper breaks long compound Tamil words into multiple shorter incorrect words:

```
Ground truth:  மக்களிடையே        (1 word)
Whisper said:  மக்கள் அடையும்    (2 words — WRONG)

Ground truth:  மனோநிலைகளைத்      (1 word)
Whisper said:  மண நிலைகளை        (2 words — WRONG)
```

This is caused by Tamil's agglutinative morphology — words are formed by joining morphemes together, creating very long tokens that Whisper (trained mostly on English) does not expect.

### Finding 6 — Sentence Length Correlation
Shorter sentences yield lower WER. Recording 18 (5 words: *வடையின் சுவையோ விடேன் விடேன் என்றது*) achieved **0% WER** — perfect transcription. Recording 1 (9 complex compound words) achieved **111.1% WER**.

---

##  Setup & Installation

### Prerequisites
- Python 3.9+
- VS Code (recommended)
- ~3GB free disk space (for dataset + models)

### Install Dependencies

```bash
pip install librosa matplotlib numpy soundfile pandas
pip install openai-whisper
pip install torch
```

### Download Dataset
1. Go to https://commonvoice.mozilla.org/en/datasets
2. Select **Tamil** and download **CV Corpus 24.0**
3. Extract to your project folder

### Run

```bash
# Explore the dataset
python lesson1_3.py

# Visualize audio
python lesson1_4.py

# Run ASR evaluation
python lesson2_1.py
```


---

##  Why This Matters

Tamil is spoken by **80+ million people** as a first language — more than French or German. Yet virtually no commercial voice AI system handles Tamil reliably. This project is part of a broader effort to understand and close the performance gap between high-resource and low-resource language AI systems.

The methodology developed here (pipeline construction + systematic WER analysis) can be applied to other underrepresented languages: Konkani, Yoruba, Bengali, and hundreds more.

---

