# Chatterbox Streaming TTS

This Python script enables real-time text-to-speech synthesis using the [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) model. It's designed to take sentences streamed via standard input (e.g., through a pipe), convert them to speech "on the fly," and play the audio back immediately for each sentence.

## Features

- **Streaming TTS:** Processes text input line by line.
- **On-the-fly Playback:** Audio for each sentence is played as soon as it's synthesized.
- **Chatterbox AI:** Utilizes the high-quality Chatterbox TTS model.
- **Automatic Device Detection:** Works with CUDA, MPS (Apple Silicon), or CPU.
- **Easy to Use:** Simple command-line interface.

## Prerequisites

- Python 3.12+
- `uv` package manager (see [uv installation](https://github.com/astral-sh/uv))
- A working speaker setup for audio playback.
- Internet connection for the first run (to download Chatterbox model files).

## Setup

1.  **Clone the repository (if you haven't already):**

    ```bash
    git clone https://github.com/WismutHansen/chatter_pipe.git
    ```

2.  **Install dependencies**
    ```bash
    uv sync
    ```

## Usage

Pipe text into the script, with one sentence per line. Optionally combine with [atqsm](https://github.com/WismutHansen/async-tqsm) to convert a stream of tokens into streamed sentences, so each sentences is turned into speech individually

**Example 1: Single sentence**

```bash
echo "Hello, this is a test of the Chatterbox streaming TTS." | uv run chatter_pipe.py
```

**Example 2: read from a file**

```bash
cat sentences.txt | uv run chatter_pipe.py
```

**Example 3: Pipe response sentences from an LLM in combination with atqsm**

```bash
ollama run llama3.2:latest "How are you" | atqsm | uv run chatter_pipe.py
```
