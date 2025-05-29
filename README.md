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

## Usage: `chatter_pipe.py`

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

## Background TTS Service: `chatter_daemon.py`

The `chatter_daemon.py` script provides a persistent, background text-to-speech service. It listens for text input on a named pipe (FIFO), synthesizes it into speech using ChatterboxTTS, and then either plays the audio directly or saves it as a WAV file. This allows other applications to easily integrate TTS capabilities by simply writing text to the named pipe.

### Features

- **Background Service:** Runs continuously after being started.
- **Named Pipe Input:** Accepts text lines from any process that can write to its named pipe.
- **Flexible Output:**
    - Play audio directly using `sounddevice`.
    - Save audio as WAV files to a specified directory using `soundfile`.
- **Concurrent Processing:** Uses a queueing system to process TTS and audio output without blocking new input from the pipe.
- **Configurable:** Pipe name, output mode, and output directory can be customized via command-line arguments.
- **Graceful Shutdown:** Handles `SIGINT` (Ctrl+C) and `SIGTERM` for proper cleanup.

### Prerequisites (in addition to base prerequisites)

- **`soundfile`:** Required if using the file output mode. You can install it via pip:
  ```bash
  uv pip install soundfile
  ```
  (Ensure `sounddevice` is also installed as per the main project setup if you intend to use playback mode.)

### Usage

1.  **Start the daemon:**
    Run the script from your terminal. It's recommended to run it in the background using `&`.
    ```bash
    uv run chatter_daemon.py &
    ```
    You can also redirect its standard error (where logs are printed) to a file:
    ```bash
    uv run chatter_daemon.py > /tmp/chatter_daemon.log 2>&1 &
    ```

2.  **Send text to the daemon:**
    Once the daemon is running, write sentences to its named pipe. By default, the pipe is located at `/tmp/chatter_fifo`.
    ```bash
    echo "Hello, daemon world!" > /tmp/chatter_fifo
    echo "This message will be spoken or saved to a file." > /tmp/chatter_fifo
    ```

### Command-Line Options

-   `--pipe-name <path>`: Specifies the path for the named pipe.
    (Default: `/tmp/chatter_fifo`)
-   `--output-mode <mode>`: Sets the audio output mode.
    -   `play`: Plays audio directly (default).
    -   `file`: Saves audio as WAV files.
-   `--output-dir <directory>`: Specifies the directory for saving WAV files when `--output-mode` is `file`.
    (Default: `./audio_output/`)
-   `--debug`: Enables verbose debug logging to standard error.

**Example: Start daemon to save files to `my_audio_clips/` using a custom pipe**
```bash
uv run chatter_daemon.py --output-mode file --output-dir my_audio_clips/ --pipe-name /tmp/my_tts_pipe &
```
Then, to send text:
```bash
echo "Save this clip." > /tmp/my_tts_pipe
```
