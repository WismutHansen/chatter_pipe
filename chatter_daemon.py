#!/usr/bin/env python3
"""
Chatter Daemon: Text-to-Speech Background Service via Named Pipe

This script runs as a background service (daemon) that listens for text input
on a named pipe (FIFO), synthesizes it into speech using ChatterboxTTS,
and then either plays the audio directly or saves it as a WAV file.

Running the Script:
To run the daemon in the background:
  python chatter_daemon.py &

To run with specific options (e.g., file output):
  python chatter_daemon.py --output-mode file --output-dir ./my_audio_files &

Named Pipe (FIFO):
- The script creates a named pipe (FIFO) to receive text input.
- Default pipe location: /tmp/chatter_fifo
- To send text to the daemon, write to this pipe. For example:
    echo "Hello world, this is a test." > /tmp/chatter_fifo
    echo "Another sentence for TTS." > /tmp/chatter_fifo
  Each line sent to the pipe is processed as a separate sentence.

Command-Line Arguments:
  --pipe-name PATH      : Sets the path for the named pipe.
                          (Default: /tmp/chatter_fifo)
  --output-mode {play,file} : Specifies the output behavior.
                          'play': Plays audio directly (default).
                          'file': Saves audio as WAV files.
  --output-dir DIR      : Directory to save WAV files if --output-mode is 'file'.
                          (Default: ./audio_output/)
  --debug               : Enables verbose debug logging to stderr.

Key Dependencies:
- chatterbox.tts (from Resemble AI's Chatterbox) for speech synthesis.
- torch (PyTorch) as a dependency for Chatterbox.
- sounddevice for audio playback (if output-mode is 'play').
- soundfile for saving WAV files (if output-mode is 'file').
  (Install with: pip install chatterbox-tts torch sounddevice soundfile)

Shutdown:
- The daemon can be shut down using Ctrl+C (SIGINT) if running in the foreground.
- If running in the background, use the `kill` command with the process ID (PID):
    kill <PID_OF_DAEMON> (sends SIGTERM)
- The script is designed to catch SIGTERM and SIGINT signals to perform a
  graceful shutdown, which includes:
    - Processing any remaining audio tasks in the queue.
    - Closing the audio worker thread.
    - Removing the named pipe from the filesystem.
"""

import sys
import os
import time
import argparse
import torch
from chatterbox.tts import ChatterboxTTS
import queue
import threading
import sounddevice as sd
import soundfile as sf
import signal

# --- Constants ---
DEFAULT_PIPE_NAME = "/tmp/chatter_fifo"

# --- Global Queue for Audio Playback/Saving ---
# Similar to chatter_pipe.py, for handling audio processing off the main read loop
audio_task_queue = queue.Queue()

# --- Global Refs for Signal Handler Cleanup ---
_pipe_name_ref = None
_audio_task_queue_ref = None
_audio_thread_ref = None
_args_debug_ref = False # Default

# --- Cleanup Function ---
def perform_cleanup():
    global _pipe_name_ref, _audio_task_queue_ref, _audio_thread_ref, _args_debug_ref
    print("Cleaning up resources...", file=sys.stderr)
    
    if _audio_task_queue_ref:
        _audio_task_queue_ref.put(None) # Sentinel
    if _audio_thread_ref:
        _audio_thread_ref.join(timeout=5) 
        if _args_debug_ref and _audio_thread_ref.is_alive():
            print("Audio thread did not terminate cleanly.", file=sys.stderr)

    if _pipe_name_ref and os.path.exists(_pipe_name_ref):
        try:
            os.remove(_pipe_name_ref)
            if _args_debug_ref: print(f"Removed named pipe: {_pipe_name_ref}", file=sys.stderr)
        except OSError as e:
            print(f"Error removing named pipe {_pipe_name_ref} during cleanup: {e}", file=sys.stderr)
    print("Cleanup complete.", file=sys.stderr) # Changed message slightly to differentiate from signal-specific one

# --- Signal Handler ---
def signal_handler(signum, frame):
    print(f"Received signal {signal.Signals(signum).name}. Shutting down...", file=sys.stderr)
    perform_cleanup()
    sys.exit(0)

# --- Audio Playback/Saving Worker ---
def audio_worker(output_mode="play", output_dir="audio_out"):
    # output_mode can be "play" or "file"
    # output_dir is where files are saved if mode is "file"
    if output_mode == "file" and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}", file=sys.stderr)

    while True:
        task = audio_task_queue.get()
        if task is None: # Sentinel for shutdown
            audio_task_queue.task_done()
            break

        wav_tensor, sample_rate, sentence_text = task
        
        print(f"Processing audio for: '{sentence_text}'", file=sys.stderr)

        if output_mode == "play":
            try:
                sd.play(wav_tensor.squeeze(0).cpu().numpy(), sample_rate, blocking=True)
                print(f"Finished playing: '{sentence_text}'", file=sys.stderr)
            except Exception as e:
                print(f"Error playing audio: {e}", file=sys.stderr)
        elif output_mode == "file":
            output_filename = "" # Initialize to ensure it's defined for error messages
            try:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                # Create a short, safe prefix from the sentence
                filename_prefix = "".join(filter(str.isalnum, sentence_text[:30])).lower().replace(" ", "_") or "speech"
                if not filename_prefix: # Handle cases where sentence might be all non-alnum
                    filename_prefix = "speech"
                output_filename = os.path.join(output_dir, f"{timestamp}_{filename_prefix}.wav")
                
                sf.write(output_filename, wav_tensor.squeeze(0).cpu().numpy(), sample_rate)
                print(f"Saved audio to: {output_filename}", file=sys.stderr)
            except NameError: # If sf (soundfile) was not imported
                print("Error: The 'soundfile' library is required for file output but not installed. Please install it (e.g., pip install soundfile).", file=sys.stderr)
            except Exception as e:
                if output_filename: # If filename was generated
                    print(f"Error saving audio to file {output_filename}: {e}", file=sys.stderr)
                else: # If error occurred before filename generation
                    print(f"Error saving audio to file: {e}", file=sys.stderr)
        
        audio_task_queue.task_done()

# --- Main Application Logic ---
def main():
    parser = argparse.ArgumentParser(description="Background TTS daemon using a named pipe.")
    parser.add_argument("--pipe-name", type=str, default=DEFAULT_PIPE_NAME,
                        help=f"Name/path of the FIFO named pipe (default: {DEFAULT_PIPE_NAME})")
    parser.add_argument("--output-mode", type=str, choices=["play", "file"], default="play",
                        help="Mode for audio output: 'play' directly or save to 'file' (default: play)")
    parser.add_argument("--output-dir", type=str, default="audio_output",
                        help="Directory to save audio files if output-mode is 'file' (default: audio_output/)")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing.")
    args = parser.parse_args()

    global _pipe_name_ref, _audio_task_queue_ref, _audio_thread_ref, _args_debug_ref
    _args_debug_ref = args.debug # Set global debug ref

    # --- Device Detection (copied from chatter_pipe.py) ---
    if torch.cuda.is_available():
        device = "cuda"
        if args.debug: print("Using CUDA device.", file=sys.stderr)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        if args.debug: print("Using MPS device (Mac M1/M2/M3/M4).", file=sys.stderr)
        torch_load_original = torch.load
        def patched_torch_load(*args_load, **kwargs_load):
            if "map_location" not in kwargs_load:
                kwargs_load["map_location"] = torch.device(device)
            return torch_load_original(*args_load, **kwargs_load)
        torch.load = patched_torch_load
    else:
        device = "cpu"
        if args.debug: print("Using CPU device.", file=sys.stderr)

    # --- Model Loading (copied from chatter_pipe.py) ---
    if args.debug: print("Loading ChatterboxTTS model...", file=sys.stderr)
    model = ChatterboxTTS.from_pretrained(device=device)
    if args.debug: print("Model loaded.", file=sys.stderr)

    # --- Start Audio Worker Thread ---
    # The worker needs to know the output mode and directory
    audio_thread = threading.Thread(target=audio_worker, args=(args.output_mode, args.output_dir), daemon=True)
    _audio_thread_ref = audio_thread # Set global ref
    _audio_task_queue_ref = audio_task_queue # Set global ref
    audio_thread.start()
    if args.debug: print(f"Audio worker thread started. Mode: {args.output_mode}", file=sys.stderr)

    # --- Named Pipe Creation & Handling ---
    pipe_name = args.pipe_name
    _pipe_name_ref = pipe_name # Set global ref
    if not os.path.exists(pipe_name):
        try:
            os.mkfifo(pipe_name)
            if args.debug: print(f"Created named pipe: {pipe_name}", file=sys.stderr)
        except OSError as e:
            print(f"Failed to create named pipe: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        if args.debug: print(f"Using existing named pipe: {pipe_name}", file=sys.stderr)


    print(f"Daemon started. Listening on named pipe: {pipe_name}", file=sys.stderr)
    print(f"Output mode: {args.output_mode}", file=sys.stderr)
    if args.output_mode == "file":
        print(f"Output directory for files: {args.output_dir}", file=sys.stderr)

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    if args.debug: print("Registered signal handlers for SIGTERM and SIGINT.", file=sys.stderr)
    
    try:
        while True:
            if args.debug: print(f"Opening pipe {pipe_name} for reading...", file=sys.stderr)
            with open(pipe_name, "r") as fifo:
                if args.debug: print(f"Pipe {pipe_name} opened. Waiting for input...", file=sys.stderr)
                while True: # Keep reading from the currently open pipe
                    line = fifo.readline()
                    if not line: # Pipe closed by writer or empty line if writer closes and reopens
                        if args.debug: print("Writer closed the pipe or sent empty line. Re-opening.", file=sys.stderr)
                        break # Break from inner loop to reopen pipe

                    sentence = line.strip()
                    if not sentence:
                        continue

                    if args.debug: print(f"Received from pipe: '{sentence}'", file=sys.stderr)
                    
                    # --- TTS Inference ---
                    print(f"Synthesizing: '{sentence}'", file=sys.stderr)
                    wav_tensor = model.generate(sentence)
                    
                    # --- Enqueue for Playback/Saving ---
                    audio_task_queue.put((wav_tensor, model.sr, sentence))
                    print(f"Queued for {args.output_mode}: '{sentence}'", file=sys.stderr)

    except Exception as e: # Catch other exceptions that might occur in the main loop
        print(f"An unexpected error occurred in main loop: {e}", file=sys.stderr)
    finally:
        # perform_cleanup is called by signal_handler on SIGINT/SIGTERM,
        # but call it here too for normal exit or other exceptions.
        # If called by signal handler first, this might do less or nothing if resources are gone.
        if args.debug: print("Main loop/try block exited, entering finally for cleanup.", file=sys.stderr)
        perform_cleanup() 
        # Note: If signal_handler already called sys.exit(), this part of finally might not be reached.
        # However, if the loop exits normally or due to an Exception other than a handled signal,
        # this ensures cleanup.

if __name__ == "__main__":
    # Check for sounddevice dependency early, similar to chatter_pipe.py
    try:
        import sounddevice as sd_check
        if hasattr(sd_check, 'check_output_settings'): # Basic check
            pass 
    except ImportError:
        print(
            "The 'sounddevice' library is not installed. Please install it to play audio.",
            file=sys.stderr,
        )
        print("You can install it using: pip install sounddevice", file=sys.stderr)
        # Exiting here if sounddevice is needed for play mode and not installed.
        # The script might still be usable for 'file' output mode if we make sounddevice optional later.
        # For now, let's assume it's a general requirement.
        # TODO: Make sounddevice import conditional if output_mode is 'file' and no playback is ever intended.
        sys.exit(1)
        
    main()
