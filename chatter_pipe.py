#!/usr/bin/env python3

import sys
import torch
from chatterbox.tts import ChatterboxTTS
import queue
import threading

try:
    import sounddevice as sd
except ImportError:
    print(
        "The 'sounddevice' library is not installed. Please install it to play audio.",
        file=sys.stderr,
    )
    print("You can install it using: pip install sounddevice", file=sys.stderr)
    sys.exit(1)

audio_queue = queue.Queue()

def audio_playback_worker():
    while True:
        wav_tensor, sample_rate, sentence = audio_queue.get()
        print(f"Playing from queue: '{sentence}'", file=sys.stderr)
        sd.play(wav_tensor.squeeze(0).cpu().numpy(), sample_rate, blocking=True)
        audio_queue.task_done()

def main():
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA device.", file=sys.stderr)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS device (Mac M1/M2/M3/M4).", file=sys.stderr)
        # Apply patch for MPS if necessary (from Chatterbox example_for_mac.py)
        # This helps torch.load map tensors to the MPS device correctly when
        # loading model weights.
        torch_load_original = torch.load

        def patched_torch_load(*args, **kwargs):
            if "map_location" not in kwargs:
                kwargs["map_location"] = torch.device(device)
            return torch_load_original(*args, **kwargs)

        torch.load = patched_torch_load
    else:
        device = "cpu"
        print("Using CPU device.", file=sys.stderr)

    print(
        "Loading ChatterboxTTS model... This may take a moment, especially on first run (downloading models).",
        file=sys.stderr,
    )
    # Initialize the ChatterboxTTS model
    # from_pretrained will download models from Hugging Face Hub on the first run.
    # It loads a default voice.
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded. Ready to synthesize speech.", file=sys.stderr)
    print(
        'Pipe sentences to this script (one sentence per line). Example: echo "Hello world" | python this_script.py',
        file=sys.stderr,
    )
    print("Press Ctrl+D (EOF) to exit after all input is processed.", file=sys.stderr)

    # Start the playback thread
    playback_thread = threading.Thread(target=audio_playback_worker, daemon=True)
    playback_thread.start()

    try:
        for line in sys.stdin:
            sentence = line.strip()
            if not sentence:
                continue  # Skip empty lines

            print(f"\nSynthesizing: '{sentence}'", file=sys.stderr)
            # Generate audio waveform
            # The model.generate method handles text normalization (e.g., punc_norm)
            # and uses the default voice and synthesis parameters.
            # Output is a torch.Tensor of shape (1, num_samples).
            wav_tensor = model.generate(sentence)

            print("Audio queued for playback.", file=sys.stderr)
            # Play the audio using sounddevice
            # sounddevice.play expects a NumPy array.
            # .squeeze(0) removes the batch dimension.
            # .cpu().numpy() moves to CPU and converts to NumPy.
            # Instead of playing directly, put the audio data and sentence into the queue
            audio_queue.put((wav_tensor, model.sr, sentence))
            print(f"Added to queue: '{sentence}'", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nExiting due to user interruption (Ctrl+C).", file=sys.stderr)
    except EOFError:
        # This is expected when input pipe closes
        print("\nEnd of input stream.", file=sys.stderr)
    finally:
        print("Waiting for audio queue to empty...", file=sys.stderr)
        audio_queue.join()  # Wait for all items in the queue to be processed
        print("Shutting down.", file=sys.stderr)


if __name__ == "__main__":
    main()
