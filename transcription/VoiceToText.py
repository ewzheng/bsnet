"""
Real-time voice-to-text using faster_whisper.
Listens indefinitely, printing and resetting the transcription every 100 chars.
Each 100-char chunk is passed to on_chunk() so it can be fed into a BERT model.

Install deps:
    pip install faster-whisper pyaudio webrtcvad-wheels
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix: multiple OpenMP runtimes on Windows

import io
import wave
import collections
import pyaudio
import webrtcvad
from faster_whisper import WhisperModel

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_SIZE         = "base"    # tiny=fastest | base | small | medium | large-v3
DEVICE             = "cpu"     # "cpu" or "cuda"
COMPUTE_TYPE       = "int8"  # int8 (CPU) | float16 (GPU)

SAMPLE_RATE        = 16000    # Hz – Whisper expects 16 kHz
FRAME_MS           = 30       # VAD frame size: must be 10 | 20 | 30 ms
FRAME_SAMPLES      = int(SAMPLE_RATE * FRAME_MS / 1000)   # samples per frame
FRAME_BYTES        = FRAME_SAMPLES * 2                     # 16-bit = 2 bytes/sample
VAD_AGGRESSIVENESS = 2        # 0=permissive … 3=aggressive (filters more noise)

# Silence threshold: consecutive silent frames before we treat utterance as done
SILENCE_FRAMES     = int(300 / FRAME_MS)   # ~600 ms of silence → transcribe
PREROLL_FRAMES     = int(200 / FRAME_MS)   # ~200 ms kept before speech starts


# ─────────────────────────────────────────────────────────────────────────────


def pcm_to_wav_bytes(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def transcribe(model: WhisperModel, pcm: bytes) -> str:
    wav = pcm_to_wav_bytes(pcm)
    segments, _ = model.transcribe(
        io.BytesIO(wav),
        language="en",
        beam_size=1,                    # greedy – 3-4x faster than beam_size=5
        best_of=1,
        temperature=0.0,                # deterministic, no random sampling
        condition_on_previous_text=False,  # prevents hallucination loops
        vad_filter=False,               # we handle VAD ourselves
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )
    return " ".join(seg.text.strip() for seg in segments)


def listen():
    """
    Listen indefinitely. Every time 100+ characters accumulate, the chunk is
    printed and yielded. Resets after each yield. Stops on Ctrl+C.

    Usage:
        for chunk in listen():
            # chunk is a 100+ char string you can use however you like
            pass
    """
    print(f"Loading Whisper '{MODEL_SIZE}' on {DEVICE} …")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("Model ready. Speak — transcription triggers after silence.\n")

    vad    = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    pa     = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAME_SAMPLES,
    )

    accumulated  = ""
    ring         = collections.deque(maxlen=PREROLL_FRAMES)
    voiced       = []
    silent_count = 0
    speaking     = False

    print("Listening … (Ctrl+C to stop)\n")

    try:
        while True:
            frame = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
            if len(frame) < FRAME_BYTES:
                continue

            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if is_speech:
                if not speaking:
                    speaking = True
                    voiced   = list(ring) + [frame]
                else:
                    voiced.append(frame)
                silent_count = 0

            else:
                ring.append(frame)
                if speaking:
                    silent_count += 1
                    voiced.append(frame)

                    if silent_count >= SILENCE_FRAMES:
                        speaking     = False
                        silent_count = 0
                        pcm          = b"".join(voiced)
                        voiced       = []

                        text = transcribe(model, pcm).strip()
                        if not text:
                            continue

                        accumulated += (" " if accumulated else "") + text
                        print(f"[{len(accumulated):>4} chars] {text}")

                        if len(accumulated) >= 100:
                            chunk = accumulated.strip()
                            print(f"\n── chunk ──\n{chunk}\n───────────\n")
                            yield chunk
                            accumulated = ""

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    for chunk in listen():
        pass  # chunks are already printed inside listen()
