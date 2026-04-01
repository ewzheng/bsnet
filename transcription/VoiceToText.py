import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # hates duplicate libraries being installed, so gets around it
os.environ["OMP_NUM_THREADS"] = "1"  # fixes slow loading on AMD
from whisper_live.server import TranscriptionServer
from whisper_live.client import TranscriptionClient
import threading
import time

# starts the server for transcription
def start_server():
    server = TranscriptionServer()
    server.run(host="0.0.0.0", port=9090, backend="faster_whisper")

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

# sleeps until everything is ready
time.sleep(2)

# sets up the trancsription host
transcription = TranscriptionClient(
    host="localhost",
    port=9090,lang="en",
    model="tiny",use_vad=True)

# defining buffers
messages = []
word_buffer = []
file = open("output.txt", "w", encoding="utf-8")

# writes the trancription to the file
def on_transcription(text):
    messages.append(text)
    print(f"[Transcribed]: {text}")
    file.write(text + "\n")
    file.flush()

# helper method for the transcription
def transcript_daemon():
    last_seen_index = 0

    # wait until client is connected
    while transcription.client is None:
        time.sleep(0.5)

    while True:
        time.sleep(0.5)
        # attempts to get the transcription from the host
        try:
            transcript = transcription.client.transcript or []
        except Exception:
            transcript = []

        new_segments = transcript[last_seen_index:]

        # gets the transcription text since whisper stores the text in a dictionary for no reason
        if new_segments:
            last_seen_index = len(transcript)
            for segment in new_segments:
                text = segment.get('text', '') if isinstance(segment, dict) else str(segment)
                text = text.strip()
                print(f"[Watcher found]: {text}") # debug text
                word_buffer.extend(text.split())
            
        # writes the text to the file once 100 words have been reached
        if len(word_buffer) >= 100:
            chunk = " ".join(word_buffer[:100])
            del word_buffer[:100]
            on_transcription(chunk)

# polls the transcript_daemon so that it can get the text live
watcher_thread = threading.Thread(target=transcript_daemon, daemon=True)
watcher_thread.start()

print("Running... press Ctrl+C to stop")

try:
    # starts the transcription
    transcription()
finally:
    file.close()
    print("complete")