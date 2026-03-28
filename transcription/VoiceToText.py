import threading
import queue
import time
import numpy as np
import sounddevice as sd
import whisper


def Transcribe():

    # Creating model in whisper
    model = whisper.load_model("base")
    # getting the transcription from the audio file
    result = model.transcribe("output.mp3",fp16=False, language="English")

    # writing the transcription to the output file
    with open('output.txt','w') as file:
        file.write(result["text"])

Transcribe()