from streaming_translation.stream import speaker
from .byte2np import load_audio
import whisper
import torch
from datetime import datetime, timedelta


class WhisperStream:
    PARSE_TIMEOUT = 60

    def __init__(self, language, stream=speaker.Stream()) -> None:
        self.stream = stream
        self.language = language

    def __enter__(self):
        self.stream.__enter__()
        self.model = whisper.load_model("medium", device="cuda")
        self.parse_time = 0
        return self

    def convert(self):
        audio = load_audio(self.stream.get_buffer(time_offset=self.parse_time - 5))
        result = self.model.transcribe(audio, language=self.language, fp16=True)

        # print(result)
        # print()
        segments = result["segments"]
        sentents = []
        # print(len(segments))
        for i in range(len(segments) - 2, 0, -1):
            # print(segments[i]["start"], self.parse_time)
            if segments[i]["start"] > self.parse_time:
                sentents.append(segments[i]["text"])
            else:
                break
        
        # print(len(sentents))
        sentents = sentents[::-1]
        # print()
        if len(sentents):
            self.parse_time = segments[-2]["start"]

        return sentents

    def __exit__(self):
        self.stream.__exit__()
