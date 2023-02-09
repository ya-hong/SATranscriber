import pyaudiowpatch as pyaudio
import numpy as np
import threading
import librosa

from . import stream


def get_speaker():
    with pyaudio.PyAudio() as p:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"])
        if not default_speakers["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                """
                Try to find loopback device with same name(and [Loopback suffix]).
                Unfortunately, this is the most adequate way at the moment.
                """
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                print("Default loopback output device not found.\n\nRun `python -m pyaudiowpatch` to check available devices.\nExiting...\n")
                exit()
    return default_speakers


class Stream(stream.Stream):
    def __enter__(self):
        self.p = pyaudio.PyAudio()

        speaker = get_speaker()

        self.speaker_sr = int(speaker["defaultSampleRate"])
        self.speaker_ac = speaker["maxInputChannels"]

        self.buffer = np.ndarray((self.speaker_ac, 0), dtype=np.float32)
        self.lock = threading.Lock()

        self.stream = self.p.open(
            format=                 pyaudio.paInt16,
            channels=               self.speaker_ac,
            rate=                   self.speaker_sr,
            frames_per_buffer=      1024,
            input=                  True,
            input_device_index=     speaker["index"],
            stream_callback=        self.callback
        )

        return self

    def __exit__(self, type, value, traceback):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def read(self) -> np.ndarray:
        with self.lock:
            if self.buffer.shape[-1] == 0:
                return np.ndarray(0, np.float32)
            buffer, self.buffer = self.buffer, np.ndarray((self.speaker_ac, 0), dtype=np.float32)
        
        result = librosa.resample(
            buffer, 
            res_type="kaiser_fast", 
            orig_sr=self.speaker_sr, 
            target_sr=self.SAMPLE_RATE, 
            scale=True
        )
        """
        scale 需要为True, 要保证音量
        """
        return np.sum(result, axis=0, keepdims=False) / self.speaker_ac / 32768.0

    def callback(self, in_data, frame_count, time_info, status):
        mat = np.frombuffer(
            in_data, dtype=np.int16
        ).astype(np.float32).reshape((self.speaker_ac, -1), order="F")
        with self.lock: 
            self.buffer = np.concatenate([self.buffer, mat], axis=1)
        return (None, pyaudio.paContinue)


if __name__ == "__main__":
    import time
    with Stream() as stream:
        for i in range(1000000):
            time.sleep(1)
            print(stream.read())
