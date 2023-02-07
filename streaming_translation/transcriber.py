from typing import *
import torch
import threading
import torch

import whisper
from whisper.audio import N_FRAMES, N_MELS, N_FFT
from whisper.audio import log_mel_spectrogram

from .audio import Stream
from .method.read import TranscribeResult, ReadRequest
from .method.read import read as read_function
from .method.transcribe import transcribe_step as transcribe_step_function


class Transcriber:
    def __init__(
        self,
        audio_stream: Stream,
        model: str =                                "medium",
        language: str =         					"English",
        temperature: Union[Tuple[float], float] = 	(0, 0.2, 0.6),
        task: str =             					"transcribe",
        fp16: bool =								True,
        verbose: bool =         					True,
    ) -> None:

        self.model: whisper.Whisper = whisper.load_model(model, "cuda")
        self.audio_stream = audio_stream
        self.language = language 
        self.temperature_list = temperature if isinstance(temperature, Iterable) else [temperature]
        self.task = task
        self.dtype = torch.float16 if fp16 else torch.float32
        self.verbose = verbose

        from whisper.utils import exact_div
        self.input_stride = exact_div(
            N_FRAMES, self.model.dims.n_audio_ctx
        )
    
    def __enter__(self):
        self.lock = threading.RLock()
        self.is_exited = False
        self.try_read = False

        self.temperature_idx = 0

        self.mel_offset: int = 0
        self.mel_buffer: torch.Tensor = torch.zeros((N_MELS, 0))

        self.result_buffer: List[whisper.DecodingResult] = list()
        self.output_buffer: List[TranscribeResult] = list()

        self.transcribe_thread = threading.Thread(target=self.transcribe)
        self.transcribe_thread.start()
        return self
    
    def __exit__(self, type, value, traceback):
        self.is_exited = True
        self.transcribe_thread.join()

    def temperature(self) -> float:
        return self.temperature_list[self.temperature_idx]
    
    def try_temperature_up(self) -> bool:
        if self.temperature_idx + 1 < len(self.temperature_list):
            self.temperature_idx += 1
            return True
        return False
    
    def try_temperature_down(self) -> bool:
        if self.temperature_idx > 0:
            self.temperature_idx -= 1
            return True
        return False

    def extend_mel(self, mel):
        self.mel_buffer = torch.cat([self.mel_buffer, mel], dim=1)

    def extend_offset(self, offset):
        """
        将最旧的offset位mel谱设置为不会再访问
        同时将温度置0
        """
        self.mel_offset += offset
        self.mel_buffer = self.mel_buffer[:, offset:]
        self.temperature_idx = 0
    
    def transcribe(self):
        while not self.is_exited:
            if self.try_read:
                continue

            self.lock.acquire()
            try:
                self.read_step()
                if self.mel_buffer.shape[-1] > N_FRAMES:
                    self.output_buffer.extend(read_function(self, ReadRequest(padding=500)))
                transcribe_step_function(self)
            except:
                self.is_exited = True
                raise
            finally:
                self.lock.release()
    
    def read_step(self):
        audio = self.audio_stream.read()
        if len(audio) < N_FFT:
            return
        self.extend_mel(log_mel_spectrogram(audio))
    
    def read(self, r: ReadRequest = ReadRequest()) -> List[TranscribeResult]:

        def qualify(result: TranscribeResult):
            return result.avg_logprob > r.logprob_threshold and \
                result.compression_ratio < r.compression_ratio_threshold and \
                len(result.text) >= r.length_threshold
        
        self.try_read = True
        self.lock.acquire()
        try:
            buffer = [result for result in self.output_buffer if qualify(result)]
            self.output_buffer = []
            return buffer + read_function(self, r)
        except:
            self.is_exited = True
            raise
        finally:
            self.try_read = False
            self.lock.release()


if __name__ == "__main__":
    from .audio import speaker
    import time
    from .text2text import baidu, youdao

    btranslator = baidu.Translator("jp")
    ytranslator = youdao.translator("ja")

    with speaker.Stream() as stream, Transcriber(stream, language="ja", fp16=True, task="transcribe", temperature=0) as transcriber:
        while True:
            time.sleep(2)
            results = transcriber.read(ReadRequest(logprob_threshold=-0.5, padding=200))
            print([result.text for result in results])
            for result in results:
                print("=====>", btranslator.translate(result.text))
                print("=====>", ytranslator.translate(result.text))
            
                