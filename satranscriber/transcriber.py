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
    NON_TRANSABLE_LENGTH = 200

    def __init__(
        self,
        audio_stream: Stream,
        model: str =                                "medium",
        task: str =             					"transcribe",
        language: str =         					"English",
        temperature: Union[Tuple[float], float] = 	(0, 0.2, 0.6),
        beam_size: int =                            10,
        best_of: int =                              10,
        fp16: bool =								True,
        verbose: bool =         					False,
        **kwargs
    ) -> None:

        self.model: whisper.Whisper = whisper.load_model(model, "cuda")
        self.audio_stream = audio_stream
        self.task = task
        self.language = language 
        self.temperature_list = temperature if isinstance(temperature, Iterable) else [temperature]
        self.beam_size = beam_size
        self.best_of = best_of
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

        self.decode_result: whisper.DecodingResult = None
        self.output_buffer: List[TranscribeResult] = list()

        self.transcribe_thread = threading.Thread(target=self.transcribe)
        self.transcribe_thread.start()
        return self
    
    def __exit__(self, type, value, traceback):
        self.is_exited = True
        print("EXIT!!!!")
        self.transcribe_thread.join()

    def temperature(self) -> float:
        return self.temperature_list[self.temperature_idx]
    
    def try_temperature_up(self) -> bool:
        """
        升温。在一般情况加不应该使温度降低, 除非转录的音频发生变化。
        所以没有相应的try_temperature_down方法, 而是在extend_offset的同时使温度置0
        """
        if self.temperature_idx + 1 < len(self.temperature_list):
            self.temperature_idx += 1
            return True
        return False

    def extend_mel(self, mel):
        self.mel_buffer = torch.cat([self.mel_buffer, mel], dim=1)

    def extend_offset(self, offset):
        """
        将最旧的offset位mel谱设置为不会再访问
        由于接下来转录的音频发生了变化，所以：
            将温度置0
            将decode_result置为None
        """
        self.mel_offset += offset
        self.mel_buffer = self.mel_buffer[:, offset:]
        self.temperature_idx = 0
        self.decode_result = None
    
    def transcribe(self):
        while not self.is_exited:
            if self.try_read:
                continue

            self.lock.acquire()
            try:
                if self.mel_buffer.shape[-1] > N_FRAMES:
                    self.output_buffer.extend(read_function(self, ReadRequest(padding=500)))
                self.read_step()
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
        self.try_read = True
        self.lock.acquire()
        try:
            buffer = [result for result in self.output_buffer if r.is_qulity(result)]
            self.output_buffer = []
            return buffer + read_function(self, r)
        except:
            self.is_exited = True
            raise
        finally:
            self.try_read = False
            self.lock.release()

