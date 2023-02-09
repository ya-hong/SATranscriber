from typing import *
import torch
import threading
import time

import whisper
from whisper.audio import N_FRAMES, N_MELS, N_FFT
from whisper.audio import log_mel_spectrogram
from whisper.tokenizer import get_tokenizer, Tokenizer

from .audio import Stream
# from .method.read import TranscribeResult
# from .method.read import read as read_function
from .method.transcribe import transcribe_step as transcribe_step_function
from .method.parse_result import split_decode_result, to_transcribe_results, TranscribeResult

class Transcriber:
    NON_TRANSABLE_LENGTH = 200

    def __init__(
        self,
        audio_stream: Stream,
        model: str                                  = "medium",
        task: str                                   = "transcribe",

        # transcriber arguments
        language: str                               = "English",
        temperature: Union[Tuple[float], float]     = (0, 0.2, 0.6),
        beam_size: int                              = 10,
        best_of: int                                = 10,
        
        # decode arguments 
        logprob_threshold: float				    = -1.0,
        compression_ratio_threshold: float          = 2.4,
        no_speech_threshold: float				    = 0.6,
        padding: int 							    = 200,

        fp16: bool                                  = True,
        verbose: bool                               = False,
        **kwargs
    ) -> None:

        self.model: whisper.Whisper = whisper.load_model(model, "cuda")
        self.audio_stream = audio_stream
        self.task = task

        self.language = language 
        self.temperature_list = temperature if isinstance(temperature, Iterable) else [temperature]
        self.beam_size = beam_size
        self.best_of = best_of
        
        self.logprob_threshold = logprob_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        self.no_speech_threshold = no_speech_threshold
        self.padding = padding

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
        self.try_log("EXIT!!!")
        if self.transcribe_thread.isAlive():
            self.transcribe_thread.join(timeout=1)

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
    
    def is_quality(self, result: whisper.DecodingResult):
        return result.avg_logprob > self.logprob_threshold and \
            result.compression_ratio < self.compression_ratio_threshold and \
            result.no_speech_prob < self.no_speech_threshold

    def transcribe(self):
        INIT_STEP = 3
        step = INIT_STEP
        while not self.is_exited:
            self.try_log("wait audio buffer for {}s".format(step))
            time.sleep(step) # 实际间隔为 step + process_time
            self.lock.acquire()
            try:
                self.read_audio_step()
                transcribe_step_function(self)

                if not self.is_quality(self.decode_result):
                    can_temp_up = self.try_temperature_up()
                    step /= 2
                    self.try_log("low quality transcribe result!")
                    self.try_log(self.decode_result)
                    if step < 0.1 and not can_temp_up:
                        self.try_log("drop!")
                        self.extend_offset(self.mel_buffer.shape[-1])
                        step = INIT_STEP
                    continue

                tokenizer: Tokenizer = get_tokenizer(
                    self.model.is_multilingual,
                    language=self.language, 
                    task=self.task)
                results: List[TranscribeResult] = to_transcribe_results(self, split_decode_result(tokenizer, self.decode_result))

                def is_stable(result: TranscribeResult):
                    return result.tposition < self.mel_offset + self.mel_buffer.shape[-1] - self.padding

                results = [result for result in results if is_stable(result)]
                if len(results):
                    self.output_buffer.extend(results)
                    self.extend_offset(results[-1].tposition - self.mel_offset)

                step = INIT_STEP
            except:
                self.is_exited = True
                raise
            finally:
                self.lock.release()
    
    def read_audio_step(self):
        audio = self.audio_stream.read()
        if len(audio) < N_FFT:
            return
        self.extend_mel(log_mel_spectrogram(audio))
    
    def read(self) -> List[TranscribeResult]:
        self.lock.acquire()
        try:
            buffer, self.output_buffer = self.output_buffer, []
            return buffer
        except:
            self.is_exited = True
            raise
        finally:
            self.lock.release()
    
    def try_log(self, log) -> None:
        if self.verbose:
            import pprint
            pprint.pprint(log)

