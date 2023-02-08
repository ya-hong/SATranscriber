from typing import *

if TYPE_CHECKING:
    from satranscriber.transcriber import Transcriber

import dataclasses
import whisper
from whisper.audio import N_FRAMES, HOP_LENGTH, SAMPLE_RATE


@dataclasses.dataclass(frozen=True)
class ReadRequest:
    logprob_threshold: float				= -1.0
    compression_ratio_threshold: float		= 2.4
    no_speech_threshold: float				= 0.6
    length_threshold: int 					= 1
    padding: int 							= 200

    # def __post_init__(self):
    # 	self.logprob_threshold = self.logprob_threshold or 						1.0
    # 	self.compression_ratio_threshold = self.compression_ratio_threshold or 	2.4
    # 	self.no_speech_threshold = self.no_speech_threshold or 					0.6
    # 	self.length_threshold = self.length_threshold or 						3
    # 	self.padding = self.padding or 											300


@dataclasses.dataclass
class TranscribeResult:
    text: str
    start: float
    end: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    temprature: float


def read(
    self: "Transcriber",
    r: ReadRequest,
) -> List[TranscribeResult]:
    """
    一次read最好至少输出一句话。如果什么也不输出需要调整策略
    """

    if len(self.result_buffer) == 0:
        return []

    stoken = self.result_buffer[0].tokens[0]

    def qualify(result: whisper.DecodingResult) -> bool:
        return result.avg_logprob > r.logprob_threshold \
            and result.compression_ratio < r.compression_ratio_threshold \
            and len(result.text) >= r.length_threshold \
            and (result.tokens[-1] - stoken) * self.input_stride < min(N_FRAMES, self.mel_buffer.shape[-1]) - r.padding \
            # and result.no_speech_prob < no_speech_threshold

    if self.verbose:
        import pprint
        if self.result_buffer[0].temperature > 0:
            pprint.pprint(self.result_buffer)

    results = [result for result in self.result_buffer if qualify(result)]

    if len(results) == 0:
        if self.mel_buffer.shape[-1] > self.NON_TRANSABLE_LENGTH and not self.try_temperature_up():
            # 达到可转录长度，但无法转录 -> 升温， 但温度已到顶 -> 丢弃
            self.extend_offset(min(self.mel_buffer.shape[-1], N_FRAMES))
        return []
    
    ret = [TranscribeResult(
        text=       result.text, 
        start=      (self.mel_offset + (result.tokens[0] - stoken) * self.input_stride) * HOP_LENGTH / SAMPLE_RATE,
        end=        (self.mel_offset + (result.tokens[-1] - stoken) * self.input_stride) * HOP_LENGTH / SAMPLE_RATE,
        avg_logprob=        result.avg_logprob,
        compression_ratio=  result.compression_ratio,
        no_speech_prob=     result.no_speech_prob,
        temprature=         result.temperature,
    ) for result in results]

    ttoken = results[-1].tokens[-1]
    gap = (ttoken - stoken) * self.input_stride
    self.extend_offset(gap)

    return ret

