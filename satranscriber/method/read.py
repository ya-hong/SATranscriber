from typing import *

if TYPE_CHECKING:
    from satranscriber.transcriber import Transcriber

import dataclasses
import whisper
from whisper import DecodingResult
from whisper.audio import N_FRAMES, HOP_LENGTH, SAMPLE_RATE
from whisper.tokenizer import get_tokenizer, Tokenizer



@dataclasses.dataclass
class TranscribeResult:
    text: str
    tokens: List[int]
    start: float
    end: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    temprature: float


@dataclasses.dataclass(frozen=True)
class ReadRequest:
    logprob_threshold: float				= -1.0
    compression_ratio_threshold: float		= 2.4
    no_speech_threshold: float				= 0.6
    padding: int 							= 200

    def is_qulity(self, result: Union[DecodingResult, TranscribeResult]) -> bool:
        return result.avg_logprob > self.logprob_threshold and \
            result.compression_ratio < self.compression_ratio_threshold and \
            result.no_speech_prob < self.no_speech_threshold


def split_decode_result(tokenizer: Tokenizer, result: DecodingResult) -> List[DecodingResult]:
    result_list = list()
    tokens = result.tokens

    min_start_token = tokens[0]

    stoken_idx = 0
    while stoken_idx < len(tokens):
        if stoken_idx == len(tokens) - 1:
            break
        
        for ttoken_idx in range(stoken_idx + 1, len(tokens)):
            if tokens[ttoken_idx] > min_start_token:
                break
        
        setence_tokens = tokens[stoken_idx:ttoken_idx+1]

        result_list.append(DecodingResult(**{
            **dataclasses.asdict(result),
            "tokens": setence_tokens,
            "text": tokenizer.decode(setence_tokens[1:-1]),
        }))

        stoken_idx = ttoken_idx + 1

    return result_list


def read(
    self: "Transcriber",
    r: ReadRequest,
) -> List[TranscribeResult]:
    """
    一次read最好至少输出一句话。如果什么也不输出需要调整策略
    """

    if self.decode_result is None or not r.is_qulity(self.decode_result):
        length = self.mel_buffer.shape[-1]
        if length > self.NON_TRANSABLE_LENGTH and not self.try_temperature_up() and length > N_FRAMES:
            # 达到可转录长度，但无法转录 -> 升温， 但温度已到顶 -> 处理时间过长 -> 丢弃
            self.extend_offset(min(length, N_FRAMES))
        return []

    stoken = self.decode_result.tokens[0]

    def is_stable(result: DecodingResult):
        return (result.tokens[-1] - stoken) * self.input_stride < self.mel_buffer.shape[-1] - r.padding

    tokenizer: Tokenizer = get_tokenizer(
        self.model.is_multilingual,
        language=self.language, 
        task=self.task)

    results = split_decode_result(tokenizer, self.decode_result)
    
    stable_results = [TranscribeResult(
        text=       result.text, 
        tokens=     result.tokens,
        start=      (self.mel_offset + (result.tokens[0] - stoken) * self.input_stride) * HOP_LENGTH / SAMPLE_RATE,
        end=        (self.mel_offset + (result.tokens[-1] - stoken) * self.input_stride) * HOP_LENGTH / SAMPLE_RATE,
        avg_logprob=        result.avg_logprob,
        compression_ratio=  result.compression_ratio,
        no_speech_prob=     result.no_speech_prob,
        temprature=         result.temperature,
    ) for result in results if is_stable(result)]

    if len(stable_results) == 0:
        return []

    ttoken = stable_results[-1].tokens[-1]
    gap = (ttoken - stoken) * self.input_stride
    self.extend_offset(gap)

    return stable_results

