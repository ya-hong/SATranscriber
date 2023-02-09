from typing import *
import dataclasses
from whisper import DecodingResult
from whisper.audio import HOP_LENGTH, SAMPLE_RATE
from whisper.tokenizer import get_tokenizer, Tokenizer



@dataclasses.dataclass
class TranscribeResult:
    text: str
    tokens: List[int]
    start: float
    end: float
    sposition: int
    tposition: int
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    temprature: float


def split_decode_result(result: DecodingResult, tokenizer: Tokenizer) -> List[DecodingResult]:
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


def to_transcribe_results(results: List[DecodingResult], start_offset: int, input_stride: int) -> List[TranscribeResult]:
    if len(results) == 0:
        return []
    stoken = results[0].tokens[0]

    transcribe_results = [TranscribeResult(
        text                = result.text, 
        tokens              = result.tokens,
        start               = (start_offset + (result.tokens[0] - stoken) * input_stride) * HOP_LENGTH / SAMPLE_RATE,
        end                 = (start_offset + (result.tokens[-1] - stoken) * input_stride) * HOP_LENGTH / SAMPLE_RATE,
        sposition           = start_offset + (result.tokens[0] - stoken) * input_stride,
        tposition           = start_offset + (result.tokens[-1] - stoken) * input_stride,
        avg_logprob         = result.avg_logprob,
        compression_ratio   = result.compression_ratio,
        no_speech_prob      = result.no_speech_prob,
        temprature          = result.temperature,
    ) for result in results]

    return transcribe_results

