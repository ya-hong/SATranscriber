from typing import *
import dataclasses
from whisper import DecodingResult
from whisper.audio import HOP_LENGTH, SAMPLE_RATE
from whisper.tokenizer import get_tokenizer, Tokenizer

if TYPE_CHECKING:
    from satranscriber.transcriber import Transcriber


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


def to_transcribe_results(transcriber: "Transcriber", results: List[DecodingResult]) -> List[TranscribeResult]:
    if len(results) == 0:
        return []
    stoken = results[0].tokens[0]

    transcribe_results = [TranscribeResult(
        text                = result.text, 
        tokens              = result.tokens,
        start               = (transcriber.mel_offset + (result.tokens[0] - stoken) * transcriber.input_stride) * HOP_LENGTH / SAMPLE_RATE,
        end                 = (transcriber.mel_offset + (result.tokens[-1] - stoken) * transcriber.input_stride) * HOP_LENGTH / SAMPLE_RATE,
        sposition           = transcriber.mel_offset + (result.tokens[0] - stoken) * transcriber.input_stride,
        tposition           = transcriber.mel_offset + (result.tokens[-1] - stoken) * transcriber.input_stride,
        avg_logprob         = result.avg_logprob,
        compression_ratio   = result.compression_ratio,
        no_speech_prob      = result.no_speech_prob,
        temprature          = result.temperature,
    ) for result in results]

    return transcribe_results