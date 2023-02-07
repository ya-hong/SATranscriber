from typing import *

if TYPE_CHECKING:
    from streaming_translation.transcriber import Transcriber

import torch
import whisper
from whisper.audio import pad_or_trim, N_FRAMES
from whisper import DecodingOptions, DecodingResult
from whisper.tokenizer import get_tokenizer, Tokenizer
import dataclasses


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


def update_results(self: "Transcriber", result_list: List[DecodingResult]):
    MAX_COMPRESSION_RATIO = 2.4

    def better(A: DecodingResult, B: DecodingResult) -> bool:
        if (A.compression_ratio < MAX_COMPRESSION_RATIO) == (B.compression_ratio < MAX_COMPRESSION_RATIO):
            return A.avg_logprob > B.avg_logprob
        return A.compression_ratio < MAX_COMPRESSION_RATIO
    
    def update(idx: int, result: DecodingResult):
        if idx < len(self.result_buffer):
            self.result_buffer[idx] = result
        else:
            self.result_buffer.append(result)

    """
    注意temperature>0时结果是非确定的, 两次transcribe会有很大差异所以不好逐个比较
    """

    if self.temperature() == 0:
        for idx, result in enumerate(result_list):
            if idx >= len(self.result_buffer) or better(result, self.result_buffer[idx]):
                update(idx, result)
    else:
        if len(self.result_buffer) == 0 or better(result_list[0], self.result_buffer[0]):
            self.result_buffer = result_list


def transcribe_step(self: "Transcriber"):
    """
    刷新buffer内前30s音频的转录结果
    """

    model: whisper.Whisper = self.model

    mel = pad_or_trim(self.mel_buffer, N_FRAMES).to(model.device).to(self.dtype)

    options = DecodingOptions(
        task=               self.task,
        language=           self.language,
        temperature=        self.temperature(),
        fp16=               True if self.dtype == torch.float16 else False,
        beam_size=          15 if self.temperature() == 0 else None,
        best_of=            None if self.temperature() == 0 else 15,
    )

    result: DecodingResult = model.decode(mel, options)

    tokenizer: Tokenizer = get_tokenizer(
        self.model.is_multilingual,
        language=self.language, 
        task=self.task)
    
    if result.avg_logprob < -1 or result.compression_ratio > 2.4:
        self.try_temperature_up()
        return
    
    self.try_temperature_down()
    result_list = split_decode_result(tokenizer, result)
    self.result_buffer = result_list