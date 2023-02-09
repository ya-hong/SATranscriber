from typing import *

import torch
import whisper
from whisper.audio import pad_or_trim, N_FRAMES
from whisper import DecodingOptions, DecodingResult



# def update_results(self: "Transcriber", result_list: List[DecodingResult]):
#     MAX_COMPRESSION_RATIO = 2.4

#     def better(A: DecodingResult, B: DecodingResult) -> bool:
#         if (A.compression_ratio < MAX_COMPRESSION_RATIO) == (B.compression_ratio < MAX_COMPRESSION_RATIO):
#             return A.avg_logprob > B.avg_logprob
#         return A.compression_ratio < MAX_COMPRESSION_RATIO
    
#     def update(idx: int, result: DecodingResult):
#         if idx < len(self.result_buffer):
#             self.result_buffer[idx] = result
#         else:
#             self.result_buffer.append(result)

#     """
#     注意temperature>0时结果是非确定的, 两次transcribe会有很大差异所以不好逐个比较
#     """

#     if self.temperature() == 0:
#         for idx, result in enumerate(result_list):
#             if idx >= len(self.result_buffer) or better(result, self.result_buffer[idx]):
#                 update(idx, result)
#     else:
#         if len(self.result_buffer) == 0 or better(result_list[0], self.result_buffer[0]):
#             self.result_buffer = result_list

def decode(model: whisper.Whisper, mel_buffer: torch.Tensor, dtype, **decode_options) -> DecodingResult:
    mel = pad_or_trim(mel_buffer, N_FRAMES).to(model.device).to(dtype)
    options = DecodingOptions(**decode_options)
    return model.decode(mel, options)


# def transcribe_step(self: "Transcriber"):
#     """
#     刷新buffer内前30s音频的转录结果
#     """

#     model: whisper.Whisper = self.model

#     mel = pad_or_trim(self.mel_buffer, N_FRAMES).to(model.device).to(self.dtype)

#     options = DecodingOptions(
#         task=               self.task,
#         language=           self.language,
#         temperature=        self.temperature(),
#         fp16=               True if self.dtype == torch.float16 else False,
#         beam_size=          self.beam_size if self.temperature() == 0 else None,
#         best_of=            None if self.temperature() == 0 else self.best_of,
#     )

#     self.decode_result = model.decode(mel, options)
