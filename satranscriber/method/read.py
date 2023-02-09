# from typing import *

# if TYPE_CHECKING:
#     from satranscriber.transcriber import Transcriber

# import dataclasses
# import whisper
# from whisper import DecodingResult
# from whisper.audio import N_FRAMES, HOP_LENGTH, SAMPLE_RATE
# from whisper.tokenizer import get_tokenizer, Tokenizer

# from .parse_result import split_decode_result, to_transcribe_results, TranscribeResult


# def read(
#     self: "Transcriber",
# ) -> List[TranscribeResult]:
#     """
#     一次read最好至少输出一句话。如果什么也不输出需要调整策略(现在调整策略的部分移动到了其他函数中)
#     """

#     self.try_log("read from:")
#     self.try_log(self.decode_result)

#     if self.decode_result is None or not self.is_quality(self.decode_result):
#         return []

#     stoken = self.decode_result.tokens[0]

#     def is_stable(result: DecodingResult):
#         return (result.tokens[-1] - stoken) * self.input_stride < self.mel_buffer.shape[-1] - self.padding

#     tokenizer: Tokenizer = get_tokenizer(
#         self.model.is_multilingual,
#         language=self.language, 
#         task=self.task)

#     results = split_decode_result(tokenizer, self.decode_result)
#     results = to_transcribe_results(self, results)

#     if len(stable_results) == 0:
#         return []

#     ttoken = stable_results[-1].tokens[-1]
#     gap = (ttoken - stoken) * self.input_stride
#     self.extend_offset(gap)

#     return stable_results

