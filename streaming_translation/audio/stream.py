from abc import ABC, abstractmethod
import whisper
import numpy as np

class Stream(ABC):
	SAMPLE_RATE = whisper.audio.SAMPLE_RATE

	@abstractmethod
	def read(self) -> np.ndarray:
		"""
		取得单声道、采样率16000的float32的numpy ndarray
		要注意转换为float时需根据样本宽度进行normalization。也就是int16的音频转换为float需要除以(1<<16)-1
		"""
		pass

	@abstractmethod
	def __enter__(self):
		pass

	@abstractmethod
	def __exit__(self, type, value, traceback) -> None:
		pass
