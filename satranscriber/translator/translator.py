import abc


class Translator(abc.ABC):
	def __init__(self, source_lang: str, target_lang: str, **kwargs) -> None:
		self.source_lang = source_lang
		self.target_lang = target_lang
	
	@abc.abstractmethod
	def authentication(self, **kwargs):
		pass
	
	@abc.abstractmethod
	def translate(self, text: str) -> str:
		pass
