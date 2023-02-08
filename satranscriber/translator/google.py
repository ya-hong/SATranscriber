import googletrans

from . import translator


class Translator(translator.Translator):
	def __init__(self, source_lang: str, target_lang: str) -> None:
		super().__init__(source_lang, target_lang)
		self.translator = googletrans.Translator()
	
	def authentication(self, **kwargs):
		return
	
	def translate(self, text: str) -> str:
		return self.translator.translate(text, src=self.source_lang, dest=self.target_lang).text

