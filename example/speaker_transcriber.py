import satranscriber
from satranscriber import audio
import time


if __name__ == "__main__":
	with audio.speaker.Stream() as audio_stream, \
			satranscriber.Transcriber(audio_stream, "medium", "Chinese", 0) as transcriber:
		while True:
			time.sleep(3)
			results = transcriber.read(satranscriber.ReadRequest(
				logprob_threshold=-.6,
			))
			for result in results:
				print(result.text)
