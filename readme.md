# SATranscriber

一个基于[openai/Whisper](https://github.com/openai/whisper)的实时语音流转录工具。

# Usage

## cli

```shell
python3 cli.py -h
```

转录扬声器中播放的日语音频， 并通过谷歌翻译API翻译为中文:

```shell
python3 cli.py --audio speaker --language ja --translator_api google --source_lang ja --target_lang zh-CN
```

从文件读取配置

```shell
python3 cli.py --config path/to/config.json
```

## python 

```python
import time

import satranscriber
from satranscriber.audio import speaker

audio_stream = speaker.Stream()
transcriber = satranscriber.Transcriber(audio_stream=audio_stream)

with audio_stream, transcriber:
	while True:
		time.sleep(3)
		for result in transcriber.read(satranscriber.ReadRequest()):
			print(result.text)
```

如果需要使用扬声器以外的其他音频源，可以从`satreanscriber.audio.Stream`继承实现一个新的音频流。

