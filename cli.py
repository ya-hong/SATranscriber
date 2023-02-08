from typing import *

import argparse
import time
import importlib
import dataclasses
import json

import satranscriber
# from satranscriber import audio, translator


def get_parser() -> argparse.ArgumentParser:
    import whisper
    from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=None, type=str, help="path to json config file")
    parser.add_argument("--verbose", default=False, type=bool)

    transcriber = parser.add_argument_group("transcriber")
    transcriber.add_argument("--model", default="medium", choices=whisper.available_models(), help="name of the Whisper model to use")
    transcriber.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    transcriber.add_argument("--language", type=str, default="English", choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")
    transcriber.add_argument("--temperature", type=float, nargs='+', default=(0), help="temperature to use for sampling")
    transcriber.add_argument("--beam_size", type=int, default=10, help="number of beams in beam search, only applicable when temperature is zero")
    transcriber.add_argument("--best_of", type=int, default=10, help="number of candidates when sampling with non-zero temperature")
    transcriber.add_argument("--fp16", type=bool, default=True, help="whether to perform inference in fp16; True by default")

    verification = parser.add_argument_group("verification")
    verification.add_argument("--logprob_threshold", type=float, default=-0.6, help="if the average log probability is lower than this value, treat the decoding as failed")
    verification.add_argument("--compression_ratio_threshold", type=float, default=1.8, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    verification.add_argument("--no_speech_threshold", type=float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    verification.add_argument("--padding", type=int, default=200, help="if the distance from the end of the audio is greater than this value, treat the sentence as incomplete (padding 100 = 1s)")

    audio = parser.add_argument_group("audio")
    audio.add_argument("--audio", type=str, default="speaker", choices=["speaker"], help="audio streaming to transcribe")

    translator = parser.add_argument_group("translator")
    translator.add_argument("--translator_api", type=str, choices=["youdao", "baidu", "google"], help="translate api for X -> Y translate")
    translator.add_argument("--app_key", type=str, help="youdao api app key / baidu api app id")
    translator.add_argument("--app_secret", type=str, help="youdao api app secret / baidu api app key")
    translator.add_argument("--secret_file", type=str, help="path to secret file")
    translator.add_argument("--source_lang", type=str, help="source language for tranlator api")
    translator.add_argument("--target_lang", type=str, help="target language for tranlator api")

    return parser


if __name__ == "__main__":
    from pprint import pprint
    
    args = get_parser().parse_args().__dict__

    if args["config"]:
        with open(args["config"], "r") as f:
            cfg = json.loads(f.read())
        args = {**args, **cfg}

    pprint(args)

    try:
        module = importlib.import_module("satranscriber.audio.{}".format(args["audio"]))
        AudioStream = getattr(module, "Stream")
        audio_stream: satranscriber.audio.Stream = AudioStream()
    except:
        print("failed to load audio module")
        raise
    
    try:
        transcriber = satranscriber.Transcriber(audio_stream=audio_stream, **args)
    except:
        print("failed to load transcriber")
        raise

    try:
        translator = None
        
        if args["secret_file"]:
            with open(args["secret_file"], "r") as f:
                lines = f.readlines()
            args["app_key"] = lines[0].strip()
            args["app_secret"] = lines[1].strip()
        
        if args["translator_api"]:
            module = importlib.import_module("satranscriber.translator.{}".format(args["translator_api"]))
            Translator = getattr(module, "Translator")
            translator: satranscriber.translator.Translator = Translator(args["source_lang"], args["target_lang"])
            translator.authentication(**args)
    except:
        print("failed to load translator")
        raise

    verification = dict()
    for filed in dataclasses.fields(satranscriber.ReadRequest):
        if filed.name in args:
            verification[filed.name] = args[filed.name]
    r = satranscriber.ReadRequest(**verification)

    with audio_stream, transcriber:
        while True:
            time.sleep(3)
            results = transcriber.read(r)
            for result in results:
                text = result.text
                if translator:
                    try:
                        text = translator.translate(text)
                    except:
                        pass
                print(text)
