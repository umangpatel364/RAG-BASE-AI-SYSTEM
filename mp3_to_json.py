import whisper
import json
import os


model = whisper.load_model('large-v2')
AUDIO = os.listdir("AUDIO")

for audio in AUDIO:
    if("_" in audio):
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        print(number, title)

    #    result = model.transcribe(audio = "AUDIO/sample.mp3",      # use for creat 10 sec text from every video
        result = model.transcribe(audio = f"AUDIO/{audio}",
                         language = 'hi',
                         task = 'translate',
                         word_timestamps=False    )

        chunks = []
        for segment in result['segments']:
            chunks.append({"number" : number,"title":title,"start" : segment["start"],"end" : segment["end"],"text" : segment["text"]})
        
        chunk_and_metadata = {"chunks":chunks,"text" : result['text']}

        with open(f"jsons/{audio}.json" , "w") as f:
            json.dump(chunk_and_metadata,f)