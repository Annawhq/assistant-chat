import json
from vosk import Model, KaldiRecognizer
import os
import subprocess


class Speech:
    model = Model(r"D:/vosk/vosk-model-small-ru-0.22/vosk-model-small-ru-0.22")
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)
    ffmpeg_path = "C:/ffmpeg/ffmpeg.exe"

    def recognize(self, audio_file_name=None):
        # Конвертация аудио в wav и результат в process.stdout
        process = subprocess.Popen(
            [self.ffmpeg_path,
             "-loglevel", "quiet",
             "-i", audio_file_name,  # имя входного файла
             "-ar", str(16000),  # частота выборки
             "-ac", "1",  # кол-во каналов
             "-f", "s16le",  # кодек для перекодирования, у нас wav
             "-"  # имя выходного файла нет, тк читаем из stdout
             ],
            stdout=subprocess.PIPE
        )

        # Чтение данных по кусочкам и распознование через модель
        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                break
            if self.rec.AcceptWaveform(data):
                pass

        # Возвращаем распознанный текст в виде str
        result_json = self.rec.FinalResult()  # это json в виде str
        result_dict = json.loads(result_json)  # это dict
        return result_dict["text"]  # текст в виде str

