import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from filter import text_filter

with open("intents_dataset.json", "r", encoding="utf-8") as config_file:
    data = json.load(config_file, strict=False)
INTENTS = data["intents"]

X = []
Y = []

for intent in INTENTS:
    examples = INTENTS[intent]["examples"]
    for example in examples:
        example = text_filter(example)
        X.append(example)  # Добавляем фразу в список X
        Y.append(intent)   # Добвляем интент в списко Y

vectorizer = CountVectorizer(analyzer="char") #Настройки в скобочках
#попробовать ngram_range (вектоизацию по буквам, а не словам), analyzer
vectorizer.fit(X) #Обучаем векторайзер

vecX = vectorizer.transform(X)    #Все тексты преобразуем в вектора
model = RandomForestClassifier() #попробовать Настройки, n_estimators = ?, max_depth?
model.fit(vecX, Y) #Обучение модели
