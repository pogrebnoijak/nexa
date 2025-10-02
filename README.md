# NEXA

Веб-сервис предиктивной аналитики физиологических данных с интеграцией в медицинское оборудование

Полностью запустить можно так (нужно иметь установленный docker, docker-compose):

`docker compose up` - поднимется `backend` (http://localhost:8000) и `frontend` (http://localhost:4000).

(если что-то поменялось - стоит пересобрать `docker compose build --no-cache`)

## Бэкенд

Можно запустить отдельно с помощью [start](backend/start.sh) предварительно установив [requirements](backend/requirements.txt)

`swagger` - http://localhost:8000/docs

### Структура

- [api](backend/app/api.py), [модели данных](backend/app/abstract/abstract.py)

- [аналитика](backend/app/compute/logic.py)
- [тренировка моделей](backend/app/compute/training.py)
- [препроцессинг данных](backend/app/compute/preprocessing.py)
- модели [cnn](backend/app/compute/prediction_cnn_gru.py), [base](backend/app/compute/prediction.py)

- [хранение данных](backend/app/storage/storage.py)

- [вспомогательные функции](backend/app/utils)

- [веса](backend/app/assets), [данные](backend/app/data), [константы](backend/app/consts.py), 

## Фронтенд

Можно запустить отдельно с помощью [start](frontend/start.sh) предварительно установив `node`

[front](tools/frontend_client.py) - простой скрипт для эмуляции фронтенда

## Генератор

[generator](tools/generator.py)

Можно настроить путь `_prefix` (оттуда будут браться данные) и ускорение относительно временных меток - `_gen_sleep_k`

Можно запускать при работающем бэкенде. Будет отдавать ему данные через `websocket`
