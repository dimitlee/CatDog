# Анализ классификатора изображений "Котопес"

Данное приложение выводит результаты классификации изображений кошек и собак в .xlsx файл. У приложения две составные части: серверная и клиентская. Код написан на Python 3.8 с применением PyTorch 1.10. Установите требуемые библиотеки перед запуском приложения:

```
python -m pip -r requirements.txt
```

Для классификации изображений сначала запустите сервер:

```
python main.py --port 8000
```

Затем через клиентский скрипт отправьте данные о фото на сервер:

```
python request.py --port 8000 --directory data
```

**--directory** - директория в которой содержатся изображения

Клиент кодирует изображения в base64 формат и отправляет внутри json-объекта на сервер, где он расшифровывается и пропускается через классификатор. Клиент затем принимает ответ сервера и записывает результаты в Excel файл.

## Credits

Классификатор был взят у [amitrajitbose](https://github.com/amitrajitbose/cat-v-dog-classifier-pytorch).
Изображения для анализа взяты из [Kaggle](https://www.kaggle.com/chetankv/dogs-cats-images)
