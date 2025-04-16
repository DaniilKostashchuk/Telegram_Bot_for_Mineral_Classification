# Telegram_Bot_for_Mineral_Classification
Этот репозиторий содержит бота для Telegram, который классифицирует минералы и петрографические шлифы горных пород с использованием ONNX-моделей.

## Основные возможности
* __Две модели классификации:__ отдельно для минералов и для шлифов
* __Интерактивный интерфейс:__ общение через меню с кнопками
* __Обработка изображений:__ анализ загружаемых фотографий
* __Визуализация результатов:__ вывод изображения с подписанным классом

## Структура кода
### Основные компоненты
```python
CHOOSING_MODEL, PROCESSING_PHOTO = range(2)  # Состояния диалога
```

### Класс MineralClassifierBot
Основной класс бота с ключевыми методами:
#### Инициализация
```python
def __init__(self):
    # Загрузка ONNX-моделей
    self.mineral_model = ort.InferenceSession('efficientnet_minet.onnx')
    self.section_model = ort.InferenceSession('efficientnet_sections.onnx')
    
    # Классы классификации
    self.mineral_classes = ['biotite', 'bornite', 'chrysocolla', ...]
    self.section_classes = ['Базальтоиды', 'Гранитоиды']
    
    # Настройка бота Telegram
    self.app = Application.builder().token("ВАШ_ТОКЕН").build()
    
    # Настройка обработчика диалога
    conv_handler = ConversationHandler(...)
    self.app.add_handler(conv_handler)
```
#### Основные методы
1. __Обработка изображений:__
```python
def preprocess_image(self, image_path: str):
    """Подготовка изображений для модели"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(...)
    ])
    return transform(image).unsqueeze(0).numpy()
```
2. __Обработка фотографий:__
```python
async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Анализ загруженных фото"""
    # Скачивание, обработка, классификация
    # Генерация визуализации результата
    await update.message.reply_photo(...)
```
3. __Выбор модели:__
```python
async def choose_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Переключение между моделями"""
    if choice == "Минералы":
        self.current_model = self.mineral_model
    else:
        self.current_model = self.section_model
```
#### Запуск бота
```python
async def main():
    bot = MineralClassifierBot()
    try:
        await bot.run()  # Запуск бота
    except KeyboardInterrupt:
        await bot.shutdown()

if __name__ == '__main__':
    # Работает как в скрипте, так и в Jupyter
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(main())
    else:
        asyncio.run(main())
```
## Требования
* Python 3.7+
* Необходимые пакеты:
```
python-telegram-bot
numpy
onnxruntime
torch
torchvision
pillow
matplotlib
```
## Установка
1. Укажите токен бота:
```python
self.app = Application.builder().token("ВАШ_ТОКЕН").build()
```
2. Поместите файлы моделей в корневую папку:
* `efficientnet_minet.onnx`
* `efficientnet_sections.onnx`
3. Запустите бота:
```python
python bot.py
```
## Использование
1. Начните работу командой `/start`
2. Выберите тип классификации (минералы или шлифы)
3. Загрузите фотографии для анализа
4. Используйте `/stop` для остановки бота

Бот выводит визуализированные результаты с подписанными классами и поддерживает состояние диалога между запросами.

## Лицензия
Этот проект распространяется под лицензией MIT. Подробности см. в файле `LICENSE`.

## Ссылки
Ссылка на нейронную сеть: https://github.com/DaniilKostashchuk/Classification_of_images_of_minerals_and_rock_cuts
