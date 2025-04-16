# Состояния для ConversationHandler
CHOOSING_MODEL, PROCESSING_PHOTO = range(2)

class MineralClassifierBot:
    def __init__(self):
        # Загрузка обеих моделей при инициализации
        self.mineral_model = ort.InferenceSession('efficientnet_minet.onnx')
        self.section_model = ort.InferenceSession('efficientnet_sections.onnx')
        
        # Классы для каждой модели
        self.mineral_classes = ['biotite', 'bornite', 'chrysocolla', 'malachite', 'muscovite', 'pyrite', 'quartz']
        self.section_classes = ['Базальтоиды', 'Гранитоиды']
        
        # Текущая выбранная модель
        self.current_model = None
        self.current_model_name = None
        self.current_classes = None
        
        # Настройка бота
        self.app = Application.builder().token("ВАШ_ТОКЕН").build()
        self.running = False
        
        # Настройка клавиатуры для выбора модели
        self.model_selection_keyboard = [
            ["Минералы", "Шлифы"],
            ["/stop"]
        ]
        self.model_selection_markup = ReplyKeyboardMarkup(self.model_selection_keyboard, one_time_keyboard=True)
        
        # Клавиатура после выбора модели
        self.photo_processing_keyboard = [
            ["Выбрать классификацию"],
            ["/stop"]
        ]
        self.photo_processing_markup = ReplyKeyboardMarkup(self.photo_processing_keyboard, one_time_keyboard=True)
        
        # Регистрация обработчиков
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('start', self.start)],
            states={
                CHOOSING_MODEL: [
                    MessageHandler(filters.Regex("^(Минералы|Шлифы)$"), self.choose_model),
                ],
                PROCESSING_PHOTO: [
                    MessageHandler(filters.PHOTO, self.handle_photo),
                    MessageHandler(filters.Regex("^Выбрать классификацию$"), self.back_to_model_selection),
                ],
            },
            fallbacks=[CommandHandler('stop', self.stop)],
        )
        
        self.app.add_handler(conv_handler)
        self.app.add_handler(CommandHandler("stop", self.stop))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        self.running = True
        await update.message.reply_text(
            "🔍 Выберите тип классификации:",
            reply_markup=self.model_selection_markup
        )
        return CHOOSING_MODEL

    async def choose_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Выбор модели для классификации"""
        choice = update.message.text
        if choice == "Минералы":
            self.current_model = self.mineral_model
            self.current_model_name = "минералов"
            self.current_classes = self.mineral_classes
        else:
            self.current_model = self.section_model
            self.current_model_name = "шлифов"
            self.current_classes = self.section_classes
            
        await update.message.reply_text(
            f"Выбрана модель для классификации {self.current_model_name}. "
            "Отправьте фото для анализа или нажмите 'Выбрать классификацию' для смены типа.",
            reply_markup=self.photo_processing_markup
        )
        return PROCESSING_PHOTO

    async def back_to_model_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Возврат к выбору модели классификации"""
        await update.message.reply_text(
            "🔍 Выберите тип классификации:",
            reply_markup=self.model_selection_markup
        )
        return CHOOSING_MODEL

    async def stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stop для graceful shutdown"""
        self.running = False
        await update.message.reply_text("🛑 Остановка бота...")
        await self.shutdown()
        return ConversationHandler.END

    def preprocess_image(self, image_path: str):
        """Синхронная обработка изображения (одинаковая для обеих моделей)"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0).numpy()

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка фотографий"""
        if not self.running:
            await update.message.reply_text("❌ Бот в данный момент остановлен. Используйте /start")
            return
            
        if not self.current_model:
            await update.message.reply_text("ℹ️ Сначала выберите модель с помощью /start")
            return
            
        try:
            # Скачивание фото
            photo_file = await update.message.photo[-1].get_file()
            temp_file = 'temp_photo.jpg'
            await photo_file.download_to_drive(temp_file)
            
            # Обработка изображения
            input_data = self.preprocess_image(temp_file)
            input_name = self.current_model.get_inputs()[0].name
            outputs = self.current_model.run(None, {input_name: input_data})[0]
            predicted_class = np.argmax(outputs, axis=1)[0]
            class_name = self.current_classes[predicted_class]
            
            # Визуализация результата
            image = Image.open(temp_file)
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            plt.title(f"Predicted: {class_name}", fontsize=16)
            plt.axis('off')
            result_path = 'result.jpg'
            plt.savefig(result_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close()
            
            await update.message.reply_photo(
                photo=open(result_path, 'rb'),
                caption=f"🏆 Результат ({self.current_model_name}): {class_name}",
                reply_markup=self.photo_processing_markup
            )
            
            # Очистка временных файлов
            os.remove(temp_file)
            os.remove(result_path)
            
            # Остаемся в состоянии обработки фото, не предлагаем выбор модели
            return PROCESSING_PHOTO
            
        except Exception as e:
            await update.message.reply_text(f"❌ Ошибка: {str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return PROCESSING_PHOTO

    async def run(self):
        """Запуск бота"""
        self.running = True
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        print("🤖 Бот запущен. Для остановки:\n1. Нажмите Ctrl+C\n2. Или отправьте /stop в чат")
        
        while self.running:
            await asyncio.sleep(1)

    async def shutdown(self):
        """Корректное завершение работы"""
        if hasattr(self, 'app') and self.app.running:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
        print("✅ Бот успешно остановлен")

async def main():
    bot = MineralClassifierBot()
    try:
        await bot.run()
    except (asyncio.CancelledError, KeyboardInterrupt):
        await bot.shutdown()
    except Exception as e:
        print(f"⚠️ Критическая ошибка: {e}")
        await bot.shutdown()

if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Принудительная остановка")
