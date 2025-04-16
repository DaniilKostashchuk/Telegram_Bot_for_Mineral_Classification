import asyncio
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    ConversationHandler,
    filters
)
