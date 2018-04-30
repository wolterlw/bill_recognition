import telegram
from telegram.ext import Updater, CommandHandler
from telegram.ext import MessageHandler, Filters

from requests import post as r_post
import io

import logging
logging.basicConfig(level=logging.INFO)

credentials = {
	"token": open('tg_bot_token','r').read().strip()
}

updater = Updater(token = credentials["token"])
dispatcher = updater.dispatcher

def start(bot, update):
	bot.send_message(chat_id=update.message.chat_id,
		text="""
		Hello, I am a bill recognition bot (WIP)
		So far I can only cut bills from images and detect lines of text
		!!!DO NOT CROP PHOTOS!!!
		The whole idea is to automate it
		""")

start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

def get_photo(bot, update):

	photo = max(update.message.photo, key=lambda x: x.height)
	print("received: " + str(photo.file_id))


	ph_file = bot.get_file(photo.file_id)
	ph_file.download("./tmp/unlabelled_ones/" + str(photo.file_id) + ".jpg")
	image = open("./tmp/unlabelled_ones/" + str(photo.file_id) + ".jpg",mode='rb').read()

	payload = {"image": image}

	resp = r_post('http://127.0.0.1:5000/predict', files=payload)

	bot.send_photo(chat_id=update.message.chat_id, photo=io.BytesIO(resp.content))


photo_handler = MessageHandler(Filters.photo, get_photo)
dispatcher.add_handler(photo_handler)

updater.start_polling()