from config import API
import logging
from skimage.io import imread, imshow
from aiogram import Bot, Dispatcher, executor, types
logging.basicConfig(level=logging.INFO)
import aiogram
from aiogram.utils.helper import Helper, HelperMode, ListItem
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.types import ReplyKeyboardRemove, ReplyKeyboardMarkup, KeyboardButton
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext
from transfer import  *
import P2P
from skimage.io import imread, imsave
from PIL import Image
import os
import numpy as np
from PIL import ImageFile
from threading import Thread
from simpsons_gan import simpsonification


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Initialize bot and dispatcher
bot = Bot(token=API)
dp = Dispatcher(bot, storage=MemoryStorage())
dp.middleware.setup(LoggingMiddleware())

class StateMachine(StatesGroup):
    mode = HelperMode.snake_case
    NST = State()
    PIX2PIX = State()
    SIMPSONS = State()

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    nst_btn = KeyboardButton('NST')
    pix2pix_btn = KeyboardButton('pix2pix')
    gan_btn = KeyboardButton('Simsponification')
    start_btns = ReplyKeyboardMarkup(resize_keyboard=True)
    start_btns.add(nst_btn, pix2pix_btn, gan_btn)
    await message.reply("Привет!",  reply_markup=start_btns)


@dp.message_handler(lambda message:message.text in ["NST", "Simsponification", "pix2pix"], state='*')
async def what_a_task(message: types.Message):
    if message.text == 'NST':

        await StateMachine.NST.set()
        await message.reply("Чудесно! Скинь мне две картинки: одну подпиши как 'Content' (ее я буду менять), \n другую как 'Style' (ее стиль я буду использовать) \n (Это может занять много времени, не переживай)", reply=False)
    if message.text == 'pix2pix':
        print("Got p2p")
        await StateMachine.PIX2PIX.set()
        await message.reply("Чудесно! Отправь поврежденное фото и я его восстановлю (попробую)", reply=False)
    if message.text == 'Simsponification':
        print("Got Simsponification")
        await StateMachine.SIMPSONS.set()
        await message.reply("Чудесно! Отправь мне селфи и я его обработаю", reply=False)


@dp.message_handler(state=StateMachine.NST, content_types=['photo'])
async def nst(message: types.Message, state:FSMContext):
    uname = message['from']['username']
    await message.photo[-1].download(uname+'_'  + message['caption'].lower()+'.jpg')
    try:
        style = Image.open(uname +'_style.jpg')
        content =  Image.open(uname +'_content.jpg')
        print('both are here')
        await message.reply("Чудесно! Подожди примерно 10 минут", reply=False)
        res =  ThreadWithReturnValue(target=style_transfer, args=(content, style))
        res.start()
        res = res.join()
        res = Image.fromarray(res)
        res=res.resize(content.size)
        res.save(uname+'result.jpg')
            #imsave(message['from']['username']+'result.jpg', res)
        os.remove(uname +'_style.jpg')
        os.remove(uname +'_content.jpg')
        await bot.send_photo(message.from_user.id, aiogram.types.input_file.InputFile(message['from']['username']+'result.jpg'),
                                 caption="Style is transferred!")
        os.remove(uname +'result.jpg')
        await state.finish()
    except:
        await message.reply("И еще одну...", reply=False)
        pass

@dp.message_handler(state=StateMachine.PIX2PIX, content_types=['photo'])
async def pix2pix_func(message: types.Message, state:FSMContext):
    await message.photo[-1].download(message['from']['username']+'damaged.jpg')
    print(message)
    img = Image.open(message['from']['username']+'damaged.jpg')
    res = P2P.restore(img)
    imsave(message['from']['username']+'kinda fine.jpg', res)
    res = Image.open(message['from']['username']+'kinda fine.jpg')
    res = res.resize(img.size)#.
    #print(res.size)
    res.save(message['from']['username']+'result.jpg')
    os.remove(message['from']['username']+'damaged.jpg')
    os.remove(message['from']['username']+'kinda fine.jpg')
    await bot.send_photo(message.from_user.id, aiogram.types.input_file.InputFile(message['from']['username']+'result.jpg'),
                         caption="Restored! (or not)")
    await state.finish()


@dp.message_handler(state=StateMachine.SIMPSONS, content_types=['photo'])
async def simpsons_func(message: types.Message, state:FSMContext):
    await message.photo[-1].download(message['from']['username']+'selfie.jpg')
    print(message)
    img = Image.open(message['from']['username']+'selfie.jpg')
    res = simpsonification(img)
    imsave(message['from']['username']+'simpson.jpg', res)
    res = Image.open(message['from']['username']+'simpson.jpg')
    res = res.resize(img.size)#.
    #print(res.size)
    res.save(message['from']['username']+'result.jpg')
    os.remove(message['from']['username']+'selfie.jpg')
    os.remove(message['from']['username']+'simpson.jpg')

    await bot.send_photo(message.from_user.id, aiogram.types.input_file.InputFile(message['from']['username']+'result.jpg'),
                         caption="You look nice!")
    os.remove(message['from']['username']+'result.jpg')
    await state.finish()


@dp.message_handler(state="*", content_types=['photo'])
async def first_test_state_case_met(message: types.Message, state:FSMContext):
    print(message)
    photos = []
    photos.append(message.photo)
    print(len(photos))
    await state.finish()


if __name__ == '__main__':
    executor.start_polling(dp)
