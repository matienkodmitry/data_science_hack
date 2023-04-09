import io
import streamlit as st
from rubert_category.bert_model import BertClassifierCategory
from PIL import Image
from PIL import ImageDraw, ImageFont
import torch
import gdown
from pathlib import Path


@st.cache(allow_output_mutation=True)
def load_model():
    my_file = Path("rubert_category/bert.pth")
    if my_file.is_file():
        model = torch.load('rubert_category/bert.pth', map_location=torch.device('cpu'))
    else:
        url = 'https://drive.google.com/file/d/1KC9mdiHunwVFwmlf4Lp2RLuqY6a9DO-2/view?usp=share_link'
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            gdown.download_folder(url, quiet=True, use_cookies=False)

        model = model = torch.load('rubert_category/bert.pth', map_location=torch.device('cpu'))
    return model


bert_category = load_model()
# Выводим заголовок страницы средствами Streamlit
st.title('Определение категории по отзыву')
# Вызываем функцию создания формы загрузки изображения
sentence = st.text_input('Напишите отзыв:')
result = st.button('Предсказать категорию')

if sentence and result:
    st.write(bert_category.predict(sentence))



