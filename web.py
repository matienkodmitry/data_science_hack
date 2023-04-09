import streamlit as st
from rubert_category.bert_model import BertClassifierCategory
from rubert_semantic.bert_model import BertClassifierSemantic
import torch
import gdown
from pathlib import Path
import json

@st.cache(allow_output_mutation=True)
def load_model_rubert_category():
    my_file = Path("rubert_category/bert.pth")
    if my_file.is_file():
        model = torch.load('rubert_category/bert.pth', map_location=torch.device('cpu'))
    else:
        with st.spinner("Загружаем модель... Это может занять время! \n Не останавливайте это!"):
            url = 'https://drive.google.com/file/d/1KC9mdiHunwVFwmlf4Lp2RLuqY6a9DO-2/view?usp=share_link'
            gdown.download_folder(url, quiet=True, use_cookies=False)
            model = model = torch.load('rubert_category/bert.pth', map_location=torch.device('cpu'))

    bert_model = BertClassifierCategory(model_path='cointegrated/rubert-tiny',
                                        tokenizer_path='cointegrated/rubert-tiny',
                                        n_classes=4,
                                        epochs=10,
                                        model_save_path='rubert_category/bert.pth')
    bert_model.model = model

    return bert_model


@st.cache(allow_output_mutation=True)
def load_model_rubert_semantic():
    my_file = Path("rubert_semantic/bert.pth")
    if my_file.is_file():
        model = torch.load('rubert_semantic/bert.pth', map_location=torch.device('cpu'))
    else:
        with st.spinner("Загружаем модель... Это может занять время! \n Не останавливайте это!"):
            url = 'https://drive.google.com/file/d/1Kw5rKEFCIQBZFeObpybTpmUfYN657mz-/view?usp=share_link'
            gdown.download_folder(url, quiet=True, use_cookies=False)
            model = model = torch.load('rubert_semantic/bert.pth', map_location=torch.device('cpu'))

    bert_model = BertClassifierSemantic(model_path='cointegrated/rubert-tiny',
                                        tokenizer_path='cointegrated/rubert-tiny',
                                        n_classes=4,
                                        epochs=10,
                                        model_save_path='rubert_semantic/bert.pth')
    bert_model.model = model
    return bert_model

def read_label2cat():
    with open('rubert_category/label2cat.json', 'r') as f:
        label2cat = json.load(f)
    return label2cat

def read_label2sent():
    with open('rubert_semantic/label2sent.json', 'r') as f:
        label2sent = json.load(f)
    return label2sent


bert_category = load_model_rubert_category()
bert_semantic = load_model_rubert_semantic()
label2cat = read_label2cat()
label2sent = read_label2sent()

st.title('Определение категории по отзыву')
sentence = st.text_input('Напишите отзыв:')
result = st.button('Предсказать категорию')

if sentence and result:
    st.write(f'Категория отзыва: {label2cat[str(bert_category.predict(sentence))]}')
    st.write(f'Категория отзыва: {label2sent[str(bert_category.predict(sentence))]}')
