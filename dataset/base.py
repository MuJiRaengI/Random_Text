from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import torch
import numpy as np
import json
import re
import h5py
import pandas as pd

# from transformers import AutoTokenizer
import secrets
from hangul_utils_master.hangul_utils import join_jamos
from konlpy.tag import Okt
from configs import *

# dataset download : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71748

os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-22\\bin\\server"


class KoreanText(Dataset):
    def __init__(self, data_dir, mode="train", number_filter_rate=0.1):
        self.data_dir = data_dir
        self.save_name = "dataset_%s.h5py" % mode
        self.mode = mode
        self.path = self.find_json_files(self.data_dir)
        self.length = self.calc_data_length()
        self.okt = Okt()
        self.number_filter_rate = number_filter_rate

    def calc_data_length(self):
        count = 0
        if not os.path.exists(self.save_name):
            with h5py.File(self.save_name, "w") as h5f:
                sentences_ds = h5f.create_dataset(
                    "sentences",
                    (10000,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    maxshape=(None,),
                )
                for path in tqdm(self.path):
                    with open(path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                    data = json_data["data_info"]
                    for content in data:
                        for k in [
                            "data_title",
                            "contents",
                            "question",
                            "answer",
                            "answer01",
                            "answer02",
                            "answer03",
                            "answer04",
                            "answer05",
                        ]:
                            if k in content:
                                sentence = content[k]
                                if isinstance(content[k], dict):
                                    sentence = sentence["contents"]
                                sentence = sentence.replace("\n", " ")
                                sentences_ds.resize((count + 1,))
                                sentences_ds[count] = sentence
                                count += 1

                h5f.attrs["length"] = count

        else:
            print(
                f"The {self.save_name} file already exists. Using the existing dataset."
            )
            with h5py.File(self.save_name, "r") as h5f:
                count = h5f.attrs["length"]

        return count

    def find_json_files(self, directory):
        json_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
        return json_files

    def get_random_sentence(self, number_filter_rate):
        sentence = ""
        sentence_length = secrets.randbelow(4500) + 500

        for i in range(sentence_length):
            random_number = secrets.randbelow(
                NUMBER + KEY + SHIFT_KEY + SPECIAL_KEY
            )

            if random_number < NUMBER:
                selected_char = NUMBER_LIST[random_number]
                if secrets.randbelow(100) / 100 >= number_filter_rate:
                    selected_char = ""
            elif random_number < NUMBER + KEY:
                selected_char = KEY_LIST[random_number - NUMBER]
            elif random_number < NUMBER + KEY + SHIFT_KEY:
                selected_char = SHIFT_KEY_LIST[random_number - NUMBER - KEY]
            else:
                selected_char = SPECIAL_KEY_LIST[
                    random_number - NUMBER - KEY - SHIFT_KEY
                ]

            sentence += selected_char

        meaningless_pos = ["KoreanParticle"]
        filtered_sentence = ""
        for word in join_jamos(sentence).split():
            tagged_words = self.okt.pos(word)
            # print(tagged_words)
            filtered_words = [
                word
                for word, pos in tagged_words
                if not pos in meaningless_pos
            ]
            # print(filtered_words)
            filtered_sentence += "".join(filtered_words) + " "

        return filtered_sentence

    def __len__(self):
        # return self.length * 2
        return min(400, self.length)

    def __getitem__(self, idx):
        remainder = idx % 2
        if remainder == 0:
            # idx = idx // 2
            with h5py.File(self.save_name, "r") as h5f:
                text = h5f["sentences"][np.random.randint(0, self.length)]
            text = text.decode("utf-8")

            text = re.sub(r"[^\w\s]", "", text)
            label = torch.ones((1,))
        else:
            text = self.get_random_sentence(
                number_filter_rate=self.number_filter_rate
            )
            label = torch.zeros((1,))

        # inputs = self.tokenizer.encode(text, return_tensors="pt")[0, :300]
        # print(text, inputs)
        return text, label
