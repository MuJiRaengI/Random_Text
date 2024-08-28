import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import cv2
import logging
import time
import keyboard
from numpy import ndarray
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import secrets
from configs import *
from konlpy.tag import Okt
from hangul_utils_master.hangul_utils import join_jamos

os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-22\\bin\\server"


def main():
    device = "cuda:0"
    save_dir = "results_txt"

    model = AutoModelForSequenceClassification.from_pretrained(
        "beomi/kcbert-base", num_labels=1
    )
    model.load_state_dict(
        torch.load(
            "C:\\Users\\stpe9\\Desktop\\vscode\\Random_Text\\results_model\\2024y_08m_19d_23h_07m_56s\\best_loss_model.pth"
        )
    )
    # model load
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")

    logger = logging.getLogger("Random Noise Text logger")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("log.log")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    okt = Okt()

    threshold = 0.5
    max_prob = 0.5
    count = 0.0
    logger.info(f"<< Start >>")
    tmr = time.time()
    with torch.no_grad():
        while True:
            time.sleep(0.02)
            print(f"running... count : {count} max_prob : {max_prob:.3f}")

            sentence = ""
            sentence_length = secrets.randbelow(9500) + 500

            for i in range(sentence_length):
                random_number = secrets.randbelow(
                    NUMBER + KEY + SHIFT_KEY + SPECIAL_KEY
                )

                if random_number < NUMBER:
                    selected_char = NUMBER_LIST[random_number]
                elif random_number < NUMBER + KEY:
                    selected_char = KEY_LIST[random_number - NUMBER]
                elif random_number < NUMBER + KEY + SHIFT_KEY:
                    selected_char = SHIFT_KEY_LIST[
                        random_number - NUMBER - KEY
                    ]
                else:
                    selected_char = SPECIAL_KEY_LIST[
                        random_number - NUMBER - KEY - SHIFT_KEY
                    ]

                sentence += selected_char

            meaningless_pos = ["KoreanParticle"]
            filtered_sentence = ""
            for word in join_jamos(sentence).split():
                tagged_words = okt.pos(word)
                filtered_words = [
                    word
                    for word, pos in tagged_words
                    if not pos in meaningless_pos
                ]
                filtered_sentence += "".join(filtered_words) + " "

            meaningless_pos2 = ["Number"]
            filtered_sentence2 = ""
            for word in join_jamos(filtered_sentence).split():
                tagged_words = okt.pos(word)
                filtered_words = [
                    word
                    for word, pos in tagged_words
                    if not pos in meaningless_pos2
                ]
                filtered_sentence2 += "".join(filtered_words) + " "

            data = tokenizer.encode(filtered_sentence2, return_tensors="pt")[
                0, :300
            ]
            prob = F.sigmoid(
                model(data.unsqueeze(0).to(device))["logits"]
            ).item()
            # check max prob
            if prob > max_prob:
                max_prob = prob
                save_path = os.path.join(
                    save_dir, "max_prob_%.6f.txt" % max_prob
                )
                save_text(
                    save_path,
                    sentence,
                    filtered_sentence,
                    filtered_sentence2,
                )
                logger.info(
                    f"[max_prob]Save {save_path}. max_prob : {max_prob}"
                )

            # check threshold
            if prob > threshold:
                save_path = os.path.join(save_dir, "prob_%.6f.txt" % prob)
                save_text(
                    save_path,
                    sentence,
                    filtered_sentence,
                    filtered_sentence2,
                )
                logger.info(f"[threshold]Save {save_path}. prob : {prob}")

            dirs = os.listdir()

            if "_stop_" in dirs:
                total_time = round(time.time() - tmr)
                d, h, m, s = convert_seconds(total_time)
                logger.info(f"Stop. running time : {d}d {h}h {m}m {s}s")
                break


def save_text(path, sentence, filtered_sentence, filtered_sentence2):
    with open(path, "w", encoding="utf-8") as file:
        file.write("원본 내용" + "\n")
        file.write(sentence + "\n\n")
        file.write("1차 필터 내용" + "\n")
        file.write(filtered_sentence + "\n\n")
        file.write("2차 필터 내용" + "\n")
        file.write(filtered_sentence2 + "\n")


def convert_seconds(seconds):
    days, remainder = divmod(seconds, 86400)  # 1일 = 86400초
    hours, remainder = divmod(remainder, 3600)  # 1시간 = 3600초
    minutes, seconds = divmod(remainder, 60)  # 1분 = 60초

    return days, hours, minutes, seconds


def press_and_release(key, delay=1):
    keyboard.press(key)
    time.sleep(delay)
    keyboard.release(key)


if __name__ == "__main__":
    main()
