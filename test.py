import os

os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-22\\bin\\server"

from hangul_utils_master.hangul_utils import join_jamos, sent_word_tokenize
from konlpy.tag import Kkma, Okt
import secrets


NUMBER_LIST = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
NUMBER = len(NUMBER_LIST)

KEY_LIST = [
    "ㅂ",
    "ㅈ",
    "ㄷ",
    "ㄱ",
    "ㅅ",
    "ㅛ",
    "ㅕ",
    "ㅑ",
    "ㅐ",
    "ㅔ",
    "ㅁ",
    "ㄴ",
    "ㅇ",
    "ㄹ",
    "ㅎ",
    "ㅗ",
    "ㅓ",
    "ㅏ",
    "ㅣ",
    "ㅋ",
    "ㅌ",
    "ㅊ",
    "ㅍ",
    "ㅠ",
    "ㅜ",
    "ㅡ",
]
KEY = len(KEY_LIST)

SHIFT_KEY_LIST = ["ㅃ", "ㅉ", "ㄸ", "ㄲ", "ㅆ", "ㅒ", "ㅖ"]
SHIFT_KEY = len(SHIFT_KEY_LIST)

SPECIAL_KEY_LIST = [" "]
SPECIAL_KEY = len(SPECIAL_KEY_LIST)

okt = Okt()

sentence = ""
sentence_length = secrets.randbelow(9500) + 500

for i in range(sentence_length):
    random_number = secrets.randbelow(NUMBER + KEY + SHIFT_KEY + SPECIAL_KEY)

    if random_number < NUMBER:
        selected_char = NUMBER_LIST[random_number]
    elif random_number < NUMBER + KEY:
        selected_char = KEY_LIST[random_number - NUMBER]
    elif random_number < NUMBER + KEY + SHIFT_KEY:
        selected_char = SHIFT_KEY_LIST[random_number - NUMBER - KEY]
    else:
        selected_char = SPECIAL_KEY_LIST[
            random_number - NUMBER - KEY - SHIFT_KEY
        ]

    sentence += selected_char

# print(join_jamos(sentence).split())

# Extraction of Meaningful Morphemes
meaningless_pos = ["KoreanParticle"]
filtered_sentence = ""
for word in join_jamos(sentence).split():
    tagged_words = okt.pos(word)
    # print(tagged_words)
    filtered_words = [
        word for word, pos in tagged_words if not pos in meaningless_pos
    ]
    # print(filtered_words)
    filtered_sentence += "".join(filtered_words) + " "
    # print(filtered_sentence)

print(filtered_sentence)

# Remove Numbers
meaningless_pos2 = ["Number"]
filtered_sentence2 = ""
for word in join_jamos(filtered_sentence).split():
    tagged_words = okt.pos(word)
    # print(tagged_words)
    filtered_words = [
        word for word, pos in tagged_words if not pos in meaningless_pos2
    ]
    # print(filtered_words)
    filtered_sentence2 += "".join(filtered_words) + " "
    # print(filtered_sentence2)

print(filtered_sentence2)


print()
