from dataset.base import KoreanText


def main():
    data_dir = "D:\\dataset\\korean\\121.한국어 성능이 개선된 초거대AI 언어모델 개발 및 데이터\\3.개방데이터\\1.데이터\\Training"
    mode = "train"
    # data_dir = "D:\\dataset\\korean\\121.한국어 성능이 개선된 초거대AI 언어모델 개발 및 데이터\\3.개방데이터\\1.데이터\\Validation"
    # mode = "valid"
    dataset = KoreanText(data_dir, mode)
    img, label = dataset.__getitem__(0)
    img, label = dataset.__getitem__(1)
    img, label = dataset.__getitem__(2)
    img, label = dataset.__getitem__(3)
    img, label = dataset.__getitem__(4)
    img, label = dataset.__getitem__(5)

    print("finish")


if __name__ == "__main__":
    main()
