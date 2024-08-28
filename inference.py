import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


class TestMode(nn.Module):
    def forward(self, x):
        return x[:, :1, 0, 0]


def main():
    device = "cuda:0"
    save_dir = "."

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

    with torch.no_grad():
        text = "안녕하세요 무지랭이입니다 오늘은 무한 원숭이 정리를 직접 실험해보도록 하겠습니다"
        # text = "러몰누랑느래노러무퍼마프차투퍼도 혿여먀 퓽너뮤 포댜갲뮤 펑뮾 뮤 패뮤퓨엄모"
        # text = "안녕하세요 무지랭이입니다 러몰누랑느래노러무퍼마프차투퍼도 혿여먀 퓽너뮤 포댜갲뮤 펑뮾 뮤 패뮤퓨엄모"
        # text = "러몰누랑느래노러무퍼마프차투퍼도 혿여먀 퓽너뮤 포댜갲뮤 펑뮾 뮤 패뮤퓨엄모 안녕하세요 무지랭이입니다"
        # text = "러몰누랑느래노러무퍼마프차투퍼도 안녕하세요 무지랭이입니다 혿여먀 퓽너뮤 포댜갲뮤 펑뮾 뮤 패뮤퓨엄모"
        data = tokenizer.encode(text, return_tensors="pt")[0, :300]

        prob = F.sigmoid(model(data.unsqueeze(0).to(device))["logits"])
        print(f"prob {prob.item()}")


if __name__ == "__main__":
    main()
