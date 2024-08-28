import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import logging
from datetime import datetime
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset.base import KoreanText
from transformers import AutoModelForSequenceClassification
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from metric.accuracy import AccuracyCalculator


def set_log(save_dir):
    log_file_path = os.path.join(save_dir, "log.txt")
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger("")
    root_logger.addHandler(console_handler)


def make_result_dir_path(path):
    current_time = datetime.now()
    year_str = str(current_time.year)
    month_str = str(current_time.month).zfill(2)
    day_str = str(current_time.day).zfill(2)
    hour_str = str(current_time.hour).zfill(2)
    minute_str = str(current_time.minute).zfill(2)
    second_str = str(current_time.second).zfill(2)
    now = f"{year_str}y_{month_str}m_{day_str}d_{hour_str}h_{minute_str}m_{second_str}s"
    return os.path.join(path, now)


def collate_fn(batch):
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    data, labels = zip(*batch)
    encodings = tokenizer.batch_encode_plus(
        list(data),
        add_special_tokens=True,
        return_attention_mask=False,
        padding="longest",
        max_length=300,
        truncation=True,
        return_tensors="pt",
    )
    data = encodings["input_ids"]
    data_padded = pad_sequence(
        [d.clone().detach() for d in data],
        batch_first=True,
        padding_value=0,
    )
    max_length = 300
    data_padded = torch.stack(
        [
            (
                torch.cat([d, torch.zeros(max_length - d.size(0))])
                if d.size(0) < max_length
                else d[:max_length]
            )
            for d in data_padded
        ]
    )
    return data_padded, torch.tensor(labels)


def train():
    device = "cuda:0"
    save_path = make_result_dir_path("results_model")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    set_log(save_path)
    batch_size = 8
    num_workers = 0
    lr = 1e-6
    epochs = 100

    train_dataset = KoreanText(
        "D:\\dataset\\korean\\121.한국어 성능이 개선된 초거대AI 언어모델 개발 및 데이터\\3.개방데이터\\1.데이터\\Training",
        mode="train",
        number_filter_rate=0.1,
    )
    valid_dataset = KoreanText(
        "D:\\dataset\\korean\\121.한국어 성능이 개선된 초거대AI 언어모델 개발 및 데이터\\3.개방데이터\\1.데이터\\Validation",
        mode="valid",
        number_filter_rate=0.1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2 if num_workers > 2 else 0,
        collate_fn=collate_fn,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "beomi/kcbert-base", num_labels=1
    )
    model.to(device)

    metric = AccuracyCalculator()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    criterion = nn.BCELoss()

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )

    best_loss = float("inf")
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for data, labels in tqdm(train_loader):
            data, labels = data.long().to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = F.sigmoid(model(data)["logits"])
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += metric(outputs, labels.unsqueeze(1)).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        model.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        with torch.no_grad():
            for data, labels in tqdm(valid_loader):
                data, labels = data.long().to(device), labels.to(device)
                outputs = F.sigmoid(model(data)["logits"])
                loss = criterion(outputs, labels.unsqueeze(1))
                valid_loss += loss.item()
                valid_acc += metric(outputs, labels.unsqueeze(1)).item()

        avg_valid_loss = valid_loss / len(valid_loader)
        avg_valid_acc = valid_acc / len(valid_loader)

        scheduler.step(avg_valid_loss)

        logging.info(
            f"[{epoch+1}/{epochs}]Epoch, Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f} Train Acc: {avg_train_acc:.4f}, Valid Acc: {avg_valid_acc:.4f}"
        )

        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(
                model.state_dict(),
                os.path.join(save_path, "best_loss_model.pth"),
            )
            logging.info(
                f"Best model saved with validation loss: {best_loss:.6f}"
            )

        if avg_valid_acc > best_acc:
            best_acc = avg_valid_acc
            torch.save(
                model.state_dict(),
                os.path.join(save_path, "best_acc_model.pth"),
            )
            logging.info(
                f"Best model saved with validation acc: {best_acc:.4f}"
            )


if __name__ == "__main__":
    train()
