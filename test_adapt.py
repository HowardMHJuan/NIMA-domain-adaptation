import argparse
import torch
import torchvision.transforms as transforms
from torch import nn
from PIL import Image
from pathlib import Path

from model.model import Encoder, Classifier, Discriminator, emd_loss


class Inference():
    def __init__(self, device):
        self._encoder = Encoder()
        self._classifier = Classifier()

        self._encoder.load_state_dict(torch.load("saved/encoder.pth"))
        self._classifier.load_state_dict(torch.load("saved/classifier.pth"))

        self._encoder.to(device)
        self._classifier.to(device)

        self._encoder.eval()
        self._classifier.eval()

        self._device = device

    def infer(self, batch):
        batch = batch.to(self._device)

        with torch.no_grad():
            feat = self._encoder(batch)
            pred = self._classifier(feat)

        pred = pred.cpu()

        score_mean = torch.zeros(pred.size(0))
        score_std = torch.zeros(pred.size(0))
        for score in range(10):
            score_mean += score * pred[:, score]
        for score in range(10):
            score_std += pred[:, score] * (score - score_mean) ** 2
        score_std = score_std ** 0.5

        return score_mean, score_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str)
    parser.add_argument('--img-path', type=str)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(6174)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transform = transforms.Compose([
        # transforms.Scale(256), 
        transforms.Resize((224, 224)), 
        # transforms.RandomCrop(224), 
        transforms.ToTensor()
    ])

    inference = Inference(device)

    if args.img_path:
        img = Image.open(args.img_path)
        img = img.convert("RGB")
        img = transform(img).unsqueeze(0)
        mean, std = inference.infer(img)
        print(f"{args.img_path} -- mean: {round(mean.item(), 2)}, std: {round(std.item(), 2)}")

    if args.img_dir:
        img_list = []
        img_path_list = []
        # for img_path in sorted(Path(args.img_dir).iterdir()):
        for i in range(1, 101):
            img_path = f"/tmp/CC2020/train_data/train_{i}.png"
            try:
                img = Image.open(img_path)
            except BaseException:
                continue
            img = img.convert("RGB")
            img = transform(img)
            img_list.append(img)
            img_path_list.append(img_path)

        print(img_path_list)
        while img_list:
            img_batch = img_list[:16]
            img_path_batch = img_path_list[:16]

            img_batch = torch.stack(img_batch)
            mean, std = inference.infer(img_batch)
            for i, img_path in enumerate(img_path_batch):
                # print(f"{img_path} -- mean: {round(mean[i].item(), 2)}, std: {round(std[i].item(), 2)}")
                print(f"{round(mean[i].item(), 2)}")

            img_list = img_list[16:]
            img_path_list = img_path_list[16:]