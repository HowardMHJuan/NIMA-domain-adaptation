import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from model.model import NIMA, Encoder, Classifier, Discriminator, emd_loss
from dataset.dataset import AVADataset, MEDataset, GroupDataset


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(6174)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- Step 1 ---
    encoder = Encoder()
    classifier = Classifier()
    discriminator = Discriminator(h_feat=1024)

    state_dict = torch.load("saved/model.pth")
    encoder_state_dict = {key: val for key, val in state_dict.items() if key.split(".")[0] == "features"}
    classifier_state_dict = {key: val for key, val in state_dict.items() if key.split(".")[0] == "classifier"}
    encoder.load_state_dict(encoder_state_dict)
    classifier.load_state_dict(classifier_state_dict)

    encoder.to(device)
    classifier.to(device)
    discriminator.to(device)
    # --- ---

    transform = transforms.Compose([
        transforms.Scale(256), 
        # transforms.Resize((224, 224)),
        transforms.RandomCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor()
    ])
    s_set = AVADataset(csv_file="data/AVA/train_label.csv", root_dir="/tmp/r09922083/AVA/images", transform=transform)
    t_set = MEDataset(csv_file="data/ME/label.csv", root_dir="data/ME/images", transform=transform)

    s_loader = DataLoader(s_set, batch_size=len(t_set), shuffle=True, num_workers=16)
    t_loader = DataLoader(t_set, batch_size=len(t_set), shuffle=True, num_workers=16)

    group_set = GroupDataset(s_loader, t_loader, len(t_set))

    # --- Step 2 ---
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    encoder.eval()
    classifier.eval()
    discriminator.train()
    for epoch in range(10):
        group_set.update_and_process()
        group_loader = DataLoader(group_set, batch_size=30, shuffle=True, num_workers=16)

        loss_list = []
        for data in group_loader:
            group_list = [
                data["G1"],
                data["G2"],
                data["G3"],
                data["G4"],
            ]
            for yg, group in enumerate(group_list):
                x1, x2 = group
                x1 = x1.to(device)
                x2 = x2.to(device)
                yg = torch.full((len(x1),), yg, dtype=torch.long, device=device)

                optimizer_d.zero_grad()

                feat = torch.cat([encoder(x1), encoder(x2)], dim=1)
                yg_pred = discriminator(feat)

                loss = loss_fn(yg_pred, yg)
                loss.backward()
                optimizer_d.step()

                loss_list.append(loss.item())
        
        print(f"Step 2 | epoch {epoch + 1} | loss: {sum(loss_list) / len(loss_list)}")
    # --- ---

    # --- Step 3 ---
    optimizer_gh = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
 
    for epoch in range(100):
        # --- train g and h, d frozen
        group_set.update_and_process()
        group_loader = DataLoader(group_set, batch_size=30, shuffle=True, num_workers=16)

        loss_list = []
        encoder.train()
        classifier.train()
        discriminator.eval()
        for data in group_loader:
            group_list = [
                data["G1"],
                data["G2"],
                data["G3"],
                data["G4"],
            ]
            y_list = [
                data["Y1"],
                data["Y2"],
                data["Y3"],
                data["Y4"],
            ]
            for yg, (group, y) in enumerate(zip(group_list, y_list)):
                if yg == 0 or yg == 2:
                    continue
                x1, x2 = group
                x1 = x1.to(device)
                x2 = x2.to(device)
                y1, y2 = y
                y1 = y1.to(device)
                y2 = y2.to(device)
                yg = torch.full((len(x1),), yg, dtype=torch.long, device=device)

                optimizer_gh.zero_grad()

                feat_1 = encoder(x1)
                feat_2 = encoder(x2)
                feat_cat = torch.cat([feat_1, feat_2], dim=1)

                y1_pred = classifier(feat_1).view(-1, 10, 1)
                y2_pred = classifier(feat_2).view(-1, 10, 1)
                yg_pred = discriminator(feat_cat)

                loss_1 = emd_loss(y1_pred, y1)
                loss_2 = emd_loss(y2_pred, y2)
                loss_d = loss_fn(yg_pred, yg)

                loss_sum = loss_1 + loss_2 + 0.2*loss_d
                loss_sum.backward()

                nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(classifier.parameters()), 2)
                optimizer_gh.step()

                loss_list.append(loss_sum.item())

        print(f"Step 3 | epoch {epoch + 1} | GH | loss: {sum(loss_list) / len(loss_list)}")
        
        encoder.eval()
        classifier.eval()
        discriminator.train()
        loss_list = []
        for data in group_loader:
            group_list = [
                data["G1"],
                data["G2"],
                data["G3"],
                data["G4"],
            ]
            for yg, group in enumerate(group_list):
                x1, x2 = group
                x1 = x1.to(device)
                x2 = x2.to(device)
                yg = torch.full((len(x1),), yg, dtype=torch.long, device=device)

                optimizer_d.zero_grad()

                feat = torch.cat([encoder(x1), encoder(x2)], dim=1)
                yg_pred = discriminator(feat)

                loss = loss_fn(yg_pred, yg)
                loss.backward()

                nn.utils.clip_grad_norm_(discriminator.parameters(), 2)
                optimizer_d.step()

                loss_list.append(loss.item())
        
        print(f"Step 3 | epoch {epoch + 1} | D | loss: {sum(loss_list) / len(loss_list)}")

    torch.save(encoder.state_dict(), "saved/encoder.pth")
    torch.save(classifier.state_dict(), "saved/classifier.pth")
    torch.save(discriminator.state_dict(), "saved/discriminator.pth")

    #testing
    # acc = 0
    # for data, labels in test_dataloader:
    #     data = data.to(device)
    #     labels = labels.to(device)
    #     y_test_pred = classifier(encoder(data))
    #     acc += (torch.max(y_test_pred, 1)[1] == labels).float().mean().item()

    # accuracy = round(acc / float(len(test_dataloader)), 3)

    # print("step3----Epoch %d/%d  accuracy: %.3f " % (epoch + 1, opt['n_epoches_3'], accuracy))
