# -*- coding: utf-8 -*-
import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torch import nn

import os
import csv
import json
from tqdm import tqdm
import pickle
import typing
import cv2

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsClassifier

# import seaborn as sns
import matplotlib.pyplot as plt
# from IPython.display import clear_output
import torch.nn.functional as F


CLASSES_CNT = 205
CUDA = 0


class DatasetRTSD(Dataset):
    """
    class for reading and 
    :param root_folders: 
    :param path_to_classes_json: 
    """
    def __init__(self, root_folders, path_to_classes_json) -> None:
        super(DatasetRTSD, self).__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        self.samples = []  # tuple (path to image, index of cluster)
        self.classes_to_samples = {}  # dict of images
        self.transform = A.Compose([
            A.Resize(224, 224),
            # A.CenterCrop(224),
            A.HorizontalFlip(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

        for class_name in self.classes:
            self.classes_to_samples[self.class_to_idx[class_name]] = []

        for root in root_folders:
            for class_dir in os.listdir(root):
                for image in os.listdir(os.path.join(root, class_dir)):
                    self.samples.append((os.path.join(root, class_dir, image), self.class_to_idx[class_dir]))
                    self.classes_to_samples[self.class_to_idx[class_dir]].append(len(self.samples) - 1)

    def __getitem__(self, index):
        """
        Returns triple (tensor with image, path to file, number of class file [if there is no label, then -1])
        """
        image_path, class_idx = self.samples[index]
        img = np.array(Image.open(image_path).convert('RGB'))
        img = self.transform(image=img)
        return img, image_path, class_idx

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def get_classes(path_to_classes_json):
        """
        Reads the info from classes.json the information about classes
        :param path_to_classes_json:
        """
        class_to_idx = {}
        classes = []
        fd = open(path_to_classes_json, 'r')
        for key, item in json.load(fd).items():
            class_to_idx[key] = item['id']
            classes.append(key)
        return classes, class_to_idx


class TestData(Dataset):
    """
    A class to read and store the test dataset.
    :param root: path to the folder with pictures of characters
    :param path_to_classes_json: path to classes.json
    :param annotations_file: path to .csv file with annotations (optional)
    """
    def __init__(self, root, path_to_classes_json, annotations_file=None):
        super(TestData, self).__init__()
        self.classes, self.class_to_idx = DatasetRTSD.get_classes(path_to_classes_json)
        self.root = root
        self.samples = []  # paths to images

        for image in os.listdir(self.root):
            self.samples.append(image)

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

        self.targets = None 
        if annotations_file is not None:
            self.targets = {}
            with open(annotations_file, newline='') as csvfile:
                rd = csv.reader(csvfile, delimiter=',')
                next(rd)
                for row in rd:
                    self.targets[row[0]] = self.class_to_idx[row[1]]

    def __getitem__(self, index):
        """
        Returns a triple: image tensor, file path, file class number (if no markup, "-1").
        """
        image_path = self.samples[index]
        img = np.array(Image.open(os.path.join(self.root, image_path)).convert('RGB'))
        img = self.transform(image=img)
        if self.targets:
            return img, image_path, self.targets.get(image_path, -1)
        else:
            return img, image_path, -1

    def __len__(self):
        return len(self.samples)


class CustomNetwork(torch.nn.Module):
    """
    A class that implements a neural network for classification.
    :param features_criterion: loss-function on the features extracted by the neural network before classification (None when there is no such loss)
    :param internal_features: internal number of features
    """
    def __init__(self, features_criterion=None, internal_features=1024):
        super().__init__()

        self.criterion = features_criterion
        self.model = resnet50(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, internal_features),
            nn.ReLU(),
            nn.Linear(internal_features, CLASSES_CNT)
        )
        for child in list(self.model.children()):
            for param in child.parameters():
                param.requires_grad = False

        device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        """
        Function for predicting class-response. Returns an np array with class indices.
        :param x: batch with images
        """
        logits = self.model(x)
        return logits.argmax(dim=1)


def training_epoch(model, optimizer, criterion, train_loader, tqdm_desc, device):
    train_loss, train_accuracy = 0.0, 0.0
    model.train()
    for images, paths, labels in tqdm(train_loader, desc=tqdm_desc):
        images = images['image']
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.shape[0]
        train_accuracy += (logits.argmax(dim=1) == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
    return train_loss, train_accuracy


def full_train(model, optimizer, scheduler, criterion, train_loader, test_loader, num_epochs, device):
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}',
            device=device
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        train_accuracies += [train_accuracy]
        # plot_losses(train_losses, test_losses, train_accuracies, test_accuracies)

    return train_losses, test_losses, train_accuracies, test_accuracies


def train_simple_classifier():
    """A function for training a simple classifier on raw data.""""
    print(f'using CUDA:{CUDA}')
    num_epochs = 3

    train_dataset = DatasetRTSD(root_folders=['./additonal_files/cropped-train'],
                                path_to_classes_json='./additonal_files/classes.json')

    test_dataset = TestData(root='./additonal_files/smalltest',
                            path_to_classes_json='./additonal_files/classes.json',
                            annotations_file='./additonal_files/smalltest_annotations.csv')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
    model = CustomNetwork()
    lr = 1e-3
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    full_train(model, optimizer, scheduler, criterion, train_loader, test_loader, num_epochs, device=device)
    torch.save(model.state_dict(), "simple_model.pth")
    return model


def apply_classifier(model, test_folder, path_to_classes_json):
    """
    A function that applies a model and gets its predictions.
    :param model: the model to be tested
    :param test_folder: path to the folder with the test data
    :param path_to_classes_json: path to the file with class information classes.json
    """
    test_dataset = TestData(root=test_folder,
                            path_to_classes_json=path_to_classes_json)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
    results = []
    model.eval()
    for img, img_path, img_class in test_loader:
        sign_class = model.predict(img['image'].to('cpu')).cpu().detach().numpy().ravel().item()
        results.append({
            'filename': img_path[0],
            'class': test_dataset.classes[sign_class]
        })
    return results


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        reader = csv.DictReader(fhandle)
        for row in reader:
            res[row['filename']] = row['class']
    return res


def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == 'all' or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt)


def test_classifier(model, test_folder, path_to_classes_json, annotations_file):
    """
    Function for testing the quality of a model.
    Returns Precision on all signs, Recall on rare signs and Recall on frequent signs.
    :param model: model to be tested
    :param test_folder: path to the folder with test data
    :param annotations_file: path to .csv file with annotations (optional)
    """
    output = apply_classifier(model, test_folder, path_to_classes_json)
    output = {elem['filename']: elem['class'] for elem in output}
    gt = read_csv(annotations_file)
    y_pred = []
    y_true = []

    for k, v in output.items():
        y_pred.append(v)
        y_true.append(gt[k])

    with open(path_to_classes_json, "r") as fr:
        classes_info = json.load(fr)
    class_name_to_type = {k: v['type'] for k, v in classes_info.items()}

    total_acc = calc_metric(y_true, y_pred, 'all', class_name_to_type)
    rare_recall = calc_metric(y_true, y_pred, 'rare', class_name_to_type)
    freq_recall = calc_metric(y_true, y_pred, 'freq', class_name_to_type)
    return total_acc, rare_recall, freq_recall


def motion_blur(img, random_angle, kernel_size=5):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel /= kernel_size
    kernel = A.rotate(kernel, angle=random_angle)
    img = cv2.filter2D(img, -1, kernel)
    return img


def generate_sign(icon_path):
    img = Image.open(icon_path)
    if img.mode == 'LA':
        mask_channel = 1
    else:
        mask_channel = 3
    sign_mask = np.array(img)[..., mask_channel]
    img = np.array(img.convert('RGB'))

    random_size = np.random.choice(np.arange(16, 129))
    random_pad_percent = np.random.choice(np.arange(0, 16)) / 100

    transform1 = A.Compose([
        A.Resize(random_size, random_size),
        A.CropAndPad(percent=random_pad_percent),
        A.ColorJitter(hue=0.2, contrast=(1, 1), p=1),
        A.Rotate(limit=(-15, 15))
    ])

    gaussian_kernel_size = max(3, (1 - int(random_size * 0.1) % 2) + int(random_size * 0.1))
    transform2 = A.Compose([
        A.GaussianBlur(blur_limit=(1, gaussian_kernel_size), p=0.3)
    ])
    transformed = transform1(image=img, mask=sign_mask)
    # transformed = transform2(image=transformed['image'], mask=transformed['mask'])
    sign_mask = transformed['mask']
    img = transform2(image=transformed['image'])['image']

    random_angle = np.random.choice(np.arange(-90, 90))
    img = motion_blur(img, random_angle, random_size // 15)

    # plt.axis('off')
    # plt.imshow(img * np.dstack((sign_mask, sign_mask, sign_mask)).astype('bool'))
    return img, sign_mask.astype('bool')


def fuse_with_background(sign_img, sign_mask, background_path):
    background_img = np.array(Image.open(background_path).convert('RGB'))
    crop_size = int(sign_img.shape[0] * 1.05)
    background_img = A.RandomCrop(crop_size, crop_size)(image=background_img)['image']

    offset = (background_img.shape[0] - sign_img.shape[0]) // 2
    y1, y2 = offset, offset + sign_img.shape[0]
    x1, x2 = offset, offset + sign_img.shape[0]

    background_mask = ~sign_mask
    for c in range(0, 3):
        background_img[y1:y2, x1:x2, c] = (sign_mask * sign_img[:, :, c] +
                                           background_mask * background_img[y1:y2, x1:x2, c])
    return background_img


def generate_one_icon(args):
    """
    A function that generates synthetic data for a single class.
    :param args: This is a list of parameters: [path to icon file, path to output folder, path to backgrounds folder, number of examples of each class]
    """
    icon_path, output_folder, backgrounds_folder_path, sample_quantity = args

    icon_class = os.path.split(icon_path)[1][:-4]
    icon_output_folder = os.path.join(output_folder, icon_class)
    if not os.path.exists(icon_output_folder):
        os.makedirs(icon_output_folder)

    backgrounds_files = os.listdir(backgrounds_folder_path)
    for i in range(sample_quantity):
        sign_img, sign_mask = generate_sign(icon_path)
        bg_number = np.random.choice(len(backgrounds_files))
        plt.imsave(os.path.join(icon_output_folder, f'{icon_class}_{i}.png'),
                   fuse_with_background(sign_img, sign_mask, os.path.join(backgrounds_folder_path, backgrounds_files[bg_number])))


def generate_all_data(output_folder, icons_path, background_path, samples_per_class=1000):
    """
    A function that generates synthetic data.
    This function starts a pool of parallel running processes, each of which will generate an icon of a different type.
    This is necessary because the generation process is very long.
    Each process runs in the function generate_one_icon.
    :param output_folder: Path to the output directory
    :param icons_path: Path to the icon directory
    :param background_path: Path to the directory with background images
    :param samples_per_class: Number of samples of each class to generate
    """
    with ProcessPoolExecutor(8) as executor:
        params = [[os.path.join(icons_path, icon_file), output_folder, background_path, samples_per_class]
                  for icon_file in os.listdir(icons_path)]
        list(tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier():
    """A function for training a simple classifier on a mixture of raw and sythetic data.""""

    print(f'using CUDA:{CUDA}')
    num_epochs = 1

    train_dataset = DatasetRTSD(root_folders=['./additonal_files/cropped-train', './synthetic_samples'],
                                path_to_classes_json='./additonal_files/classes.json')

    # test_dataset = TestData(root='./additonal_files/smalltest',
    #                        path_to_classes_json='./additonal_files/classes.json',
    #                        annotations_file='./additonal_files/smalltest_annotations.csv')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
    model = CustomNetwork()
    lr = 1e-3
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    full_train(model, optimizer, scheduler, criterion, train_loader, None, num_epochs, device=device)
    torch.save(model.state_dict(), "simple_model_with_synt.pth")
    return model


class FeaturesLoss(torch.nn.Module):
    """
    A class for computing a loss function on the features of the penultimate layer of a neural network.
    """
    def __init__(self, margin):
        super(FeaturesLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, outputs, labels):
        output1, output2 = outputs
        label1, label2 = labels
        distances = (output2 - output1).square().sum(1)
        distances_neg = F.relu(self.margin - (distances + self.eps).sqrt()).square()
        losses = 0.5 * torch.where(label1 == label2, distances, distances_neg)
        return losses.mean()


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    A class to sample batches with a controlled number of classes and examples of each class.
    :param data_source: This is the RTSD dataset
    :param elems_per_class: Number of elements of each class
    :param classes_per_batch: The number of different classes in a single batches
    """
    def __init__(self, data_source, elems_per_class, classes_per_batch):
        self.training_data = data_source.samples
        self.class_count = len(data_source.classes)
        self.classes_to_samples = data_source.classes_to_samples
        self.elems_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch
        self.batch_size = elems_per_class * classes_per_batch

        # self.training_label = data_source.train_labels.to(device)
        # self.training_data = self.training_data.type(torch.cuda.FloatTensor)

    def __iter__(self):
        samples = []
        for clas_idx in np.random.choice(self.class_count, size=self.classes_per_batch, replace=False):
            samples_idx = np.random.choice(self.classes_to_samples[clas_idx],
                                           size=self.elems_per_class)
            samples += list(samples_idx)
        yield samples

    def __len__(self):
        return self.batch_size
