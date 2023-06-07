import numpy as np
from PIL import Image

import torchvision
import torch
import random 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
from decimal import Decimal
import csv
from torchvision import datasets


import sys
if '/serenity/data/ppml/' not in sys.path:
    sys.path.append('/serenity/data/ppml/')
if '/serenity/data/ppml/ppml_model_serving/src/' not in sys.path:
    sys.path.append('/serenity/data/ppml/ppml_model_serving/src/')

if '/serenity/data/ppml/ppml_model_serving/src/clockwork/' not in sys.path:
    sys.path.append('/serenity/data/ppml/ppml_model_serving/src/clockwork/')
from clockwork.clockwork import *

from ppml_model_serving.src.model_serving.model_server import *
from ppml_model_serving.src.fingerprinting.fingerprint import *

# from ..train_loader import infer


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_cifar10(root, n_labeled, oracle_model=None,
                 transform_train=None, transform_val=None,
                 download=True, use_cuda=False, which_model=0, server=False, model_zoo=None,
                 fingerprinting=False, clockwork=False, clockwork_acc=None, clockwork_lat=None,
                 direct=False, dataset="CIFAR10", lb=False, latency_bound=100000.0):
    if dataset == "CIFAR10":
        test_dataset = CIFAR10_labeled(root, train=False, transform=transform_val, download=True)
    else:
        print("here")
        test_dataset = SVHN_labeled(root, split="test", transform=transform_val, download=True)
        # with open("current.csv", "w") as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     for X, y in test_dataset:
        #         csvwriter.writerow(X)
                # csvwriter.writerow(y)

        # _ = datasets.SVHN(
        #         root="../data",
        #         split="test",
        #         download=True,
        #         transform=transforms.Compose([
        #             transforms.ToTensor(),
        #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #         ])
        #     )
        
        # with open("modified.csv", "w") as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     for X, y in test_dataset:
        #         csvwriter.writerow(X)
        #         # csvwriter.writerow(y)
    reduction = 0
    server_obj = None
    acc_lat_dict = {}
    if clockwork:
        acc_lat_dict["oracle_acc"] = clockwork_acc[which_model]
        acc_lat_dict["oracle_lat"] = clockwork_lat[which_model]
        if fingerprinting:
            fa = FingerprintAlgo(acc_granularity = "0.01", lat_granularity_in_ms = "1", clockwork=True)
            start_c = fa.clockwork_queries
            models = fa.fingerprint_models_most_accurate_model_first(server_obj, torch.randn([1, 3, 32, 32], dtype=torch.float32).reshape(-1, ).numpy())
            
            ### CHANGE made to get the most accurate model
            max_acc = 0
            max_lat = 0

            for acc, lat in models:
                if lb:
                    if lat >= max_lat and lat <= latency_bound:
                        # max_acc = max(max_acc, acc)
                        max_acc = acc
                        max_lat = lat

                else:
                    max_acc, max_lat = max(max_acc, acc), max(max_lat, lat)
            
            max_lat = max(max_lat, latency_bound)

            if max_lat == 0 and lb == True: max_lat = latency_bound

            acc_lat_dict["attack_acc"], acc_lat_dict["attack_lat"] = max_acc, max_lat
            
            acc_lat_dict["oracle_acc"] = max_acc
            acc_lat_dict["oracle_lat"] = max_lat
            
            end_c = fa.clockwork_queries
            total = end_c-start_c
            print("Queries: ", total)
            # print(models)
            reduction += total
            # return
        else:
            if lb:
                acc_lat_dict["attack_lat"] = latency_bound
                acc_lat_dict["attack_acc"] = 0
            else:
                acc_lat_dict["attack_acc"] = 0
                acc_lat_dict["attack_lat"] = 1000000
        
        print("Crosshair: ", acc_lat_dict)
    
    lat_dict = {'resnet18': 2.350183878559619, 'wide_resnet28': 3.156031652353704, 'resnet34': 4.177080146968365, 'mobilenet_v2': 4.422505168244243, 'mobilenet_v3_small': 4.8054493074305356, 'mobilenet_v3_large': 5.567765949293971, 'resnet50': 6.738397986162454, 'resnet101': 12.2876848699525, 'densenet121': 13.045994964893907, 'densenet169': 17.927423528395593, 'resnet152': 18.52825658256188, 'densenet161': 20.489668818190694}
    acc_dict = {'resnet18': 0.4107636754763368, 'wide_resnet28': 0.44003534111862325, 'resnet34': 0.4844422249539029, 'mobilenet_v2': 0.5122925629993854, 'mobilenet_v3_small': 0.5697218807621389, 'mobilenet_v3_large': 0.6171250768285187, 'resnet50': 0.6522357098955132, 'resnet101': 0.7685156730178242, 'densenet121': 0.8277888752304856, 'densenet169': 0.8979333128457283, 'resnet152': 0.9064612784265519, 'densenet161': 0.919560540872772}

    if oracle_model:
        if server:
            server_obj = ModelServer()
            acc_lat_dict = {}
            for idx, (model_name, model) in enumerate(model_zoo.items()):
                model.cuda()
                model.eval()
                server_obj.register_model(model_name, model)
                print(use_cuda)
                acc, lat = profile_pt_model(server_obj, model_name, test_dataset, device=use_cuda)
                # acc, lat = acc_dict[model_name], lat_dict[model_name]
                print(f"Finished profiling {model_name}. Accuracy: {acc:.4f}, Latency: {lat:.4f} ms")
                server_obj.set_model_accuracy_latency(model_name, acc, lat)
                if idx == which_model:
                    acc_lat_dict["oracle_acc"] = acc
                    acc_lat_dict["oracle_lat"] = lat
            
            print("Model Zoo in server_obj: ")
            for model in server_obj.get_models():
                print(model)

            if fingerprinting:
                fa = FingerprintAlgo(acc_granularity = "0.01", lat_granularity_in_ms = "0.1")
                start_c = server_obj.get_query_count()
                if use_cuda:
                    models = fa.fingerprint_models_most_accurate_model_first(server_obj, torch.randn([1, 3, 32, 32], dtype=torch.float32).cuda())
                else:
                    models = fa.fingerprint_models_most_accurate_model_first(server_obj, torch.randn([1, 3, 32, 32], dtype=torch.float32))
                acc_lat_dict["attack_acc"], acc_lat_dict["attack_lat"] = models[which_model]

                acc_lat_dict["oracle_acc"], acc_lat_dict["oracle_lat"] = models[which_model]
                
                end_c = server_obj.get_query_count()
                total = end_c-start_c
                print("Queries: ", total)
                print(models)
                reduction += total
                # return
            else:
                acc_lat_dict["attack_acc"] = 0
                acc_lat_dict["attack_lat"] = 100

    if dataset == "CIFAR10":
        base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
        train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int((n_labeled-reduction)/10))
        train_labeled_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
        train_unlabeled_dataset = CIFAR10_unlabeled(root, train_unlabeled_idxs, train=True, transform=TransformTwice(transform_train))
        val_dataset = CIFAR10_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
    else:
        base_dataset = torchvision.datasets.SVHN(root, split="train", download=download)
        train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.labels, int((n_labeled-reduction)/10))
        train_labeled_dataset = SVHN_labeled(root, train_labeled_idxs, split="train", transform=transform_train)
        train_unlabeled_dataset = SVHN_unlabeled(root, train_unlabeled_idxs, split="train", transform=TransformTwice(transform_train))
        val_dataset = SVHN_labeled(root, val_idxs, split="train", transform=transform_val, download=True)
    

    oracle_train_dataset, oracle_val_dataset, oracle_test_dataset = None, None, None
    

    # print(len(train_labeled_idxs), len(train_unlabeled_idxs), len(test_dataset))
    # print(len(train_labeled_dataset), len(train_unlabeled_dataset), len(test_dataset))
    # print("Arguments:", oracle_model, server, clockwork)
    # if oracle_model:
    if clockwork:
        print("Infering from Clockwork")
        train_oracle_labels = infer_clockwork(train_labeled_dataset, acc_lat_dict["attack_acc"], int(acc_lat_dict["attack_lat"]))
        train_labeled_idxs = train_labeled_idxs[0:len(train_oracle_labels)]
        print(f"Trained Labeled Idx: {len(train_labeled_idxs)}")
        print(f"Length of labelled dataset: {len(train_labeled_dataset)}")
        print(f"Length of Oracle Labels: {len(train_oracle_labels)}")
        train_actual_oracle_labels = infer(oracle_model, train_labeled_dataset, device=use_cuda)
        # val_oracle_labels = infer_clockwork(val_dataset, acc_lat_dict["oracle_acc"], acc_lat_dict["oracle_lat"])
        # test_oracle_labels = infer_clockwork(test_dataset, acc_lat_dict["oracle_acc"], acc_lat_dict["oracle_lat"])
        val_oracle_labels, _ = infer(oracle_model, val_dataset, device=use_cuda)
        # test_oracle_labels = infer_clockwork(test_dataset, acc_lat_dict["oracle_acc"], int(acc_lat_dict["oracle_lat"]))
        test_oracle_labels, _ = infer(oracle_model, test_dataset, device=use_cuda)
    elif not server:
        train_oracle_labels, _ = infer(oracle_model, train_labeled_dataset, device=use_cuda)
        train_actual_oracle_labels = infer(oracle_model, train_labeled_dataset, device=use_cuda)
        if type(oracle_model) is list:
            val_oracle_labels, _ = infer(oracle_model[which_model], val_dataset, device=use_cuda)
            test_oracle_labels, _ = infer(oracle_model[which_model], test_dataset, device=use_cuda)
        else:
            val_oracle_labels, _ = infer(oracle_model, val_dataset, device=use_cuda)
            test_oracle_labels, _ = infer(oracle_model, test_dataset, device=use_cuda)
    else:
        print(acc_lat_dict)
        if direct:
            train_oracle_labels = infer_server(server_obj, train_labeled_dataset, acc_lat_dict["oracle_acc"], acc_lat_dict["oracle_lat"])
        else:
            train_oracle_labels = infer_server(server_obj, train_labeled_dataset, acc_lat_dict["attack_acc"], acc_lat_dict["attack_lat"])
        train_actual_oracle_labels = infer_server(server_obj, train_labeled_dataset, acc_lat_dict["oracle_acc"], acc_lat_dict["oracle_lat"])
        val_oracle_labels = infer_server(server_obj, val_dataset, acc_lat_dict["oracle_acc"], acc_lat_dict["oracle_lat"])
        test_oracle_labels = infer_server(server_obj, test_dataset, acc_lat_dict["oracle_acc"], acc_lat_dict["oracle_lat"])

        # test_oracle_labels
    if dataset == "CIFAR10":
        oracle_train_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train, oracle_labels=train_oracle_labels, oracle=True)
        oracle_actual_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train, oracle_labels=train_actual_oracle_labels, oracle=True)
        oracle_val_dataset = CIFAR10_labeled(root, val_idxs, train=True, transform=transform_val, oracle_labels=val_oracle_labels, oracle=True)
        oracle_test_dataset = CIFAR10_labeled(root, train=False, transform=transform_val, oracle_labels=test_oracle_labels, oracle=True)
    else:
        oracle_train_dataset = SVHN_labeled(root, train_labeled_idxs, split="train", transform=transform_train, oracle_labels=train_oracle_labels, oracle=True)
        oracle_actual_dataset = SVHN_labeled(root, train_labeled_idxs, split="train", transform=transform_train, oracle_labels=train_actual_oracle_labels, oracle=True)
        oracle_val_dataset = SVHN_labeled(root, val_idxs, split="train", transform=transform_val, oracle_labels=val_oracle_labels, oracle=True)
        oracle_test_dataset = SVHN_labeled(root, split="test", transform=transform_val, oracle_labels=test_oracle_labels, oracle=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, oracle_train_dataset, oracle_actual_dataset, oracle_val_dataset, oracle_test_dataset
    

def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

# cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
# cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)
global_cifar_flag = True

def normalize(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    print("normalize", x.shape, mean.shape)

    if not global_cifar_flag:
        x -= (mean*255)[:, None, None]
        x *= 1.0/((255*std)[:, None, None])
    else:
        x -= (mean*255)
        x *= 1.0/(255*std)

    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size
        # print(h, w)
        # print(x.shape, self.output_size, new_h, new_w)
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x

class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x

class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x

class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False, oracle_labels = None, oracle=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            if oracle == False:
                self.targets = np.array(self.targets)[indexs]
            else:
                self.targets = np.array(oracle_labels)
        # print(self.data.shape)
        self.data = transpose(normalize(self.data))
        # print(self.data.shape)
        if oracle:
            self.targets = np.array(oracle_labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            # print(img.shape)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])



class SVHN_labeled(torchvision.datasets.SVHN):

    def __init__(self, root, indexs=None, split="train",
                 transform=None, target_transform=None,
                 download=False, oracle_labels = None, oracle=False):
        super(SVHN_labeled, self).__init__(root, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            if oracle == False:
                self.labels = np.array(self.labels)[indexs]
            else:
                self.labels = np.array(oracle_labels)
        print("Init before norm shape:", self.data.shape)
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        self.data = transpose(normalize(self.data))
        if oracle:
            self.labels = np.array(oracle_labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # print(img.shape)
        # img = np.transpose(img, (1, 0, 2))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

class SVHN_unlabeled(SVHN_labeled):

    def __init__(self, root, indexs, split="train",
                 transform=None, target_transform=None,
                 download=False):
        super(SVHN_unlabeled, self).__init__(root, indexs, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.labels = np.array([-1 for i in range(len(self.labels))])


def infer(model, dataloader, device, debug=False):
    if type(model) is list:
        for m in model:
            m.eval()
            if device:
                m.cuda()
    else:
        model.eval()
        if device:
            model.cuda()
    labels = []
    scores = []
    with torch.no_grad():
        for idx, (X, _) in enumerate(dataloader):
            # if type(X) == list: X = X[0]
            # print(idx) 
            if device:
                X = torch.Tensor(X).cuda().unsqueeze(0)
            if type(model) is list:
                pred = (random.choice(model))(X)
                # print(random.choice(model))
            else:
                pred = model(X)
            labels_batch = pred.cpu().numpy().argmax(1)
            scores_batch = pred.cpu().numpy().max(1)
            for label, score in zip(labels_batch, scores_batch):
                labels.append(label)
                scores.append(score)
    return labels, scores    


def infer_server(server, dataloader, acc, lat):
    labels = []
    correct = 0
    total_size = 0
    for idx, (X, y) in enumerate(dataloader):
        X = X.cuda().unsqueeze(0)
        output = server.query(X, acc, lat, "random")
       
        # output = s
        # print(type(output))
        # print(output)
        # print(output.cpu().numpy().argmax(1).item())
        labels.append(output.cpu().numpy().argmax(1).item())
        if (y == output.cpu().numpy().argmax(1).item()):
            correct += 1
        total_size += 1
    print(server.num_model_triggered)

    print("Accuracy: ", correct/total_size)
    return labels

def infer_clockwork(dataloader, acc, lat):
    labels = []
    correct = 0
    total_size = 0
    num_q = len(dataloader)
    for idx, (X, corr_label) in enumerate(dataloader):
        response = None
        
        while (response is None or (response.success == False and num_q > 0)):
            input_data = X
            # print(acc, lat)
            # print(input_data.shape)
            request = InferenceRequest(acc, lat, input_data.numpy().reshape(-1, ))
            num_q-=1
            request.send()

            response = InferenceResponse.receive()
        if (response.success == False and num_q <= 0): break
        # print(np.array(response.output).argmax(0), corr_label)
        labels.append(np.array(response.output).argmax(0).item())
        if np.array(response.output).argmax(0).item() == corr_label:
            correct += 1
        total_size += 1
        if (num_q <= 0): break
    
    dataloader.data = dataloader.data[0:len(labels)]

    
    print("Accuracy: ", correct/total_size, correct, total_size)
    # print(labels)
    return labels
         


def profile_pt_model(server: ModelServer, model_id: str, test_data, latency_sample=1000, device="cpu"):
    """
    Profile a pytorch model using all data in test_data and measure latency by 
    performing at most `latency_sample` number of queries.
    """
    print(f"Start profiling {model_id} ...")

    size = len(test_data)
    correct = 0
    with torch.no_grad():
        for X, y in DataLoader(test_data, batch_size=256):
            X,y = X.cuda(), y.cuda()
            # print("here: ", device)
            # print(X.get_device(), y.get_device())
            # print("profiling shape:", X.shape)
            pred = server.query_from_model(X, model_id)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    latencies = []
    for X, y in DataLoader(test_data, batch_size=1):
        X, y = X.cuda(), y.cuda()
        start = time.perf_counter()
        pred = server.query_from_model(X, model_id)
        latencies.append((time.perf_counter() - start) * 1000) # convert to ms
        if len(latencies) >= latency_sample:
            break

    # plt.cla()
    # plt.hist(latencies, 20)
    # plt.title(f"Latencies PDF for {model_id}")
    # plt.xlabel("Latency (ms)")
    # plt.savefig(f"figures/Latencies_PDF_for_{model_id}.png")

    # plt.cla()
    # plt.hist(latencies, histtype='step', cumulative=True)
    # plt.title(f"Latencies CDF for {model_id}")
    # plt.xlabel("Latency (ms)")
    # plt.savefig(f"figures/Latencies_CDF_for_{model_id}.png")

    accuracy = correct / size
    latency = np.mean(latencies)
    return accuracy, latency