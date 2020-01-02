import time, glob, os, pickle, re, shutil
from os.path import join, basename
from collections import defaultdict
from datetime import datetime
from itertools import cycle, combinations
import numpy as np
import torch, torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Sampler
from model import Backbone
from common import *
from util_funcs import erase_print

def cal_num_similar():
    # 1) for each class, construct graph with tolerence related to age diff.
    # 2) Choose the largest connected subgraph as the master cluster. 
    # 3) For each remaining connected conponent, if its avg age is less than a threshold, deem it dirty.
    # meta
    total = 0
    t = 0.7
    tau = 20
    root = '/data/fuzhuolin/cross_age/data/aligned/mixture/train/'
    save_root = '/data/fuzhuolin/cross_age/data/dirty'
    ds = ImageFolderWithPaths(root, transform=transforms.Compose([
                transforms.Resize((112, 112)), 
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]))
    cs = ClassSampler(ds)
    dl = DataLoader(ds, batch_sampler=cs)
    device = torch.device('cuda:3')
    # pre_trained arcface model
    model_ir_50 = Backbone(50, 0, mode='ir_se')
    model_ir_50.load_state_dict(torch.load('/data/fuzhuolin/cross_age/scripts/InsightFace_Pytorch-master/pretrained_models/model_ir_se50.pth', map_location='cpu'))
    model_ir_50 = model_ir_50.to(device)
    model_ir_50.eval()
    torch.set_grad_enabled(False)
    print('    - Begin sanity check...')
    clean_result = []  # [([[clean_cluster_0], [clean_cluster_1], ...], [[dirty_cluster_0], [dirty_cluster_0], ...])] or [(  [[], []]   ,  [ [], [] ])]
    for i, (inputs, labels, paths) in enumerate(dl): # for each class
        clsname = paths[0].split('/')[-2]
        print(f'- Class {i}, {clsname}')
    #     if not clsname[0].isdigit() or len(clsname) < 5:
    #         continue
    #     assert inputs.size()[0] == len(labels), 'len(inputs) != len(labels)'
    #     assert len(labels) == len(paths), 'len(labels) != len(paths)'
    #     if len(labels) == 0:
    #         continue
        if clsname[0] != 'n':
            continue
        assert (labels[0] == labels).all(), 'inner class labels inconsistent'
        # keep track for each class
        inputs = inputs.to(device)
        # init graph
        cnt_graph = np.zeros((len(labels), len(labels)), dtype=np.int)
        # embeddings
        with torch.no_grad():
            ebds = model_ir_50(inputs)
        # construct graph accding to cos sim
        for (i0, ebd0), (i1, ebd1) in combinations(enumerate(ebds), r=2):
            #age0, age1 = list(map(path_to_age, [paths[i0], paths[i1]]))
            #age_diff = np.abs(age0 - age1)
            #assert 0 <= age0 < 120 and 0 <= age1 < 120
            cos = torch.dot(ebd0, ebd1).clamp(-1, 1).item()
            #print(f'{basename(paths[i0])}, {basename(paths[i1])}, {cos}')
            if cos >= t: #* np.exp(- age_diff / tau):
                cnt_graph[i0, i1] = cnt_graph[i1, i0] = 1
        cpnts = connected_components(cnt_graph, paths)
        outliers = find_outliers(cpnts) # ([[], []], [[], []])
        if outliers:
            for cluster in outliers:
                print(cluster)
                total += len(cluster)
                #for img in cluster:
                   #os.remove(img)
    print()
    print(f'{total} / {len(ds)} , {total/len(ds)*100}% dirty')
    pk_name = join(save_root, root.split('/')[-3]) +f'_cpnts_clean_{t:.2f}_{tau}.pickle'
    #save(clean_result, pk_name)

def save(result, pk_name):
    try:
        with open(pk_name, 'wb') as h:
            pickle.dump(result, h)
            print(f'Saved result at {pk_name}')
    except:
        print('Failed saving result...')
        with open('clean_result.bak.pickle', 'wb') as h:
            pickle.dump(result, h)
            print('Saved results at clean_result.bak.pickle instead')

def find_outliers(cpnts, t=10):
    if len(cpnts) <= 1:
        return []
    outliers = []
    cpnts = sorted(cpnts, key=lambda x: len(x), reverse=True)
    major_group = cpnts[0]
    avg_age_major_group = np.mean(list(map(path_to_age, major_group)))
    for c in cpnts[1:]:
        avg_age_c = np.mean(list(map(path_to_age, c)))
        if np.abs(avg_age_c - avg_age_major_group) < t:
            outliers.append(c)
    return outliers

def path_to_age(path):
    return int(re.split('_|\.', basename(path))[1])

if __name__ == "__main__":
    cal_num_similar()