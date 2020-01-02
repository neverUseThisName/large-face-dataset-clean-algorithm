import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler
from torchvision.datasets import ImageFolder

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class ClassSampler(Sampler):
    def __init__(self, ds):
        self.ds = ds
        self.label2idxs = defaultdict(list)
        for idx, label in enumerate(self.ds.targets):
            self.label2idxs[label].append(idx)

    def __iter__(self):
        for idxs in self.label2idxs.values():
            yield idxs
    
    def __len__(self):
        return len(self.ds)

############################################################
##################### graph  related #######################
############################################################

def DFS(mat, i, node_names, visited, cnt_cpnt):
    '''
    mat is like:
    0101
    0001
    0010
    1000
    '''
    visited[node_names[i]] = True
    cnt_cpnt.append(node_names[i])
    for j in np.nonzero(mat[i])[0]:
        if not visited[node_names[j]]:
            DFS(mat, j, node_names, visited, cnt_cpnt)

def connected_components(mat, node_names):
    assert mat.shape[0] == mat.shape[1] and mat.shape[0] == len(node_names)
    visited = dict.fromkeys(node_names, False)
    cpnts = []
    for i, name in enumerate(node_names):
        if not visited[name]:
            cpnt = []
            DFS(mat, i, node_names, visited, cpnt)
            cpnts.append(cpnt)
    return cpnts