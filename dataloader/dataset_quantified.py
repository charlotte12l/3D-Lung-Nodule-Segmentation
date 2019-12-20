from torch.utils.data import Dataset
import random
import torch

import numpy as np
from environ import PATH

from utils import rotation, reflection, crop, random_center
from utils.misc import _triple
from utils.util import segment2n_segment


class ClfDataset(Dataset):

    def __init__(self, train, crop_size=32, move=5, public=False, subset=[1], lidc=True,
                 segment_type='max', voxel_segment=False, output_segment=False, patient=False):

        # choose the dataset
        if public:
            self.subset_info = PATH.get_public_info()
            self.get_nodule = PATH.get_public_nodule()
            label_tag = 'label'
        elif lidc:
            if patient:
                info = PATH.get_lidc_patient_info()
            else:
                info = PATH.get_lidc_info()
            # select the corresponding subset split by patients
            sel = 0
            for sset in subset:
                sel = (info['subset'] == 'subset{subset}'.format(subset=sset)) | sel
            self.subset_info = info[sel].copy()
            self.subset_info = self.subset_info.fillna(value=-1)
            self.subset_info = self.subset_info.loc[self.subset_info.remark == -1, :]
            self.get_nodule = PATH.get_lidc_nodule
            if patient:
                self.subset_info = self.subset_info.loc[self.subset_info.malignancy_label != -1, :]
                label_tag = 'malignancy_label'
            else:
                label_tag = 'label'
        else:
            info = PATH.get_info()
            sel = 0
            for sset in subset:
                sel = (info['subset_by_patient'] == sset) | sel
            self.subset_info = info[sel].copy()
            self.get_nodule = PATH.get_nodule
            label_tag = 'EGFR'

        index = self.subset_info.index
        self.train = train
        self.use_lidc = lidc
        self.crop_size = crop_size
        self.voxel_segment = voxel_segment
        self.segment_type = segment_type
        self.output_segment = output_segment
        self.index = tuple(index)  # the index in the info

        self.label = tuple(self.subset_info.loc[self.index, label_tag])

        if self.train:
            self.transform = Transform(crop_size, move)
        else:
            self.transform = Transform(crop_size, None)

    def __getitem__(self, item):
        name = self.index[item]
        with np.load(self.get_nodule(name)) as npz:

            # load the segment data
            if self.voxel_segment or self.output_segment:

                # get segment in n_segment_output
                if self.segment_type == 'average':
                    answers_len = len(npz.files) - 1
                    answer = npz['answer1']
                    for a in range(answers_len - 1):
                        answer += npz['answer{}'.format(a + 2)]
                    answer = answer / answers_len
                    segment_output = answer
                elif self.segment_type == 'max':
                    answers_len = len(npz.files) - 1
                    answer = npz['answer1']
                    for a in range(answers_len - 1):
                        answer = np.logical_or(answer, npz['answer{}'.format(a + 2)])
                    segment_output = answer
                else:
                    assert self.segment_type == 'max' or self.segment_type == 'average'
                    segment_output = np.ones(npz['voxel'].shape)
            else:
                segment_output = np.ones(npz['voxel'].shape)

            # output original voxel in answer
            if self.voxel_segment:
                answer = npz['voxel'] * segment_output
            else:
                answer = npz['voxel']

            # if return segment and voxel
            if self.output_segment:
                voxel, segment_output = self.transform(answer, segment_output)
                voxel = voxel.transpose((3, 0, 1, 2)).astype(np.float32).copy()
                segment_output = segment_output.transpose((3, 0, 1, 2)).astype(np.float32).copy()
            else:
                voxel= self.transform(answer).transpose((3, 0, 1, 2)).astype(np.float32).copy()

        if self.output_segment:
            return voxel, self.label[item], segment_output, name
        else:
            return voxel, self.label[item]

    def __len__(self):
        return len(self.index)


class Transform:
    def __init__(self, size, move=None):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            center = random_center(shape, self.move)
            angle = np.random.randint(4, size=3)
            axis = np.random.randint(4) - 1

            arr_ret = crop(arr, center, self.size)
            arr_ret = rotation(arr_ret, angle=angle)
            arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                aux_ret = reflection(aux_ret, axis=axis)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret


def shuffle_repeat(lst):
    # iterator should have limited size
    total_size = len(lst)
    i = 0

    random.shuffle(lst)
    while True:
        yield lst[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(lst)


if __name__ == '__main__':
    dataset = ClfDataset(crop_size=32, move=5, train=False, subset=[1, 2, 3, 4, 5], lidc=True, voxel_segment=True,
                         output_segment=True)
    x = dataset[0]
    voxel = torch.Tensor(x[0]).unsqueeze(dim=0)
    batch_segment = torch.Tensor(x[2]).unsqueeze(dim=0)
    n_segment = segment2n_segment(batch_segment, n_sat=1024)
    voxel_feature = voxel[n_segment]
    print('Data shape is {}.'.format(x[0].shape))
    print('label is {}.'.format(x[1]))

