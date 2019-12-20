import collections
import itertools
import numpy as np


def get_uniform_assign(length, subset):
    assert subset > 0
    per_length, remain = divmod(length, subset)
    total_set = np.random.permutation(list(range(subset)) * per_length)
    remain_set = np.random.permutation(list(range(subset)))[:remain]
    return list(total_set) + list(remain_set)


def split_validation(df, subset, by):
    df = df.copy()
    for sset in df[by].unique():
        length = (df[by] == sset).sum()
        df.loc[df[by] == sset, 'subset'] = get_uniform_assign(length, subset)
    df['subset'] = df['subset'].astype(int)
    return df


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(itertools.repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class ConfigBase:

    def __init__(self, **kwargs):
        self._parse(kwargs)

    def _parse(self, setting):
        user_dict = self._get_user_dict(serializable=False)
        for k, v in setting.items():
            if k not in user_dict:
                raise ValueError('Invalid Option: "--%s"' % k)
            src_value = getattr(self, k)
            setattr(self, k, type(src_value)(v))
            print("Setting `%s`: %s->%s" % (k, src_value, getattr(self, k)))

    def _get_user_dict(self, serializable=False):
        if serializable:
            return {k: str(getattr(self, k)) for k in dir(self)
                    if not k.startswith('_')}
        return {k: getattr(self, k) for k in dir(self)
                if not k.startswith('_')}
