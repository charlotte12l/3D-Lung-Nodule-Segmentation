import os
import json


class PathManager:
    def __init__(self, environ_path=None):
        self.environ = parse_environ(environ_path)

    @property
    def egfr_dataset(self):
        return self.environ['EGFR_DATASET']

    @property
    def lidc_dataset(self):
        return self.environ['LIDC_DATASET']

    @property
    def training_path(self):
        return self.environ['TRAINING']

    @property
    def raw_path(self):
        return os.path.join(self.egfr_dataset, 'EGFR_NII_DNA')

    def get_raw(self, patient):
        return os.path.join(self.raw_path, patient)

    @property
    def processed_path(self):
        return os.path.join(self.egfr_dataset, 'processed')

    @property
    def public_path(self):
        return os.path.join(self.egfr_dataset, 'public')

    @property
    def lidc_nodule_path(self):
        return os.path.join(self.lidc_dataset, 'nodule')

    @property
    def lidc_info_path(self):
        return os.path.join(self.lidc_dataset, 'info')

    @property
    def raw_info(self):
        return os.path.join(self.raw_path, 'EGFR_info.csv')

    @property
    def step1_info(self):
        return os.path.join(self.processed_path, 'step1_info.csv')  # clean raw

    @property
    def step2_info(self):
        return os.path.join(self.processed_path, 'step2_info.csv')  # info without subset split

    @property
    def info(self):
        return os.path.join(self.processed_path, 'info_plos.csv')  # info with subset split

    def get_info(self):
        import pandas as pd
        df = pd.read_csv(self.info, index_col=0)
        return df

    @property
    def public_info(self):
        return os.path.join(self.public_path, 'public_plos.csv')  # info with subset split

    def get_public_info(self):
        import pandas as pd
        df = pd.read_csv(self.public_info, index_col=0)
        return df

    @property
    def past_lidc_info(self):
        return os.path.join(self.lidc_info_path, 'resplit_by_patient_V3.csv')  # info with subset split

    @property
    def lidc_info(self):#xiugai
        return os.path.join(self.lidc_dataset, 'quantified_nodules.csv')  # info with subset split

    @property
    def lidc_patient_info(self):
        return os.path.join(self.lidc_info_path, 'quantified_nodules_by_patient.csv')  # info with patient split

    def get_past_lidc_info(self):
        import pandas as pd
        df = pd.read_csv(self.past_lidc_info, index_col=0)
        return df

    def get_lidc_info(self):
        import pandas as pd
        df = pd.read_csv(self.lidc_info, index_col=0)
        return df

    def get_lidc_patient_info(self):
        import pandas as pd
        df = pd.read_csv(self.lidc_patient_info, index_col=0)
        return df

    @property
    def radiomics(self):
        return os.path.join(self.processed_path, 'radiomics_plos.csv')

    @property
    def icc(self):
        return os.path.join(self.processed_path, 'icc_plos.csv')

    def get_selected_features(self, thresh):
        import pandas as pd
        icc = pd.read_csv(self.icc, index_col=0)
        selected_features = list(icc[icc['icc'] > thresh].index)
        return selected_features

    def get_radiomics(self, thresh=None):
        import pandas as pd
        df = pd.read_csv(self.radiomics, index_col=0)
        if thresh is not None:
            selected_features = self.get_selected_features(thresh)
            return df[selected_features + ['subset', 'label']]
        return df

    @property
    def public_radiomics(self):
        return os.path.join(self.public_path, 'public_radiomics_plos.csv')

    def get_public_radiomics(self, thresh=None):
        import pandas as pd
        df = pd.read_csv(self.public_radiomics, index_col=0)
        if thresh is not None:
            selected_features = self.get_selected_features(thresh)
            return df[selected_features + ['label']]
        return df

    @property
    def selected_radiomics(self):
        return os.path.join(self.processed_path, 'selected_radiomics.csv')

    def get_selected_radiomics(self):
        import pandas as pd
        df = pd.read_csv(self.selected_radiomics, index_col=0)
        return df

    def get_case(self, patient):
        return os.path.join(self.processed_path, "case", "{patient}.npz".format(patient=patient))

    def get_public_case(self, patient):
        return os.path.join(self.public_path, "case", "{patient}.npz".format(patient=patient))

    def get_nodule(self, patient):
        return os.path.join(self.processed_path, "nodule", "{patient}.npz".format(patient=patient))

    def get_public_nodule(self, patient):
        return os.path.join(self.public_path, "nodule", "{patient}.npz".format(patient=patient))

    def get_lidc_nodule(self, patient):
        return os.path.join(self.lidc_nodule_path, "{patient}.npz".format(patient=patient))


def parse_environ(cfg_path=None):
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(__file__), "ENVIRON")
    assert os.path.exists(cfg_path), "`ENVIRON` does not exists."
    with open(cfg_path) as f:
        environ = json.load(f)
    return environ
