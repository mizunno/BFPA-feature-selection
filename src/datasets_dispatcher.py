import pandas as pd
from scipy.io import loadmat
from sklearn import datasets
from sklearn.model_selection import train_test_split


class DatasetDispatcher:

    def __init__(self, data_path):
        self.data_path = data_path

    @staticmethod
    def get_iris(test_size=0.33, make_split=True, random_seed=None):
        iris = datasets.load_iris()
        data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                            columns=iris['feature_names'] + ['target'])

        if make_split:
            x_train, x_val, y_train, y_val = train_test_split(iris["data"],
                                                              iris["target"],
                                                              test_size=test_size,
                                                              random_state=random_seed,
                                                              shuffle=True)

            return x_train, x_val, y_train, y_val

        return data

    def get_gli85(self, test_size=0.33, make_split=True, random_seed=None):
        return self.__process_mat_dataset("GLI_85", test_size, make_split, random_seed)

    def get_basehock(self, test_size=0.33, make_split=True, random_seed=None):
        return self.__process_mat_dataset("BASEHOCK", test_size, make_split, random_seed)

    def get_pcmac(self, test_size=0.33, make_split=True, random_seed=None):
        return self.__process_mat_dataset("PCMAC", test_size, make_split, random_seed)

    def get_relathe(self, test_size=0.33, make_split=True, random_seed=None):
        return self.get_dataset("RELATHE", test_size, make_split, random_seed)

    def get_smk_can(self, test_size=0.33, make_split=True, random_seed=None):
        return self.get_dataset("SMK_CAN_187", test_size, make_split, random_seed)

    def get_tox(self, test_size=0.33, make_split=True, random_seed=None):
        return self.get_dataset("TOX_171", test_size, make_split, random_seed)

    def get_ar10p(self, test_size=0.33, make_split=True, random_seed=None):
        return self.get_dataset("AR10P", test_size, make_split, random_seed)

    def get_dataset(self, name, test_size=0.33, make_split=True, random_seed=None):
        if name == "GLI_85":
            data = loadmat(self.data_path + "/GLI_85.mat")
        elif name == "BASEHOCK":
            data = loadmat(self.data_path + "/BASEHOCK.mat")
        elif name == "PCMAC":
            data = loadmat(self.data_path + "/PCMAC.mat")
        elif name == "RELATHE":
            data = loadmat(self.data_path + "/RELATHE.mat")
        elif name == "SMK_CAN_187":
            data = loadmat(self.data_path + "/SMK_CAN_187.mat")
        elif name == "TOX_171":
            data = loadmat(self.data_path + "/TOX_171.mat")
        elif name == "AR10P":
            data = loadmat(self.data_path + "/AR10P.mat")
        else:
            raise Exception("Provide a valid dataset name.")

        x = data['X']
        y = data['Y']

        if make_split:
            x_train, x_val, y_train, y_val = train_test_split(x,
                                                              y,
                                                              test_size=test_size,
                                                              random_state=random_seed,
                                                              shuffle=True)
            return x_train, x_val, y_train, y_val

        return x, y

