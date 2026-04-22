# phase 1 : calculate mahalanobis distance and sort it in descending order
# phase 2 : select similar parent samples to cross-over and generate new samples
import numpy as np


class MAHAKIL(object):
    def __init__(self, pfp=0.5):
        self.data_t = None                  # samples with defects
        self.pfp = pfp                      # sampling rate for new samples
        self.new = []                       # generated new samples

    def fit_resample(self, data, label):
        data_t, data_f, label_t, label_f = [], [], [], []
        unique_labels, counts = np.unique(label, return_counts=True)
        majority_class = unique_labels[np.argmax(counts)]
        minority_class = unique_labels[np.argmin(counts)]

        for i in range(label.shape[0]):
            if label[i] == minority_class:
                data_t.append(data[i])
                label_t.append(label[i])
            if label[i] == majority_class:
                data_f.append(data[i])
                label_f.append(label[i])

        t = len(data_f) / (1 - self.pfp) - len(data_f)
        self.data_t = np.array(data_t)
        d = self.mahalanobis_distance(self.data_t)
        d.sort(key=lambda x: x[1], reverse=True)

        k = len(d)
        d_index = [d[i][0] for i in range(k)]
        data_t_sorted = [list(data_t[i]) for i in d_index]
        mid = k // 2
        bin1 = [data_t_sorted[i] for i in range(0, mid)]
        bin2 = [data_t_sorted[i] for i in range(mid, k)]
        parents = zip(bin1, bin2)

        if t > 0:
            count = 0
            while count <= t:
                parents = self.update_parents(parents)
                count = len(parents)

            temp = []
            for i in range(0, len(parents), 2):
                temp.append(parents[i][0])
                temp.append(parents[i][1])
                temp.append(parents[i+1][1])

            self.new = np.array(temp)
            train_new = np.append(data_f, self.new, axis=0)
            label_new = np.append(np.zeros(len(data_f)), np.ones(len(self.new)), axis=0)

        else:
            train_new = np.append(data_f, data_t, axis=0)
            label_new = np.append(np.zeros(len(data_f)), np.ones(len(data_t)), axis=0)

        return train_new, label_new

    def mahalanobis_distance(self, x):
        mu = np.mean(x, axis=0)
        d = []
        for i in range(x.shape[0]):
            x_mu = np.atleast_2d(x[i] - mu)
            s = self.cov(x)
            m = 10 ** -6
            d_square = np.dot(np.dot(x_mu, np.linalg.inv(s + np.eye(s.shape[1]) * m)), np.transpose(x_mu))[0][0]
            d_tuple = (i, d_square)
            d.append(d_tuple)
        return d

    @staticmethod
    def cov(x):
        s = np.zeros((x.shape[1], x.shape[1]))
        mu = np.mean(x, axis=0)
        for i in range(x.shape[0]):
            x_xbr = np.atleast_2d(x - mu)
            s_i = np.dot(np.transpose(x_xbr), x_xbr)
            s = s + s_i
        return np.divide(s, x.shape[0])

    @staticmethod
    def update_parents(parents):
        temp = []
        for i in parents:
            instance = [sum(e)/2.0 for e in zip(*i)]
            temp.append([i[0], instance])
            temp.append([instance, i[1]])
        return temp
