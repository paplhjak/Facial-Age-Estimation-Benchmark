"""
Implements classes which summarize prediction errors in tables and visualize different metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics
from lib.utils import compute_error
from lib.utils import get_loss_matrix


######################################################################################
# Error summary: evaluation metrics for all prediction heads computed on
# trn/val/tst examples
######################################################################################

class ErrorSummary:
    def __init__(self, config, benchmark, face_list, prediction):
        """Summary tables of prediction errors over databases and attributes.

        Args:
            config (dict): config YAML as dictionary
            benchmark (dict): benchmark YAML as dictionary
            face_list (dict): face list CSV as dict
            prediction (list): predictions for each split
        Returns:
            error (dict): error[part] is a summary table for 
            'trn', 'val' or 'tst' part of examples.
        """
        n_splits = len(benchmark[0]['split'])
        databases = [x['tag'] for x in benchmark]

        metrics = []
        for x in config['heads']:
            for m in x['metric']:
                metrics.append(x['tag'] + '[' + m + ']')

        self.table = {'trn': None, 'val': None, 'tst': None}
        for folder, part in enumerate(['trn', 'val', 'tst']):
            self.table[part] = pd.DataFrame(
                columns=metrics, index=databases+["ALL"])

            for db_id, subset in enumerate(databases+['ALL']):
                col = -1
                for head in config['heads']:
                    tag = head['tag']
                    for metric in head['metric']:
                        col += 1
                        err = []
                        for split in range(0, n_splits):
                            if subset == 'ALL':
                                idx = np.argwhere(
                                    (prediction[split]['folder'] == folder))
                            else:
                                idx = np.argwhere((face_list['db_id'] == db_id) & (
                                    prediction[split]['folder'] == folder))

                            if len(idx) > 0:
                                err.append(compute_error(metric,
                                                         prediction[split]['true_label'][tag][idx],
                                                         prediction[split]['predicted_label'][tag][idx]))

                        err = np.array(err)
                        self.table[part].iloc[db_id,
                                              col] = f"{np.mean(err):.2f}({np.std(err):.2f})"

        if len(databases) == 1:
            for set in self.table.keys():
                self.table[set] = self.table[set].drop(['ALL'])


######################################################################################
# Compute and visualize confusion matrix for all prediction heads
######################################################################################

class VisualMetric:
    def __init__(self, config, benchmark, face_list, prediction):
        """Compute confuction matrices.
        Args:
            config (dict): config YAML as dict
            benchmark (dict): benchmark YAML as dict
            face_list (dict): face list  CSV as dict
            predciton (list): predictions for each split

            error(dict): error[database][task][part] confusion matrix for database, 
              task and part (trn,val,tst)
            roc (dic): roc[database][task][part] roc curve is a dict with 'fpr'' and 
              tpr' for database, task and part {trn,val,tst}
        """
        n_splits = len(benchmark[0]['split'])
        databases = [x['tag'] for x in benchmark]
        if len(databases) > 1:
            databases.append('ALL')

        self.n_labels = {}
        for head in config['heads']:
            self.n_labels[head['tag']] = len(head['labels'])

        self.tasks = [x['tag'] for x in config['heads']]
        self.n_splits = n_splits
        self.prediction = prediction
        self.face_list = face_list
        self.databases = databases

        self.error = {}
        for database in databases:
            self.error[database] = {
                x: {'trn': None, 'val': None, 'tst': None} for x in self.tasks}

        for db_id, database in enumerate(databases):
            for head in config['heads']:
                tag = head['tag']
                for folder, part in enumerate(['trn', 'val', 'tst']):
                    err = []
                    for split in range(0, n_splits):
                        if database == 'ALL':
                            idx = np.argwhere(
                                prediction[split]['folder'] == folder)
                        else:
                            idx = np.argwhere((face_list['db_id'] == db_id) & (
                                prediction[split]['folder'] == folder))
                        if len(idx) > 0:
                            err.append(self.compute_confusion_matrix(head,
                                                                     prediction[split]['true_label'][tag][idx],
                                                                     prediction[split]['predicted_label'][tag][idx]))

                    self.error[database][tag][part] = err

    def get_roc_curve(self, database, task, n_points=100, target_class=[0]):

        roc = {'trn': None, 'val': None, 'tst': None}
        db_id = self.databases.index(database)

        for folder, part in enumerate(['trn', 'val', 'tst']):
            fpr = np.linspace(0, 1, n_points)
            tpr = np.empty((0, n_points))
            target_class_prior = 0

            for split in range(0, self.n_splits):
                if database == 'ALL':
                    idx = np.argwhere(
                        self.prediction[split]['folder'] == folder)
                else:
                    idx = np.argwhere((self.face_list['db_id'] == db_id) & (
                        self.prediction[split]['folder'] == folder))

                if len(idx) > 0:
                    ppos = np.sum(
                        self.prediction[split]['posterior'][task][idx, target_class], axis=1)
                    ppos[ppos >= 1] = 0.9999999
                    ppos[ppos <= 0] = 0.0000001
                    score = np.log(ppos) - np.log(1.0-ppos)

                    true_label = np.array(
                        [1 if self.prediction[split]['true_label']['age'][i] in target_class else -1 for i in idx])

                    target_class_prior += np.sum(true_label ==
                                                 1)/len(true_label)

                    cur_fpr, cur_tpr, _ = metrics.roc_curve(true_label, score)
                    tpr = np.vstack((tpr, np.interp(fpr, cur_fpr, cur_tpr)))

            roc[part] = {'tpr': tpr, 'fpr': fpr,
                         'target_class_prior': target_class_prior/self.n_splits}
        return roc

    def compute_confusion_matrix(self, head, true_label, predicted_label):
        """Compute confusion metrix.
        Args:
            head (dict): is head description from config['heads'][i]   
            true_label (1d np array): true labels from 0,...,n_labels-1
            predicted_label: (1d np array) predicted label from 0,...,n_labels-1
        Returns:
            cm (np array n_labels x n_labels): cm[predicted,true]
        """
        n_labels = len(head['labels'])
        C = np.zeros((n_labels, n_labels))
        for yt, yp in zip(true_label, predicted_label):
            C[yp, yt] += 1
        return C

    def plot(self, database, attribut, metric, tag=None, subsets=None):
        """Visualize confusion matrix in a figure.

        Args:
            cm (dict): confusion matrices for trn/val/tst subsets
            matric (str): metric identifier where
            'cm' shows standard confusion matrix
            'mae' shows MAE per true category
            'mae(from,to)' shows MAE for labels in interval (from,to)
            'mean' shows MEAN prediction per category
            'mean(from,to)' shows MEAN for labels in interval (from,to)
            tag (str): attribute tag
        Returns:
            fig (matplotlib.Figure): handle of created figure
            df (DataFame): data frame sumarizing the errors
        """

        # this is for backward compatibility when the metrics was
        # identified solely by a string without any paramters
        if type(metric) is not dict:
            metric = {'tag': metric}

        C = self.error[database][attribut]
        n_labels = self.n_labels[attribut]

        # default parameter setting
        label_bounds = [0, n_labels]
        if 'min_label' in metric.keys():
            label_bounds[0] = min(max(metric['min_label'], 0), n_labels)
        if 'max_label' in metric.keys():
            label_bounds[1] = min(max(metric['max_label'], 0), n_labels)

        if 'target_class_prior' in metric.keys():
            target_class_prior = metric['target_class_prior']
        else:
            target_class_prior = None

        if 'target_class' in metric.keys():
            target_class = metric['target_class']
        else:
            target_class = [0]

        if 'target_class_name' in metric.keys():
            target_class_name = metric['target_class_name']
        else:
            target_class_name = None

        if 'fpr_points' in metric.keys():
            fpr_points = metric['fpr_points']
        else:
            fpr_points = 100

        if subsets is None:
            subsets = C.keys()

        if tag is None:
            tag = attribut

        if metric['tag'] == 'roc':

            # roc = self.roc[database][attribut]
            roc = self.get_roc_curve(
                database, attribut, n_points=fpr_points, target_class=target_class)

            fig = plt.figure()
            fig.set_size_inches(7, 7)

            for part in subsets:
                plt.plot(roc[part]['fpr'], np.mean(
                    roc[part]['tpr'], axis=0), label=f"{part}")

            plt.grid('on')
            plt.legend()
            plt.xlabel('false positive rate')
            plt.ylabel('true positive rate')
            plt.title(f"ROC: {tag}")

        elif metric['tag'] == 'prc':

            roc = self.get_roc_curve(
                database, attribut, n_points=fpr_points, target_class=target_class)

            fig = plt.figure()
            fig.set_size_inches(7, 7)

            if target_class_prior is None:
                for part in subsets:
                    target_class_prior = roc[part]['target_class_prior']
                    tpr = np.mean(roc[part]['tpr'], axis=0)
                    fpr = roc[part]['fpr']
                    r = (1-target_class_prior)/target_class_prior
                    prec = tpr/(tpr + r*fpr+1e-10)
                    plt.plot(
                        tpr[1:], prec[1:], label=f"{part} P(target)={target_class_prior:.3f}")
            else:
                tpr = np.mean(roc['tst']['tpr'], axis=0)
                fpr = roc['tst']['fpr']
                for t in target_class_prior:
                    r = (1-t)/t
                    prec = tpr/(tpr + r*fpr+1e-10)
                    plt.plot(tpr[1:], prec[1:], label=f"P(target)={t:.3f}")

            plt.grid('on')
            plt.legend()
            if target_class_name is None:
                plt.title(f"PrecRecall curve: {tag}")
            else:
                plt.title(
                    f"PrecRecall curve: {tag}, target={target_class_name}")
            plt.xlabel('Recall')
            plt.ylabel('Precision')

        elif metric['tag'] == 'cm':

            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
            for ax, part in enumerate(subsets):
                err = C[part]
                if len(err) > 0:
                    A = np.zeros(err[0].shape)
                    n_labels = A.shape[0]
                    for i in range(len(err)):
                        nconst = np.sum(err[i], axis=0)
                        A += err[i]/nconst
                    A = A / len(err)

                    df = pd.DataFrame(A, range(n_labels), range(n_labels))
                    sn.set(font_scale=1.0)  # for label size
                    sn.heatmap(df, annot=True, annot_kws={
                               "size": 12}, ax=axes[ax], cbar=False)  # font size
                    axes[ax].set_xlabel('true ' + tag)
                    axes[ax].set_ylabel('pred ' + tag)
                    axes[ax].set_title(part)
            plt.subplots_adjust(bottom=0.2, hspace=0.2)

        elif metric['tag'] == 'mae':

            new_tag = tag + "[mae]"
            df = pd.DataFrame(
                columns=['set', 'true label', 'split'] + [new_tag])
            palette = []
            for part in subsets:
                err = C[part]
                if len(err) > 0:
                    if part == 'trn':
                        palette.append('b')
                    elif part == 'val':
                        palette.append('g')
                    elif part == 'tst':
                        palette.append('r')
                    n_labels = err[0].shape[0]

                    for split in range(len(err)):
                        for label in range(label_bounds[0], label_bounds[1]):
                            nconst = np.sum(err[split][:, label])
                            if nconst > 0:
                                p = err[split][:, label]/nconst
                                mae = np.dot(
                                    p, np.abs(np.arange(n_labels)-label))
                                df = df.append(
                                    {'set': part, 'true label': label, 'split': split, new_tag: mae}, ignore_index=True)
            fig = plt.figure()
            sn.set(font_scale=1.0)  # for label size
            sn.lineplot(data=df, x='true label', y=new_tag,
                        hue='set', ci=90, palette=palette)

        elif metric['tag'] == 'bias':
            new_tag = tag + "[bias]"
            df = pd.DataFrame(
                columns=['set', 'true label', 'split'] + [new_tag])
            palette = []
            for part in subsets:
                err = C[part]
                if len(err) > 0:
                    if part == 'trn':
                        palette.append('b')
                    elif part == 'val':
                        palette.append('g')
                    elif part == 'tst':
                        palette.append('r')
                    n_labels = err[0].shape[0]

                    for split in range(len(err)):
                        for label in range(label_bounds[0], label_bounds[1]):
                            nconst = np.sum(err[split][:, label])
                            if nconst > 0:
                                p = err[split][:, label]/nconst
                                mae = np.dot(p, np.arange(n_labels)-label)
                                df = df.append(
                                    {'set': part, 'true label': label, 'split': split, new_tag: mae}, ignore_index=True)
            fig = plt.figure()
            sn.set(font_scale=1.0)  # for label size
            sn.lineplot(data=df, x='true label', y=new_tag,
                        hue='set', ci=90, palette=palette)

        elif metric['tag'] == 'mean':
            new_tag = tag + "[mean]"
            df = pd.DataFrame(
                columns=['set', 'true label', 'split'] + [new_tag])
            palette = []
            for part in subsets:
                err = C[part]
                if len(err) > 0:
                    if part == 'trn':
                        palette.append('b')
                    elif part == 'val':
                        palette.append('g')
                    elif part == 'tst':
                        palette.append('r')
                    n_labels = err[0].shape[0]

                    for split in range(len(err)):
                        for label in range(label_bounds[0], label_bounds[1]):
                            nconst = np.sum(err[split][:, label])
                            if nconst > 0:
                                p = err[split][:, label]/nconst
                                avg = np.dot(p, np.arange(n_labels))
                                df = df.append(
                                    {'set': part, 'true label': label, 'split': split, new_tag: avg}, ignore_index=True)

            fig = plt.figure()
            sn.set(font_scale=1.0)  # for label size
            sn.lineplot(data=df, x='true label', y=new_tag,
                        hue='set', ci=90, palette=palette)

        else:
            raise ("Unknown visual metric tag.")

        return fig

######################################################################################
# RiskCoverage curve: compute and visualize risk coverage curve
######################################################################################


class Uncertainty:
    def __init__(self, config, benchmark, face_list, prediction):
        """Compute risk coverage curve.

        Args:
            config (dict): config YAML as dict
            benchmark (dict): benchmark YAML as dict
            face_list (dict): face list  CSV as dict
            predciton (list): predictions for each split
        Returns:
            rcc (dict): cm[db][attr][part] is a list of NP arrays, each [n_labels x n_labels],
            where
            db (str): is the database tag and 'ALL' stands for compund database
            attr (str): is the attribute tag, e.g. 'age'
            part (str): is stands for 'trn', 'val' or 'tst' part of examples
            n_labels (int): is the number of labels for attribute attr

        """
        n_splits = len(benchmark[0]['split'])
        databases = [x['tag'] for x in benchmark]

        self.curve = {x: None for x in ['ALL']+databases}
        for subset in self.curve.keys():
            self.curve[subset] = {x['tag']: {
                'trn': None, 'val': None, 'tst': None} for x in config['heads']}

        self.metric = {}
        for head in config['heads']:
            tag = head['tag']
            self.metric[tag] = head['metric'][0]

        for db_id, subset in enumerate(databases+['ALL']):
            for head in config['heads']:
                tag = head['tag']

                for folder, part in enumerate(['trn', 'val', 'tst']):
                    xs = []
                    ys = []
                    aurc = []
                    uncertainty = []
                    pred_loss = []
                    for split in range(0, n_splits):
                        if subset == 'ALL':
                            idx = np.argwhere(
                                prediction[split]['folder'] == folder)
                        else:
                            idx = np.argwhere((face_list['db_id'] == db_id) & (
                                prediction[split]['folder'] == folder))

                        if len(idx) > 0:
                            curve_, uncertainty_, pred_loss_ = self.compute_rc_curve(head,
                                                                                     prediction[split]['true_label'][tag][idx],
                                                                                     prediction[split]['predicted_label'][tag][idx],
                                                                                     prediction[split]['uncertainty'][tag][idx])

                            xs.append(np.linspace(0, 1, len(curve_)))
                            ys.append(curve_)
                            aurc.append(np.mean(curve_))

                            uncertainty.append(uncertainty_)
                            pred_loss.append(pred_loss_)

                    if len(xs) > 0:
                        # average risk-coverage curve over splits
                        rc_mean_x_axis = np.linspace(0, 1, num=1000)
                        ys_interp = [
                            np.interp(rc_mean_x_axis, xs[i], ys[i]) for i in range(len(xs))]
                        rc_mean_y_axis = np.mean(ys_interp, axis=0)

                        # average risk-uncertainty curve over splits
                        ru_x, ru_y, uhist_x, uhist_y = self.compute_risk_uncertainty_curve(
                            uncertainty, pred_loss)

                        self.curve[subset][tag][part] = {'rc_x': rc_mean_x_axis, 'rc_y': rc_mean_y_axis,
                                                         'AuRC': np.mean(aurc),
                                                         'ru_x': ru_x, 'ru_y': ru_y, 'uhist_x': uhist_x, 'uhist_y': uhist_y}

    def compute_risk_uncertainty_curve(self, uncertainty, pred_loss):

        M = len(uncertainty)
        N = 50
        min_uncertainty = np.Inf
        max_uncertainty = -np.Inf
        for i in range(M):
            max_uncertainty = np.maximum(max_uncertainty, uncertainty[i].max())
            min_uncertainty = np.minimum(min_uncertainty, uncertainty[i].min())

        mean_x = np.linspace(min_uncertainty, max_uncertainty, num=N+1)
        mean_y = np.zeros((M, N))
        for i in range(M):
            uncertainty_ = uncertainty[i].squeeze()
            for j in range(N):
                if j < N-1:
                    idx = np.argwhere((uncertainty_ >= mean_x[j]) & (
                        uncertainty_ < mean_x[j+1]))
                else:
                    idx = np.argwhere((uncertainty_ >= mean_x[j]) & (
                        uncertainty_ <= mean_x[j+1]))

                if len(idx) > 0:
                    mean_y[i, j] = np.mean(pred_loss[i][idx[0]])

        unc_hst = np.zeros(N)
        for i in range(M):
            hst, _ = np.histogram(uncertainty[i], bins=mean_x, density=True)
            unc_hst += hst
        return 0.5*(mean_x[0:-1]+mean_x[1:]), np.mean(mean_y, 0), 0.5*(mean_x[0:-1]+mean_x[1:]), unc_hst/M

    def compute_rc_curve(self, head, true_label, pred_label, uncertainty):
        n_examples = len(true_label)
        loss_matrix = get_loss_matrix(len(head['labels']), head['metric'][0])

        prediction_loss = np.empty(n_examples)
        for i in range(n_examples):
            prediction_loss[i] = loss_matrix[true_label[i], pred_label[i]]

        # sort in descending order
        idx = np.argsort(uncertainty.squeeze())
        curve = np.empty(n_examples)
        cval = 0.0
        for i in range(n_examples):
            cval += prediction_loss[idx[i]]
            curve[i] = cval / (i+1)

        return curve, uncertainty[idx], prediction_loss[idx]

    def plot_rc_curve(self, database, attr, title=None, yaxis_label=None):
        """Visualize Risk-Coverage curve"""

        curve = self.curve[database][attr]
        if title is None:
            title = f"RiskCoverage: {attr}"
        if yaxis_label is None:
            yaxis_label = f"Selective risk [{self.metric[attr]}]"

        subsets = curve.keys()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

        for part in subsets:
            if part == 'trn':
                col = 'b'
            elif part == 'val':
                col = 'g'
            elif part == 'tst':
                col = 'r'

            if curve[part] is not None:
                ax.plot(curve[part]['rc_x'], curve[part]['rc_y'], col,
                        label=f"{part} AuRC={curve[part]['AuRC']:.2f}")
                ax.set_title(title)
                ax.set_ylabel(yaxis_label)
                ax.set_xlabel('coverage')
                ax.legend()
                ax.grid('on')

        return fig, ax

    def plot_ru_curve(self, database, attr, title=None, yaxis_label=None):
        """Visualize Risk-Coverage curve"""

        curve = self.curve[database][attr]
        if title is None:
            title = f"RiskUncertainty: {attr}"
        if yaxis_label is None:
            yaxis_label = f"risk [{self.metric[attr]}]"

        subsets = curve.keys()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

        for part in subsets:
            if part == 'trn':
                col = 'b'
            elif part == 'val':
                col = 'g'
            elif part == 'tst':
                col = 'r'

            if curve[part] is not None:
                ax.plot(curve[part]['ru_x'], curve[part]
                        ['ru_y'], col, label=f"{part}")
                ax.set_title(title)
                ax.set_ylabel(yaxis_label)
                ax.set_xlabel('uncertainty')
                ax.legend()
                ax.grid('on')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        a = min(xlim[0], ylim[0])
        b = max(xlim[1], ylim[1])
        ax.plot([a, b], [a, b], 'k--')

        return fig, ax

    def plot_uncertainty_hist(self, database, attr, title=None):
        """Visualize uncertainty histogram"""

        curve = self.curve[database][attr]
        if title is None:
            title = f"Uncertainty: {attr}"

        subsets = curve.keys()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

        for part in subsets:
            if part == 'trn':
                col = 'b'
            elif part == 'val':
                col = 'g'
            elif part == 'tst':
                col = 'r'

            if curve[part] is not None:
                ax.plot(curve[part]['uhist_x'], curve[part]
                        ['uhist_y'], col, label=f"{part}")
                ax.set_title(title)
                ax.set_ylabel('pdf')
                ax.set_xlabel('uncertainty')
                ax.legend()
                ax.grid('on')

        return fig, ax
