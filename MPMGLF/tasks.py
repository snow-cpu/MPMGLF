import json
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.backends import cudnn
import matplotlib.pyplot as plt
from layer.Module import *
from data.data_process import *
from sklearn.metrics import r2_score


class AverageMeter(object):
    def __init__(self):
        self.history = []
        self.last = None
        self.val = 0
        self.sum = 0
        self.count = 0

    def mean(self):
        if self.count == 0:
            return 0
        return self.sum / self.count

    def reset(self):
        self.last = self.mean()
        self.history.append(self.last)
        self.sum = 0
        self.count = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n


class DummyLogger(object):
    def __init__(self, args, dirname=None):
        self.args = args
        if dirname is None:
            raise Exception("No location passed in for the configuration file to be saved")
        else:
            self.dirname = dirname
            if not os.path.exists(dirname):
                os.makedirs(dirname)
                os.mkdir(os.path.join(dirname, "metrics"))
                self.log_json(args, os.path.join(self.dirname, f"config{datetime.now().timestamp()}.json"))
            else:
                self.log_json(args, os.path.join(self.dirname, f"config{datetime.now().timestamp()}.json"))
        if args.logging:
            self.f_metric = open(os.path.join(self.dirname, "metrics", f"{datetime.now().timestamp()}-metrics.log"), 'a')

    def log_json(self, data, filename, mode="w"):
        with open(filename, mode) as f:
            f.write(json.dumps(data.__dict__, indent=4, ensure_ascii=False))

    def write_to_file(self, text):
        if self.args.logging:
            self.f_metric.writelines(text + f"{datetime.now()}" + "\n")
            self.f_metric.flush()

    def close(self):
        if self.args.logging:
            self.f_metric.close()


class Timer():
    def __init__(self, name):
        super(Timer, self).__init__()
        self.name = name
        self.running = True
        self.total = 0
        self.start = round(datetime.now().timestamp(), 2)
        self.intervalTime = round(datetime.now().timestamp(), 2)
        print("<><><><> Begin[{}]<><><><>".format(self.name))

    def reset(self):
        self.running = True
        self.total = 0
        self.start = round(datetime.now().timestamp(), 2)
        return self

    def interval(self, intervalName=""):
        intervalTime = self._to_hms(round(datetime.now().timestamp() - self.intervalTime, 2))
        msg = "<> <> Timer [{}] <> <> Interval [{}]: {} <> <>".format(self.name, intervalName, intervalTime)
        print(msg)
        self.intervalTime = round(datetime.now().timestamp(), 2)
        return msg

    def _to_hms(self, sec):
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return "%dh %02dm %02ds" % (h, m, s)

    def finish(self):
        if self.running:
            self.running = False
            self.total += round(datetime.now().timestamp() - self.start, 2)
            elapsed = self._to_hms(self.total)
            print("<> <> <> Finished Timer [{}] <> <> <> Total time elapsed: {} <> <> <>".format(self.name, elapsed))

    def mytime(self):
        if self.running:
            return round(self.total + datetime.now().timestamp() - self.start, 2)
        return self.total


class Long_term_Forecast:
    def __init__(self, args):
        self.args = args
        self._train_loss = AverageMeter()
        self._train_metrics = {'nloss': AverageMeter(), 'MSE': AverageMeter()}
        self._val_loss = AverageMeter()
        self._val_metrics = {'nloss': AverageMeter(), 'MSE': AverageMeter()}
        self.logger = DummyLogger(self.args, dirname=args.out_dir)
        self.dirname = self.logger.dirname

        if not args.no_cuda and torch.cuda.is_available():
            print("[using CUDA]")
            self.device = torch.device("cuda" if args.cuda_id < 0 else 'cuda:%d' % args.cuda_id)
            cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")

        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device:
            torch.cuda.manual_seed(seed)

        self.train_data, self.train_loader = data_provider(args=self.args, flag='train')
        self._n_train_batches = len(self.train_loader)
        print("train, batch:", self._n_train_batches)

        self.val_data, self.val_loader = data_provider(args=self.args, flag='val')
        self._n_val_batches = len(self.val_loader)
        print("val, batch:", self._n_val_batches)

        self.test_data, self.test_loader = data_provider(args=self.args, flag='test')
        self._n_test_batches = len(self.test_loader)
        self._n_test_examples = len(self.test_data)
        print("test, batch:", self._n_test_batches)

        self._n_train_examples = 0
        self.model = Model(self.args)
        self.model.network = self.model.network.to(self.device)
        self.is_test = False
        self.iterpro = nn.Linear(args.outfeature, args.outfeature, bias=True).to(self.device)

    def train(self):
        self.is_test = False
        timer = Timer("Train")
        self._epoch = self._best_epoch = 0
        self.best_metrics = {}
        for i in self._val_metrics:
            self.best_metrics[i] = float("inf")
        self._reset_metrics()

        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        while self._stop_condition(self._epoch, self.args.patience):
            self._epoch += 1
            if self._epoch % self.args.printeveryepochs == 0:
                format_str = "\n >> _epoch[{}/{}]".format(self._epoch, self.args.train_epochs)
                print(format_str)
                self.logger.write_to_file(format_str)

            self._run_epoch(self.train_loader, training=True, verbose=self.args.verbose)

            if self._epoch % self.args.printeveryepochs == 0:
                format_str = "Training Epoch {} -- Loss: {:0.5f}".format(self._epoch, self._train_loss.mean())
                format_str += self.metric_to_str(self._train_metrics)
                train_epoch_time_msg = timer.interval("Training Epoch {}".format(self._epoch))
                self.logger.write_to_file(train_epoch_time_msg + '\n' + format_str)
                print(format_str)

                end_time = time.time()
                epoch_time = end_time - start_time
                mem = torch.cuda.memory_allocated()
                print(f"Memory used: {mem / 1024 ** 3:.2f} GB")

                format_str = "\n>>> Validation Epoch: [{} / {}]".format(self._epoch, self.args.train_epochs)
                print(format_str)
                self.logger.write_to_file(format_str)
                with torch.no_grad():
                    val_output, val_ground = self._run_epoch(self.val_loader, training=False,
                                                             verbose=self.args.verbose,
                                                             out_predictions=self.args.out_predictions)
                    val_output = torch.tensor([np.array(item.cpu().detach()) for item in val_output])
                    val_ground = torch.tensor([np.array(item.cpu().detach()) for item in val_ground])
                    r2 = r2_score(val_output.detach().numpy().reshape(val_output.shape[0], -1),
                                  val_ground.detach().numpy().reshape(val_ground.shape[0], -1))
                    l1 = nn.L1Loss()
                    MAE = l1(val_output, val_ground)
                    MSE = self.model.criterion(val_output, val_ground)

                if self._epoch % self.args.printeveryepochs == 0:
                    format_str = "Validation Epoch {} -- Loss: {:0.5f}".format(self._epoch, self._val_loss.mean())
                    format_str += self.metric_to_str(self._val_metrics)
                    if MSE is not None:
                        format_str += '\n   MAE: {:0.5f}  MSE: {:0.5f}  R2: {:0.5f}'.format(MAE, MSE, r2)
                    dev_epoch_time_msg = timer.interval("Validation Epoch {}".format(self._epoch))
                    self.logger.write_to_file(dev_epoch_time_msg + '\n' + format_str)
                    print(format_str)

                if self.args.eary_stop_metric == self.model.metric_name and MSE is not None:
                    cur_val_score = MSE
                else:
                    cur_val_score = self._val_metrics[self.args.eary_stop_metric].mean()

                if self.best_metrics[self.args.eary_stop_metric] > cur_val_score:
                    self._best_epoch = self._epoch
                    for k in self._val_metrics:
                        self.best_metrics[k] = self._val_metrics[k].mean()
                    if MSE is not None:
                        self.best_metrics[self.model.metric_name] = MSE
                    self.model.save(self.dirname, self._epoch)
                    if self._epoch % self.args.printeveryepochs == 0:
                        format_str = 'Saved model to {0} {1}'.format(self.dirname, datetime.now())
                        self.logger.write_to_file(format_str)
                        print(format_str)
                    format_str = "!!! Updated: " + self.best_metric_to_str(self.best_metrics)
                    self.logger.write_to_file(format_str)
                    print(format_str)

                self._reset_metrics()
                self.adjust_learning_rate(self.model.optimizer, self._epoch + 1, self.args)

        timer.finish()
        format_str = "Finished Training: {}\nTraining time: {}".format(self.dirname, timer.total) + '\n' + self.summary()
        print(format_str)
        self.logger.write_to_file(format_str)
        return self.best_metrics

    def test(self, setting, test=0):
        if self.test_loader is None:
            print("No testing set specified -- skipped testing.")
            return
        torch.cuda.reset_peak_memory_stats()
        print('Restoring best model')
        self.model.init_saved_network(r"E:\-2-params1758011227.00628.saved")
        self.model.network = self.model.network.to(self.device)
        self.is_test = True
        self._reset_metrics()
        timer = Timer("Test")
        for param in self.model.network.parameters():
            param.requires_grad = False

        output, gold = self._run_epoch(self.test_loader)
        output = np.array(output)
        preds = np.array(output)[:, -1, :]
        gold = np.array(gold)
        trues = np.array(gold)[:, -1, :]

        pdsdata = pd.DataFrame(preds)
        gtdata = pd.DataFrame(trues)
        resultdata = pd.concat([gtdata, pdsdata], axis=1)
        resultdata.to_csv(rf"{self.args.out_dir}/result2.csv", float_format='%.5f')

        metrics = self._val_metrics
        format_str = "[test] | test_exs = {} | step: [{} / {}]".format(self._n_test_examples, 1, 1)
        format_str += self.metric_to_str(metrics)

        val_output = torch.tensor([np.array(item) for item in output])
        val_ground = torch.tensor([np.array(item) for item in gold])
        l1 = nn.L1Loss()
        MAE = l1(val_output, val_ground)
        test_score = MSE = self.model.criterion(val_output, val_ground)
        r2 = r2_score(val_output.detach().numpy().reshape(val_output.shape[0], -1),
                      val_ground.detach().numpy().reshape(val_ground.shape[0], -1))
        format_str += '\nFinal  MAE:{:0.5f} MSE:{:0.5f} R2:{:0.5f}\n'.format(MAE, MSE, r2)

        print(format_str)
        self.logger.write_to_file(format_str)
        timer.finish()

        format_str = "Finished Testing: {}\nTesting time: {}".format(self.dirname, timer.total)
        print(format_str)
        self.logger.write_to_file(format_str)
        self.logger.close()

        test_metrics = {}
        for k in metrics:
            test_metrics[k] = metrics[k].mean()
        if test_score is not None:
            test_metrics[self.model.metric_name] = test_score
        return test_metrics

    def _run_epoch(self, data_loader, training=False, verbose=10, out_predictions=True):
        start_time = datetime.now().timestamp()
        mode = "train" if training else ("test" if self.is_test else "val")
        if training:
            self.model.optimizer.zero_grad()
        output = []
        ground = []

        for step, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            res = self.run_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, step, training, out_predictions)
            loss = res["loss"]
            metrics = res["metrics"]
            self._update_metrics(loss, metrics, self.args.batch_size, training=training)

            if training:
                self._n_train_examples += self.args.batch_size

            if (verbose > 0) and (step > 0) and ((step + 1) % verbose == 0):
                summary_str = self.self_report(step, mode)
                self.logger.write_to_file(summary_str)
                print(summary_str)
                print('The batch is used by the deadline: {:0.2f}s'.format(datetime.now().timestamp() - start_time))
            else:
                print("verbose{},batch:{}".format(verbose, step))

            if not training and out_predictions:
                output.extend(res['predictions'])
                ground.extend(res['targets'])

        return output, ground

    def run_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, step, training, out_predictions=False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        mode = "train" if training else ("test" if self.is_test else "val")
        network = self.model.network
        network.train(training)
        res, loss, output = self.iterdoing(batch_x, batch_y, batch_x_mark, batch_y_mark, network, mode, out_predictions)

        if out_predictions:
            res['predictions'] = output.detach().cpu()
        return res

    def iterdoing(self, batch_x, batch_y, batch_x_mark, batch_y_mark, network, mode, out_predictions):
        means = batch_x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        batch_x = batch_x - means
        batch_x /= stdev

        batch_x = network.predict_linear(batch_x.permute(0, 2, 1)).permute(0, 2, 1)
        onerawadj = network.prepare_init_fraph(batch_x)
        onegraphadj, oneadj = network.get_adj(batch_x, onerawadj)
        onepredout, oneemb = network(batch_x, oneadj)

        onepredout = onepredout * stdev
        onepredout = onepredout + means

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = onepredout[:, -self.args.pred_len:, :]
        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
        outputs = outputs.detach().cpu().numpy()
        batch_y = batch_y.detach().cpu().numpy()
        if self.test_data.scale:
            shape = outputs.shape
            outputs = self.test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
            batch_y = self.test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

        onepredout = torch.Tensor(outputs[:, :, f_dim:]).to(self.device)
        batch_y = torch.Tensor(batch_y[:, :, f_dim:]).to(self.device)

        loss1 = self.model.criterion(onepredout, batch_y)
        score1 = self.model.criterion(onepredout, batch_y)
        loss1 += self.batch_graph_loss(onegraphadj, batch_x.permute(0, 2, 1))

        first_raw_adj, first_adj = onegraphadj, oneadj
        max_iter = self.args.graphiter
        loss = 0
        iter_ = 0
        batch_last_iters = to_cuda(torch.zeros(batch_x.shape[0], dtype=torch.uint8), self.device)
        batch_stop_indicators = to_cuda(torch.ones(batch_x.shape[0], dtype=torch.uint8), self.device)
        batch_all_outputs = []

        while (iter_ == 0 or torch.sum(batch_stop_indicators).item() > 0) and iter_ < max_iter:
            iter_ += 1
            batch_last_iters += batch_stop_indicators
            pre_graphadj = onegraphadj
            pre_adj = oneadj

            oneemb1 = self.iterpro(oneemb)
            curgraphadj, curadj = network.get_adj(oneemb1, onerawadj)
            curadj = self.args.update_adj_ratio * curadj + (1 - self.args.update_adj_ratio) * pre_graphadj
            
            tmp_output, oneemb1 = network(oneemb, curadj)

            tmp_output = tmp_output * stdev
            tmp_output = tmp_output + means

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = tmp_output[:, -self.args.pred_len:, :]
            outputs = outputs.detach().cpu().numpy()
            if self.test_data.scale:
                shape = outputs.shape
                outputs = self.test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)

            tmp_output = torch.Tensor(outputs[:, :, f_dim:]).to(self.device)
            batch_all_outputs.append(tmp_output)

            tmp_loss = self.model.criterion(tmp_output, batch_y, reduction='none')
            if len(tmp_loss.shape) == 3:
                tmp_loss = tmp_loss.mean(dim=-1).mean(dim=-1)
            loss += batch_stop_indicators.float() * tmp_loss
            loss += batch_stop_indicators.float() * self.batch_graph_loss(curgraphadj, batch_x.permute(0, 2, 1),
                                                                          keep_batch_dim=True)
            tmp_stop_criteria = self.batch_diff(curgraphadj, pre_adj, first_raw_adj) > self.args.eps_adj
            batch_stop_indicators = batch_stop_indicators * tmp_stop_criteria

        if iter_ > 0:
            loss = torch.mean(loss / batch_last_iters.float()) + loss1
            onepredout = torch.zeros_like(batch_y)
            for i in range(len(batch_y)):
                onepredout[i] = batch_all_outputs[batch_last_iters[i] - 1][i]
            score = self.model.score_func(onepredout, batch_y)
        else:
            loss = loss1
            score = score1

        if mode == 'train':
            loss.backward()
            self.model.optimizer.step()
            self.model.optimizer.zero_grad()

        res = {'loss': loss.item(),
               'metrics': {'nloss': -loss.item(), self.model.metric_name: score}}
        if out_predictions:
            res['targets'] = batch_y.detach().cpu()
        return res, loss, onepredout

    def batch_diff(self, X, Y, Z):
        assert X.shape == Y.shape
        diff = torch.sum(torch.pow(X - Y, 2), (1, 2))
        norm = torch.sum(torch.pow(Z, 2), (1, 2))
        diff = diff / torch.clamp(norm, min=1e-5)
        return diff

    def _reset_metrics(self):
        self._train_loss.reset()
        self._val_loss.reset()
        for k in self._train_metrics:
            self._train_metrics[k].reset()
        for k in self._val_metrics:
            self._val_metrics[k].reset()

    def _stop_condition(self, epoch, patience=10):
        no_improvement = epoch >= self._best_epoch + patience
        exceeded_max_epochs = epoch >= self.args.train_epochs
        return False if no_improvement or exceeded_max_epochs else True

    def _update_metrics(self, loss, metrics, batch_size, training=True):
        if training:
            if loss:
                self._train_loss.update(loss)
            for k in self._train_metrics:
                self._train_metrics[k].update(metrics[k], batch_size)
        else:
            if loss:
                self._val_loss.update(loss)
            for k in self._val_metrics:
                self._val_metrics[k].update(metrics[k], batch_size)

    def self_report(self, step, mode='train'):
        if mode == "train":
            format_str = "[train-{}] step: [{} / {}] | loss = {:0.5f}".format(
                self._epoch, step, self._n_train_batches, self._train_loss.mean())
            format_str += self.metric_to_str(self._train_metrics)
        elif mode == "val":
            format_str = "[val-{}] step: [{} / {}] | loss = {:0.5f}".format(
                self._epoch, step, self._n_val_batches, self._val_loss.mean())
            format_str += self.metric_to_str(self._val_metrics)
        elif mode == "test":
            format_str = "[test] | test_exs = {} | step: [{} / {}]".format(
                self._n_test_examples, step, self._n_test_batches)
            format_str += self.metric_to_str(self._val_metrics)
        else:
            raise ValueError('mode = {} not supported.' % mode)
        return format_str

    def metric_to_str(self, metrics):
        format_str = ''
        for k in metrics:
            format_str += ' | {} = {:0.5f}'.format(k.upper(), metrics[k].mean())
        return format_str

    def best_metric_to_str(self, metrics):
        format_str = '\n'
        for k in metrics:
            format_str += '{} = {:0.5f}\n'.format(k.upper(), metrics[k])
        return format_str

    def summary(self):
        start = "\n<<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        info = "Best epoch = {}; ".format(self._best_epoch) + self.best_metric_to_str(self.best_metrics)
        end = " <<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> "
        return "\n".join([start, info, end])

    def batch_graph_loss(self, adj, feature, keep_batch_dim=False):
        if keep_batch_dim:
            graphloss = []
            for i in range(adj.shape[0]):
                L = torch.diagflat(torch.sum(adj[i], -1)) - adj[i]
                graphloss.append(self.args.smoothness_ratio * torch.trace(
                    torch.mm(feature[i].transpose(-1, -2), torch.mm(L, feature[i]))) / int(np.prod(adj.shape[1:])))
            graphloss = to_cuda(torch.Tensor(graphloss), self.device)
            ones_vec = to_cuda(torch.ones(adj.shape[:-1]), self.device)
            graphloss += -self.args.degree_ratio * torch.matmul(ones_vec.unsqueeze(1),
                                                                torch.log(torch.matmul(adj, ones_vec.unsqueeze(-1)) + 1e-5)).squeeze(-1).squeeze(-1) / adj.shape[-1]
            graphloss += self.args.sparsity_ratio * torch.sum(torch.pow(adj, 2), (1, 2)) / int(np.prod(adj.shape[1:]))
        else:
            graphloss = 0
            for i in range(adj.shape[0]):
                L = torch.diagflat(torch.sum(adj[i], -1)) - adj[i]
                graphloss += self.args.smoothness_ratio * torch.trace(
                    torch.mm(feature[i].transpose(-1, -2), torch.mm(L, feature[i]))) / int(np.prod(adj.shape))
            ones_vec = to_cuda(torch.ones(adj.shape[:-1]), self.device)
            graphloss += -self.args.degree_ratio * torch.matmul(ones_vec.unsqueeze(1),
                                                                torch.log(torch.matmul(adj, ones_vec.unsqueeze(-1)) + 1e-5)).sum() / adj.shape[0] / adj.shape[-1]
            graphloss += self.args.sparsity_ratio * torch.sum(torch.pow(adj, 2)) / int(np.prod(adj.shape))
        return graphloss

    def adjust_learning_rate(self, optimizer, epoch, args):
        if args.lradj == 'type1':
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif args.lradj == 'type2':
            lr_adjust = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            printmas = 'Updating learning rate to {}'.format(lr)
            self.logger.write_to_file(printmas + "\n")


def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x


def visual(true, preds=None, name='./pic/test.png'):
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')