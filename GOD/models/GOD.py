import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet_ETF_mutli_lora
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from utils.toolkit import count_parameters
from sklearn.metrics import confusion_matrix
num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet_ETF_mutli_lora(args, True)
        self.lora_ids = []
        self._means = None
        self._old_network =None
        self.all_ema3 = []

    def after_task(self):
        self._known_classes = self._total_classes


    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        self._network._cur_task +=1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader_warm = DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=num_workers
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    def _train(self, train_loader, test_loader):
        if self._cur_task == 0:
            self._network.backbone.add_task()
            self._network.update_simplefc(self._total_classes)
            self._network.to(self._device)
            self._network.update_fc(self.args["increment"], self.args["hidden"], self.args["increment"])
            self._network.fc.add_task_layer()
            self._network.to(self._device)
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['init_epoch'],
                                                             eta_min=self.args['min_lr'])
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            self._network.backbone.startEMA()
            result3 = self._compute_accuracy_EMA_Task(self._network, test_loader,Top=3)
            self.all_ema3.append(result3['top1_accuracy'])
        else:
            self._network.backbone.add_task()
            self.lora_ids.append(self._cur_task)
            for param in self._network.parameters():
                param.requires_grad = False
            self._network.backbone.unfreeze_lora([self._cur_task],self.args["free"])
            self._network.update_simplefc(self._total_classes)
            for param in self._network.simple_fc.parameters():
                param.requires_grad = True
            self._network.to(self._device)
            self._network.fc.add_task_layer()
            self._network.to(self._device)
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )  # 1e-5
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['init_epoch'],
                                                             eta_min=self.args['min_lr'])
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            result3 = self._compute_accuracy_EMA_Task(self._network, test_loader,Top=3)
            self.all_ema3.append(result3['top1_accuracy'])

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs,self._cur_task,Train=True)
                classlogits = self._network.simple_fc(outputs["feature"],targets)["logits"]
                logits = outputs["logits"]
                loss_etf = F.cross_entropy(logits, targets)
                loss_ac = F.cross_entropy(classlogits, targets)
                loss = loss_etf + loss_ac
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args["init_epoch"],
                losses / len(train_loader),
                train_acc,
            )

            prog_bar.set_description(info)
            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs,self._cur_task,Train=True)
                logits = outputs["logits"]
                fake_targets = targets - self._known_classes
                classlogits = self._network.simple_fc(outputs["feature"],fake_targets)["logits"]
                loss_etf = F.cross_entropy(
                    logits, fake_targets
                )
                loss_ac = F.cross_entropy(
                    classlogits , fake_targets
                )
                loss = loss_etf + loss_ac
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(fake_targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            self._network.backbone.EMA(self.args["alpha"])
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args["epochs"],
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)
            logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs,self._cur_task)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
    def _compute_accuracy_EMA_Task(self, model, loader, Top=5):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets_cpu = targets.cpu()
            batch_size = inputs.size(0)
            with torch.no_grad():
                res = model.forward_EMA(inputs)
                outputs = res["logits"]
                SL_x = res["SL_x"]
                top5_values, top5_indices = torch.topk(outputs, k=Top, largest=True, sorted=True)
                Task_ids_batch = top5_indices // self.args["init_cls"]
                Task_ids_batch = Task_ids_batch.cpu().tolist()
                unique_sorted_Task_ids = [self.deduplicate_and_sort(task_ids) for task_ids in Task_ids_batch]
                all_logits = []
                for idx in range(batch_size):
                    logit = self._network.forwardnew(SL_x[idx:idx + 1], unique_sorted_Task_ids[idx])["logits"]
                    all_logits.append(logit)
                outputs = torch.cat(all_logits, dim=0)
            predicts = torch.max(outputs, dim=1)[1]
            predicts_cpu = predicts.cpu()
            correct += (predicts_cpu == targets_cpu).sum()
            total += len(targets)
        top1_accuracy = np.around(correct.item() * 100 / total, decimals=2)
        return {
            "top1_accuracy": top1_accuracy,
        }
    def deduplicate_and_sort(self,arr):
        unique_elements = set(arr)
        sorted_result = sorted(unique_elements)
        return sorted_result