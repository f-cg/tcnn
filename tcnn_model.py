import getpass
import operator
import pdb
import time
from copy import deepcopy
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

user_name = getpass.getuser()
drive = './' if user_name == 'lily' else '/gdrive/My Drive/'
np.set_printoptions(precision=3, suppress=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
leafid = 0
nodeid = 0
leaves = []
save_target_on_leaves = False
prob_on_leaves = [0]*16

prob_on_nodes = [0]*64


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class InnerNode():

    def __init__(self, depth, args):
        self.args = args
        self.module = deepcopy(args.modules[depth-1])
        oflatten = args.module_oshape[depth-1]
        if not isinstance(oflatten, int):
            oflatten = reduce(operator.mul, oflatten, 1)
        self.fc = nn.Linear(oflatten, 1)
        # beta = torch.randn(1, device=device)
        beta = torch.tensor([1.0], device=device)
        self.beta = nn.Parameter(beta)
        self.leaf = False
        global nodeid
        # 前序遍历DLR
        nodeid += 1
        self.nodeid = nodeid
        self.depth = depth
        self.prob = None
        self.leaf_accumulator = []
        self.lmbda = self.args.lmbda * 2 ** (-depth)
        # 创建子树
        self.build_child(depth)
        self.penalties = []

    def reset(self):
        self.leaf_accumulator = []
        self.penalties = []
        self.left.reset()
        self.right.reset()

    def build_child(self, depth):
        if depth < self.args.max_depth:
            self.left = InnerNode(depth+1, self.args)
            self.right = InnerNode(depth+1, self.args)
        else:
            self.left = LeafNode(self.args)
            self.right = LeafNode(self.args)

    def forward(self, x):
        xo = self.module(x)
        x = xo.view(xo.shape[0], -1)
        rprob = F.sigmoid(self.beta*self.fc(x))
        if torch.isnan(rprob).any():
            print(self.depth)
            print('self.beta:')
            print(self.beta)
            print('x:')
            print(x)
            print('self.fc(x):')
            print(self.fc(x))
            print('rprob:')
            print(rprob)
            pdb.set_trace()
        return(rprob, xo)

    def forward_hook(self, x):
        xo = self.module(x)
        x = xo.view(xo.shape[0], -1)
        rprob = self.beta*self.fc(x)
        return(rprob, xo)

    '''
    def select_next(self, x):
        prob = self.forward(x)
        if prob < 0.5:
            return(self.left, prob)
        else:
            return(self.right, prob)
    '''

    def cal_prob(self, x, path_prob):
        self.prob, xo = self.forward(x)  # probability of selecting right node
        self.path_prob = path_prob
        left_leaf_accumulator = self.left.cal_prob(
            xo, path_prob * (1-self.prob))
        right_leaf_accumulator = self.right.cal_prob(xo, path_prob * self.prob)
        self.leaf_accumulator.extend(left_leaf_accumulator)
        self.leaf_accumulator.extend(right_leaf_accumulator)
        return(self.leaf_accumulator)

    def inferrence(self, x, path_prob):
        prob, xo = self.forward(x)  # probability of selecting right node
        rp = path_prob * prob
        lp = path_prob * (1-prob)
        global prob_on_nodes
        prob_on_nodes[self.nodeid] = (lp, rp)
        self.left.inferrence(xo, lp)
        self.right.inferrence(xo, rp)

    def inferrence_hook(self, x, path_prob, hooknode):
        print('nodeid depth', self.nodeid, hooknode, self.depth)
        if self.nodeid == hooknode:
            p, xo = self.forward_hook(x)
            return (xo, p)
        prob, xo = self.forward(x)  # probability of selecting right node
        rp = path_prob * prob
        lp = path_prob * (1-prob)
        result = self.left.inferrence_hook(xo, lp, hooknode)
        if result is not None:
            return result
        result = self.right.inferrence_hook(xo, rp, hooknode)
        if result is not None:
            return result

    def single_inferrence(self, x, node_probs):
        prob, xo = self.forward(x)  # probability of selecting right node
        node_probs.append(prob.cpu().squeeze().item())
        if prob > 0.5:
            return self.right.single_inferrence(xo, node_probs)
        else:
            return self.left.single_inferrence(xo, node_probs)

    def cal_prob_hard(self, x, path_prob):
        prob, xo = self.forward(x)  # probability of selecting right node
        prob[prob < 0.5] = 0
        prob[prob > 0.5] = 1
        self.prob = prob
        self.path_prob = path_prob
        left_leaf_accumulator = self.left.cal_prob_hard(
            xo, path_prob * (1-self.prob))
        right_leaf_accumulator = self.right.cal_prob_hard(
            xo, path_prob * self.prob)
        self.leaf_accumulator.extend(left_leaf_accumulator)
        self.leaf_accumulator.extend(right_leaf_accumulator)
        return(self.leaf_accumulator)

    def get_penalty(self):
        penalty = (torch.sum(self.prob * self.path_prob) /
                   torch.sum(self.path_prob), self.lmbda)
        if not self.left.leaf:
            left_penalty = self.left.get_penalty()
            right_penalty = self.right.get_penalty()
            self.penalties.append(penalty)
            self.penalties.extend(left_penalty)
            self.penalties.extend(right_penalty)
        return(self.penalties)


class LeafNode():
    def __init__(self, args):
        self.args = args
        self.param = torch.randn(self.args.output_dim, device=device)
#        self.param = torch.ones(self.args.output_dim)/self.args.output_dim
        self.param = nn.Parameter(self.param)
        leaves.append(self)
        self.leaf = True
        global leafid
        self.leafid = leafid
        leafid += 1
        self.softmax = nn.Softmax()
        self.hard = False

    def forward(self):
        return(self.softmax(self.param.view(1, -1)))

    def reset(self):
        pass

    def cal_prob(self, x, path_prob):
        Q = self.forward()  # (1, odim)
        Q = Q.expand((path_prob.size()[0], self.args.output_dim))  # (bs, odim)
        return([[path_prob, Q]])

    def inferrence(self, x, path_prob):
        return

    def inferrence_hook(self, x, path_prob, hooknode):
        return

    def single_inferrence(self, x, node_probs):
        return self.forward()

    def cal_prob_hard(self, x, path_prob):
        if save_target_on_leaves:
            prob_on_leaves[self.leafid] = path_prob.type(torch.bool)
        Q = self.forward()  # (1, odim)
        Q = Q.expand((path_prob.size()[0], self.args.output_dim))  # (bs, odim)
        return([[path_prob, Q]])


class SoftDecisionTree(nn.Module):

    def __init__(self, args, net):
        super(SoftDecisionTree, self).__init__()
        self.args = args
        self.args.output_dim = 10
        ml = list(net.modules())
        self.args.modules = [nn.Sequential(
            *ml[1:3]), nn.Sequential(*ml[3:5]),
            nn.Sequential(Flatten(), ml[5]), ml[6]]
        if args.model == 'resnet18':
            self.args.modules = [nn.Sequential(
                net.conv1, net.bn1, nn.ReLU(), net.layer1),
                net.layer2, net.layer3,
                nn.Sequential(net.layer4, nn.AvgPool2d(4)),
                nn.Sequential(Flatten(), net.linear)]
        elif args.model == 'naive':
            self.args.modules = [
                nn.Sequential(net.conv1, nn.Sigmoid(), net.mp1),
                nn.Sequential(net.conv2, nn.Sigmoid(), net.mp2),
                nn.Sequential(Flatten(), net.fc1, nn.Sigmoid()),
                net.fc2]
        elif args.model == 'naive_norm':
            self.args.modules = [
                nn.Sequential(net.conv1, nn.Sigmoid(), net.mp1, net.bn1),
                nn.Sequential(net.conv2, nn.Sigmoid(), net.mp2, net.bn2),
                nn.Sequential(Flatten(), net.fc1, nn.Sigmoid(), net.bn3),
                net.fc2]
        elif args.model == 'naive_linear':
            self.args.modules = [
                nn.Sequential(net.conv1, net.mp1),
                nn.Sequential(net.conv2),
                nn.Sequential(net.conv3, net.mp3),
                nn.Sequential(Flatten(), net.fc1),
            ]

        if args.dataset == 'MNIST':
            args.max_depth = 4
            if args.model == 'naive' or args.model == 'naive_norm':
                self.args.module_oshape = [(6, 12, 12), (16, 5, 5), 64, 10]
            elif args.model == 'naive_linear':
                self.args.module_oshape = [
                    (6, 12, 12), (6, 12, 12), (16, 5, 5), 10]
            else:
                print('dataset and model not match')
                exit(0)
        elif args.dataset == 'CIFAR10':
            if args.model == 'resnet18':
                self.args.module_oshape = [
                    # (64, 32, 32), (64, 32, 32), (128, 16, 16),
                    # (256, 8, 8), (512, 4, 4), (512, 1, 1), 10]
                    (64, 32, 32), (128, 16, 16), (256, 8, 8), (512, 1, 1), 10]
                args.max_depth = 5
            elif args.model == 'naive' or args.model == 'naive_linear':
                self.args.module_oshape = [(6, 14, 14), (16, 6, 6), 64, 10]
                args.max_depth = 4
            else:
                print('dataset and model not match')
                exit(0)
        else:
            print(args.dataset)
            exit(-1)

        self.root = InnerNode(1, self.args)
        self.collect_parameters()  # collect parameters and modules
        if args.optim == 'Adam':
            self.optimizer = optim.Adam(self.parameters(),
                                        lr=self.args.lr,
                                        weight_decay=args.weight_decay)
        elif args.optim == 'SGD':
            self.optimizer = optim.SGD(self.parameters(),
                                       lr=self.args.lr, momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        self.define_extras(self.args.batch_size)
        self.best_accuracy = 0.0

    def define_extras(self, batch_size):
        # define target_onehot and path_prob_init batch size
        self.target_onehot = torch.FloatTensor(
            batch_size, self.args.output_dim).to(device)
        self.path_prob_init = torch.ones(batch_size, 1, device=device)
    '''
    def forward(self, x):
        node = self.root
        path_prob = Variable(torch.ones(self.args.batch_size, 1))
        while not node.leaf:
            node, prob = node.select_next(x)
            path_prob *= prob
        return node()
    '''

    def get_node_prob(self, x):
        self.root.inferrence(x, torch.tensor([1.0], device=device))
        return prob_on_nodes

    def get_node_hook(self, x, hooknode):
        # get feature_map,lp,rp
        return self.root.inferrence_hook(x, torch.tensor([1.0], device=device), hooknode)

    def single_inferrence(self, x, node_probs):
        return self.root.single_inferrence(x, node_probs)

    def cal_loss(self, x, y):
        batch_size = y.size()[0]
        leaf_accumulator = self.root.cal_prob(
            x, self.path_prob_init)  # (leaf_nums, )
        loss = 0.
#        max_prob = [-1. for _ in range(batch_size)]
        max_prob = [-1]*batch_size
#        max_Q = [torch.zeros(self.args.output_dim)
#                 for _ in range(batch_size)]
        max_Q = [0]*batch_size
        for (path_prob, Q) in leaf_accumulator:
            TQ = torch.bmm(y.view(batch_size, 1, self.args.output_dim), torch.log(
                Q).view(batch_size, self.args.output_dim, 1)).view(-1, 1)  # 交叉熵
            loss += path_prob * TQ  # 该叶节点处的概率乘以TQ
            path_prob_numpy = path_prob.cpu().data.numpy().reshape(-1)
            for i in range(batch_size):
                #                try:
                #                    assert(path_prob[i] >= 0)
                #                except:
                #                    print(leaf_accumulator)
                #                    print(path_prob[i] < 0)
                #                    print(path_prob[i] >= 0)
                #                    exit(-1)
                #
                if max_prob[i] < path_prob_numpy[i]:
                    max_prob[i] = path_prob_numpy[i]
                    max_Q[i] = Q[i]
        loss = loss.mean()
        penalties = self.root.get_penalty()
        C = 0.
        for (penalty, lmbda) in penalties:
            C -= lmbda * 0.5 * (torch.log(penalty) + torch.log(1-penalty))
#        for mq in max_Q:
#            print(mq.device, end=' ')
        output = torch.stack(max_Q)
        self.root.reset()  # reset all stacked calculation
        # -log(loss) will always output non, because loss is always below zero.
        # I suspect this is the mistake of the paper?
        return(-loss + C, output)
        return(-loss, output)

    def cal_loss_hard(self, x, y):
        batch_size = y.size()[0]
        leaf_accumulator = self.root.cal_prob_hard(
            x, self.path_prob_init)  # (leaf_nums, )
        loss = 0.
        max_prob = [-1. for _ in range(batch_size)]
        max_Q = [torch.zeros(self.args.output_dim)
                 for _ in range(batch_size)]
        for (path_prob, Q) in leaf_accumulator:  # 交叉熵
            TQ = torch.bmm(y.view(batch_size, 1, self.args.output_dim), torch.log(
                Q).view(batch_size, self.args.output_dim, 1)).view(-1, 1)
            loss += path_prob * TQ  # 该叶节点处的概率乘以TQ
            path_prob_numpy = path_prob.cpu().data.numpy().reshape(-1)
            for i in range(batch_size):
                if max_prob[i] < path_prob_numpy[i]:
                    max_prob[i] = path_prob_numpy[i]
                    max_Q[i] = Q[i]
        loss = loss.mean()
        penalties = self.root.get_penalty()
        C = 0.
        for (penalty, lmbda) in penalties:
            C -= lmbda * 0.5 * (torch.log(penalty) + torch.log(1-penalty))
        output = torch.stack(max_Q)
        self.root.reset()  # reset all stacked calculation
        # -log(loss) will always output non, because loss is always below zero.
        # I suspect this is the mistake of the paper?
        return(-loss + C, output)
        return(-loss, output)

    def collect_parameters(self):
        nodes = [self.root]
        self.module_list = nn.ModuleList()
        self.frozen_module_list = nn.ModuleList()
        self.param_list = nn.ParameterList()
        # 广度优先遍历
        while nodes:
            node = nodes.pop(0)
            if node.leaf:
                param = node.param
                self.param_list.append(param)
            else:
                fc = node.fc
                beta = node.beta
                nodes.append(node.right)
                nodes.append(node.left)
                self.param_list.append(beta)
                self.module_list.append(fc)
                self.frozen_module_list.append(node.module)

    def count_parameters(self):
        psum = (lambda ps: sum(p.numel() for p in ps))
        allcount = psum(self.parameters())
        one = 0
        node = self.root
        while not node.leaf:
            one += psum(node.fc.parameters())+psum(node.beta) + \
                psum(node.module.parameters())
            print(one)
            node = node.left
        print('all: {} , one: {}'.format(allcount, one))

    def freeze(self, freeze_flag=True):
        for m in self.frozen_module_list:
            for p in m.parameters():
                p.requires_grad = not freeze_flag
#        if freeze_flag:
#            for p in self.param_list:
#                p.requires_grad = False
        print(['un', ''][freeze_flag]+'freeze')

    def targeted_dropout(self):
        gamma = 0.75
        alpha = 0.66
        vec = torch.nn.utils.parameters_to_vector(
            self.frozen_module_list.parameters())
        length = len(vec)
        idx = vec.abs().topk(int(length*gamma), largest=False)[1]
        mask = torch.rand(idx.shape[0], device=device) > alpha
        mask = mask.type(torch.float)
        vec[idx] = vec[idx].mul(mask)
        torch.nn.utils.vector_to_parameters(
            vec, self.frozen_module_list.parameters())

    def targeted_dropout_all(self):
        gamma = 0.75
        alpha = 0.66
        vec = torch.nn.utils.parameters_to_vector(self.parameters())
        length = len(vec)
        idx = vec.abs().topk(int(length*gamma), largest=False)[1]
        mask = torch.rand(idx.shape[0], device=device) > alpha
        mask = mask.type(torch.float)
        vec[idx] = vec[idx].mul(mask)
        torch.nn.utils.vector_to_parameters(vec, self.parameters())

    def dropout_rate(self, r):
        vec = torch.nn.utils.parameters_to_vector(
            self.frozen_module_list.parameters())
        length = len(vec)
        idx = vec.abs().topk(int(length*r), largest=False)[1]
        vec[idx] = 0.
        torch.nn.utils.vector_to_parameters(
            vec, self.frozen_module_list.parameters())

    def get_leaves(self):
        leavesv = []
        for leaf in leaves:
            leaf = leaf.forward().data.cpu().numpy()
            leavesv.append(leaf)
        return leavesv

    def make_hard(self):
        print('make_hard')
        for leaf in leaves:
            leaf.param.requires_grad_(False)
            idx, mv = leaf.param.argmax(), leaf.param.max()
            # leaf.param.data.fill_(leaf.param.min())
            leaf.param.data.fill_(-3.0)
            leaf.param.data[idx] = 5.0
            leaf.hard = True
            leaf.param.requires_grad_(False)

    def check(self):
        for k, v in self.state_dict().items():
            print(k, v.device)
            if v.device == 'cpu':
                print('oops!\n\n')

    def single_test(self, loader):
        self.eval()
        correct, total = 0, 0
        node_probs = []
        for x, y in loader.dataset:
            x = x.unsqueeze(0).to(device)
            p = self.single_inferrence(x, node_probs)
            if p.argmax() == y:
                correct += 1
            total += 1
        print('single_test acc:{:.3f}'.format(correct/total))
        return node_probs

    def train_(self, loader, metric_saved, classes, stage_print=True,
               train_flag=True, hard=True, save_target_on_leaves_=True,
               mode='test'):
        t1 = time.time()
        if train_flag:
            self.train()
        else:
            self.eval()
            leaves = self.get_leaves()
            if metric_saved:
                metric_saved['leaves'].append(np.array(leaves).tolist())
        self.define_extras(self.args.batch_size)
        total_count = 0
        correct = 0
        hard_correct = 0
        class_correct = torch.zeros(classes)
        class_total = [0]*classes
        hard_class_correct = torch.zeros(classes)
        if save_target_on_leaves_:
            global prob_on_leaves, save_target_on_leaves
            save_target_on_leaves = True
            prob_on_leaves = [0]*2**self.args.max_depth
            fr_leaves = [[0]*10 for _ in range(2**self.args.max_depth)]

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            target_ = target.view(-1, 1)
            batch_size = target_.size()[0]
            total_count += len(data)
            # convert int target to one-hot vector
            # because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
            if not batch_size == self.args.batch_size:
                self.define_extras(batch_size)
            self.target_onehot.zero_()
            self.target_onehot.scatter_(1, target_, 1.)
            # Soft
            if train_flag:
                self.optimizer.zero_grad()
            loss, output = self.cal_loss(data, self.target_onehot)
            if train_flag:
                with autograd.detect_anomaly():
                    loss.backward()
                    self.optimizer.step()
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
#            if output.device == 'cpu':
#                print(output.device, target.data.device)
#                self.check()
            c = pred.eq(target.data).cpu()
            correct += c.sum()
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += c[i]
                class_total[label] += 1

            # Hard
            if hard:
                loss, output = self.cal_loss_hard(data, self.target_onehot)
                pred = output.data.max(1)[1]
                c = pred.eq(target.data).cpu()
                hard_correct += c.sum()
                for i in range(len(target)):
                    label = target[i]
                    hard_class_correct[label] += c[i]
                if save_target_on_leaves_:
                    for lfid, leafprob in enumerate(prob_on_leaves):
                        target_here = target[leafprob.view(-1)]
                        for i in range(len(target_here)):
                            label = target_here[i]
                            fr_leaves[lfid][label] += 1
#            if train and batch_idx % self.args.log_interval == 0:
#                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},
#                        Accuracy: {}/{} ({:.4f}%) hard_accuracy:{}%'.format(
#                    epoch, batch_idx * len(data), len(train_loader.dataset),
#                    100. * batch_idx / len(train_loader), loss.data.item(),
#                    correct, len(data),
#                    total_acc, hard_total_acc))
        # Soft
        total_acc = 100. * correct.item() / total_count
        class_correct = class_correct.numpy()
        class_accs = 100*class_correct/class_total
        if metric_saved:
            metric_saved[mode]['total_acc'].append(total_acc)
            metric_saved[mode]['class_accs'].append(class_accs)
        t = time.time()-t1
        # Hard
        if hard:
            hard_total_acc = 100. * hard_correct.item() / total_count
            hard_class_correct = hard_class_correct.numpy()
            hard_class_accs = 100*hard_class_correct/class_total
            if metric_saved:
                metric_saved[mode]['hard_total_acc'].append(hard_total_acc)
                metric_saved[mode]['hard_class_accs'].append(hard_class_accs)
            summary = '{}%:{:.2f},{:.2f} time:{:.2f}s'.format(
                mode, total_acc, hard_total_acc, t)
            if save_target_on_leaves_:
                if metric_saved:
                    metric_saved['fr_leaves'].append(fr_leaves)
        else:
            summary = '{}%:{:.2f} time:{:.2f}s'.format(
                mode, total_acc, t)

        print(summary)
        return total_acc
