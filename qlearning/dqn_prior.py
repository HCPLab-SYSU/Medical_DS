import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from qlearning.layers import NoisyLinear
from qlearning.network_bodies import SimpleBody, AtariBody

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    


class Knowledge_Graph_Reasoning(nn.Module):
    def __init__(self, num_actions, dise_start, act_cardinality, slot_cardinality, dise_sym_mat, sym_dise_mat, sym_prio):
        super(Knowledge_Graph_Reasoning, self).__init__()
        self.num_actions = num_actions
        self.dise_start = dise_start
        self.act_cardinality = act_cardinality
        self.slot_cardinality = slot_cardinality
        self.dise_sym_mat = dise_sym_mat
        self.sym_dise_mat = sym_dise_mat
        self.sym_prio = sym_prio
    def forward(self, state):
        current_slots_rep = state[:, (2*self.act_cardinality+5):(2*self.act_cardinality+self.slot_cardinality)]
        # print("slot", self.slot_cardinality)
        # print("slot shape", current_slots_rep.size())
        
        batch_size = state.size(0)
        dise_num = self.dise_sym_mat.size(0)
        sym_num = self.dise_sym_mat.size(1)
        dise_start = self.dise_start
        sym_start = self.dise_start + dise_num

        sym_prio_ = self.sym_prio.repeat(batch_size,1).view(batch_size, -1)

        zeros = torch.zeros(current_slots_rep.size()).to(device)

        # not request->use prio prob
        sym_prio_prob = torch.where(current_slots_rep == 0, sym_prio_, current_slots_rep)
        # not sure->use prio prob
        sym_prio_prob = torch.where(sym_prio_prob == -2, sym_prio_, sym_prio_prob)
        #sym_prio_prob = torch.where(sym_prio_prob == -1, zeros, sym_prio_prob)
        # print("sym_prio_prob", sym_prio_prob)

        dise_prob = torch.matmul(sym_prio_prob, self.sym_dise_mat)
        sym_prob = torch.matmul(dise_prob, self.dise_sym_mat)

        action = torch.zeros(batch_size, self.num_actions).to(device)
        action[:, dise_start:sym_start] = dise_prob
        action[:, sym_start:] = sym_prob
        # print("knowledge action", action)
        return action

class KR_DQN(nn.Module):
    def __init__(self, input_shape, hidden_size, num_actions, relation_init, dise_start, act_cardinality, slot_cardinality,  sym_dise_pro, dise_sym_pro, sym_prio):
        super(KR_DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.dise_start = dise_start
        self.act_cardinality = act_cardinality
        self.slot_cardinality = slot_cardinality
        self.sym_dise_mat = sym_dise_pro
        self.dise_sym_mat = dise_sym_pro
        self.sym_prio = sym_prio

        self.fc1 = nn.Linear(self.input_shape, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_actions)
        self.tran_mat = Parameter(torch.Tensor(relation_init.size(0),relation_init.size(1)))
        self.knowledge_branch = Knowledge_Graph_Reasoning(self.num_actions, self.dise_start, self.act_cardinality, self.slot_cardinality, 
            self.dise_sym_mat, self.sym_dise_mat, self.sym_prio)

        self.tran_mat.data = relation_init

        #self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.tran_mat.data.uniform_(-stdv, stdv)
    def forward(self, state, sym_flag):
        # print(sym_flag.size())
        x = F.relu(self.fc1(state))
        x = self.fc2(x)

        rule_res = self.knowledge_branch(state)
        relation_res = torch.matmul(x, F.softmax(self.tran_mat, 0))
        # dqn+knowledge+relation
        x = F.sigmoid(x) + F.sigmoid(relation_res) + rule_res

        x = x * sym_flag
        
        return x

    def predict(self, x, sym_flag):
        with torch.no_grad():
            a = self.forward(x, sym_flag).max(1)[1].view(1, 1)
        return a.item()



    
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5, body=SimpleBody):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy
        self.body = body(input_shape, num_actions, noisy, sigma_init)

        self.fc1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(self.body.feature_size(), 512, sigma_init)
        self.fc2 = nn.Linear(512, self.num_actions) if not self.noisy else NoisyLinear(512, self.num_actions, sigma_init)

    def forward(self, x):
        x = self.body(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc1.sample_noise()
            self.fc2.sample_noise()

    def predict(self, x):
        # print(self.fc1.weight)
        with torch.no_grad():
            self.sample_noise()
            a = self.forward(x).max(1)[1].view(1, 1)
        return a.item()



class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_outputs, noisy=False, sigma_init=0.5, body=SimpleBody):
        super(DuelingDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs
        self.noisy = noisy

        self.body = body(input_shape, num_outputs, noisy, sigma_init)

        self.adv1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, sigma_init)
        self.adv2 = nn.Linear(512, self.num_actions) if not self.noisy else NoisyLinear(512, self.num_actions,
                                                                                        sigma_init)

        self.val1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, sigma_init)
        self.val2 = nn.Linear(512, 1) if not self.noisy else NoisyLinear(512, 1, sigma_init)

    def forward(self, x):
        x = self.body(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()


class CategoricalDQN(nn.Module):
    def __init__(self, input_shape, num_outputs, noisy=False, sigma_init=0.5, body=SimpleBody, atoms=51):
        super(CategoricalDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs
        self.noisy = noisy
        self.atoms = atoms

        self.body = body(input_shape, num_outputs, noisy, sigma_init)

        self.fc1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(self.body.feature_size(),
                                                                                               512, sigma_init)
        self.fc2 = nn.Linear(512, self.num_actions * self.atoms) if not self.noisy else NoisyLinear(512,
                                                                                                    self.num_actions * self.atoms,
                                                                                                    sigma_init)

    def forward(self, x):
        x = self.body(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.softmax(x.view(-1, self.num_actions, self.atoms), dim=2)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc1.sample_noise()
            self.fc2.sample_noise()


class CategoricalDuelingDQN(nn.Module):
    def __init__(self, input_shape, num_outputs, noisy=False, sigma_init=0.5, body=SimpleBody, atoms=51):
        super(CategoricalDuelingDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs
        self.noisy = noisy
        self.atoms = atoms

        self.body = body(input_shape, num_outputs, noisy, sigma_init)

        self.adv1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, sigma_init)
        self.adv2 = nn.Linear(512, self.num_actions * self.atoms) if not self.noisy else NoisyLinear(512,
                                                                                                     self.num_actions * self.atoms,
                                                                                                     sigma_init)

        self.val1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, sigma_init)
        self.val2 = nn.Linear(512, 1 * self.atoms) if not self.noisy else NoisyLinear(512, 1 * self.atoms, sigma_init)

    def forward(self, x):
        x = self.body(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv).view(-1, self.num_actions, self.atoms)

        val = F.relu(self.val1(x))
        val = self.val2(val).view(-1, 1, self.atoms)

        final = val + adv - adv.mean(dim=1).view(-1, 1, self.atoms)

        return F.softmax(final, dim=2)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()


class QRDQN(nn.Module):
    def __init__(self, input_shape, num_outputs, noisy=False, sigma_init=0.5, body=SimpleBody, quantiles=51):
        super(QRDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs
        self.noisy = noisy
        self.quantiles = quantiles

        self.body = body(input_shape, num_outputs, noisy, sigma_init)

        self.fc1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(self.body.feature_size(),
                                                                                               512, sigma_init)
        self.fc2 = nn.Linear(512, self.num_actions * self.quantiles) if not self.noisy else NoisyLinear(512,
                                                                                                        self.num_actions * self.quantiles,
                                                                                                        sigma_init)

    def forward(self, x):
        x = self.body(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x.view(-1, self.num_actions, self.quantiles)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc1.sample_noise()
            self.fc2.sample_noise()


class DuelingQRDQN(nn.Module):
    def __init__(self, input_shape, num_outputs, noisy=False, sigma_init=0.5, body=SimpleBody, quantiles=51):
        super(DuelingQRDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs
        self.noisy = noisy
        self.quantiles = quantiles

        self.body = body(input_shape, num_outputs, noisy, sigma_init)

        self.adv1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, sigma_init)
        self.adv2 = nn.Linear(512, self.num_actions * self.quantiles) if not self.noisy else NoisyLinear(512,
                                                                                                         self.num_actions * self.quantiles,
                                                                                                         sigma_init)

        self.val1 = nn.Linear(self.body.feature_size(), 512) if not self.noisy else NoisyLinear(
            self.body.feature_size(), 512, sigma_init)
        self.val2 = nn.Linear(512, 1 * self.quantiles) if not self.noisy else NoisyLinear(512, 1 * self.quantiles,
                                                                                          sigma_init)

    def forward(self, x):
        x = self.body(x)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv).view(-1, self.num_actions, self.quantiles)

        val = F.relu(self.val1(x))
        val = self.val2(val).view(-1, 1, self.quantiles)

        final = val + adv - adv.mean(dim=1).view(-1, 1, self.quantiles)

        return final

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()


########Recurrent Architectures#########

class DRQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5, gru_size=512, bidirectional=False,
                 body=SimpleBody):
        super(DRQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy
        self.gru_size = gru_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.body = body(input_shape, num_actions, noisy=self.noisy, sigma_init=sigma_init)
        self.gru = nn.GRU(self.body.feature_size(), self.gru_size, num_layers=1, batch_first=True,
                          bidirectional=bidirectional)
        self.fc2 = nn.Linear(self.gru_size, self.num_actions) if not self.noisy else NoisyLinear(self.gru_size,
                                                                                                 self.num_actions,
                                                                                                 sigma_init)

    def forward(self, x, hx=None):
        batch_size = x.size(0)
        sequence_length = x.size(1)

        x = x.view((-1,) + self.input_shape)

        # format outp for batch first gru
        feats = self.body(x).view(batch_size, sequence_length, -1)
        hidden = self.init_hidden(batch_size) if hx is None else hx
        out, hidden = self.gru(feats, hidden)
        x = self.fc2(out)

        return x, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1 * self.num_directions, batch_size, self.gru_size, device=device, dtype=torch.float)

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc2.sample_noise()


########Actor Critic Architectures#########
class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCritic, self).__init__()

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                                          lambda x: nn.init.constant_(x, 0),
                                          nn.init.calculate_gain('relu'))

        self.conv1 = init_(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4))
        self.conv2 = init_(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 32, kernel_size=3, stride=1))
        self.fc1 = init_(nn.Linear(self.feature_size(input_shape), 512))

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                                          lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(512, 1))

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                                          lambda x: nn.init.constant_(x, 0), gain=0.01)

        self.actor_linear = init_(nn.Linear(512, num_actions))

        self.train()

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        value = self.critic_linear(x)
        logits = self.actor_linear(x)

        return logits, value

    def feature_size(self, input_shape):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape)))).view(1, -1).size(1)

    def layer_init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module
