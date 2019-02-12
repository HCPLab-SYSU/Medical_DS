import random, copy
import pickle as pickle
import numpy as np

import dialog_config

from .agents import Agent
from .utils import *
from qlearning.dqn import DQN


class AgentDQN(Agent):
    def __init__(self, sym_dict=None, dise_sym_dict=None, req_dise_sym_dict=None, dise_sym_num_dict=None,
                 dise_sym_matrix=None, act_set=None, slot_set=None, params=None):
        self.sym_dict = sym_dict  # all symptoms
        self.dise_sym_dict = dise_sym_dict  # dise sym relations
        self.req_dise_sym_dict = req_dise_sym_dict  # dise with high freq syms
        self.dise_sym_num_dict = dise_sym_num_dict  # dise and sym freq
        self.act_set = act_set  # all acts
        self.slot_set = slot_set  # all slots
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())
        self.dise_sym_matrix = dise_sym_matrix  # dise with pro of syms
        self.dise_num = dise_sym_matrix.shape[0]  # dise nume
        self.feasible_actions = dialog_config.feasible_actions  # action space
        self.num_actions = len(self.feasible_actions)

        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
        self.experience_replay_pool = []  # experience replay pool <s_t, a_t, r_t, s_t+1>
        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 1000)

        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)

        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)
        self.supervise = params.get('supervise', 0)
        self.max_turn = params['max_turn'] + 4
        # self.state_dimension = 2 * self.act_cardinality + 3 * self.slot_cardinality + self.max_turn
        self.state_dimension = 2 * self.act_cardinality + 1 * self.slot_cardinality + self.max_turn

        self.mask = params.get('mask', 0)
        self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions)
        self.clone_dqn = copy.deepcopy(self.dqn)

        self.cur_bellman_err = 0

        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            self.dqn.model = copy.deepcopy(self.load_trained_DQN(params['trained_model_path']))
            self.clone_dqn = copy.deepcopy(self.dqn)
            self.predict_mode = True
            self.supervise = 2
            self.warm_start = 2

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """
        self.current_slots = {}
        # self.current_slot_id = 0
        self.phase = 0
        self.request_set = copy.deepcopy(dialog_config.sys_request_slots_highfreq)

    def simu_state_to_action(self, state, goal):

        im_syms = goal['implicit_inform_slots']
        current_slots = state['current_slots']['inform_slots']
        left_slots = [k for k in im_syms if k not in list(current_slots.keys())]
        # print(left_slots)
        action = -1
        if len(left_slots) == 0:
            action = dialog_config.sys_inform_slots_values.index(goal['disease_tag']) + 2
        else:
            slot = random.choice(left_slots)
            action = 2 + len(dialog_config.sys_inform_slots_values) + dialog_config.sys_request_slots.index(slot)
        return action

    def state_to_action(self, state, goal=None):
        """ DQN: Input state, output action """

        self.representation = self.prepare_state_representation(state)
        if self.supervise == 1:
            self.action = self.simu_state_to_action(state, goal)
        else:
            self.action = self.run_policy(self.representation, state)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        # print(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}

    def prepare_state_representation(self, state):
        """ Create the representation for each state """
        not_sure = 0.2
        user_action = state['user_action']
        current_slots = state['current_slots']
        agent_last = state['agent_action']

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            if slot not in self.slot_set:
                continue
            user_inform_slots_rep[0, self.slot_set[slot]] = user_action['inform_slots'][slot]
            if user_action['inform_slots'][slot] == -2:
                user_inform_slots_rep[0, self.slot_set[slot]] = not_sure

        # ########################################################################
        # #   Create bag of request slots representation to represent the current user action
        # ########################################################################
        # user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        # for slot in user_action['request_slots'].keys():
        #     if slot not in self.slot_set:
        #         continue
        #     user_request_slots_rep[0, self.slot_set[slot]] = -2

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            if slot not in self.slot_set:
                continue
            current_slots_rep[0, self.slot_set[slot]] = current_slots['inform_slots'][slot]
            if current_slots['inform_slots'][slot] == -2:
                current_slots_rep[0, self.slot_set[slot]] = not_sure

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        # agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        # if agent_last:
        #     for slot in agent_last['inform_slots'].keys():
        #         if slot not in self.slot_set:
        #             continue
        #         agent_inform_slots_rep[0, self.slot_set[slot]] = agent_last['inform_slots'][slot]

        # ########################################################################
        #   Encode last agent request slots
        # ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                if slot not in self.slot_set:
                    continue
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        # turn_rep = np.zeros((1,1)) + state['turn'] / 10.

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        # kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        # for slot in kb_results_dict:
        #     if slot in self.slot_set:
        #         kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        # kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum( kb_results_dict['matching_all_constraints'] > 0.)
        # for slot in kb_results_dict:
        #     if slot in self.slot_set:
        #         kb_binary_rep[0, self.slot_set[slot]] = np.sum( kb_results_dict[slot] > 0.)

        # self.final_representation = np.hstack([user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep, agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep])
        # self.final_representation = np.hstack([user_act_rep, user_inform_slots_rep, agent_act_rep, agent_request_slots_rep, current_slots_rep, turn_onehot_rep])
        self.final_representation = np.hstack([user_act_rep, agent_act_rep, current_slots_rep, turn_onehot_rep])

        return self.final_representation

    def search_batch_action_mask_1(self, Xs):
        batch_size = Xs.shape[0]
        # print(Xs.shape)
        dise_start = 2
        sym_start = dise_start + self.dise_num
        current_slots_rep = Xs[:, (2 * self.act_cardinality):(
        2 * self.act_cardinality + self.slot_cardinality)]  # representation of current slot

        action_mask = np.zeros((batch_size, self.num_actions))
        # sym_flag = np.where(current_slots_rep == 0, 1, np.nan)
        sym_flag = np.where(current_slots_rep == 0, 1, 0)
        sym_flag_1 = np.where(current_slots_rep == 0, 1, -1)
        tmp_dise_pro = np.dot(current_slots_rep, self.dise_sym_matrix.transpose())
        dise_pro = softmax_2d(tmp_dise_pro)
        # action_mask[:,dise_start:sym_start] = dise_pro*self.dise_num/self.slot_cardinality
        action_mask[:, dise_start:sym_start] = dise_pro
        # minus_dise_pro = np.where(dise_pro-0.3 > 0, dise_pro-0.3, 0)
        # action_mask[:, dise_start:sym_start] = np.where(dise_pro > 0.3, dise_pro, minus_dise_pro) # only save dise with pro > 0.5

        # tmp_dise_pro = np.where(tmp_dise_pro == 0, np.nan, tmp_dise_pro) # nan multiply any number is nan
        # multiply each sym with dise pro
        tmp_sym_pro = np.repeat(tmp_dise_pro, self.slot_cardinality, axis=0).reshape(batch_size, self.slot_cardinality,
                                                                                     -1) * (
                      self.dise_sym_matrix.transpose())
        # tmp_sym_pro = np.where(tmp_sym_pro == 0, np.inf, tmp_sym_pro)  # replace zeros with inf
        # tmp_sym_pro = np.where(tmp_sym_pro == np.nan, 0, tmp_sym_pro)  # replace nan back with zeros
        sym_pro = np.max(tmp_sym_pro, 2)
        # action_mask[:, sym_start:] = sym_pro*sym_flag
        # action_mask[:, sym_start:] = softmax_2d(sym_pro*sym_flag)
        action_mask[:, sym_start:] = softmax_2d(sym_pro) * sym_flag_1
        return action_mask

    def search_batch_action_mask(self, Xs):
        batch_size = Xs.shape[0]
        # print(Xs.shape)
        dise_start = 2
        sym_start = dise_start + self.dise_num
        current_slots_rep = Xs[:, (2 * self.act_cardinality):(
        2 * self.act_cardinality + self.slot_cardinality)]  # representation of current slot

        action_mask = np.zeros((batch_size, self.num_actions))
        # sym_flag = np.where(current_slots_rep == 0, 1, np.nan)
        sym_flag = np.where(current_slots_rep == 0, 1, 0)
        sym_flag_1 = np.where(current_slots_rep == 0, 1, -1)
        tmp_dise_pro = np.dot(current_slots_rep, self.dise_sym_matrix.transpose())
        dise_pro = softmax_2d(tmp_dise_pro)
        action_mask[:, dise_start:sym_start] = dise_pro
        # action_mask[:,dise_start:sym_start] = dise_pro*self.dise_num/self.slot_cardinality
        # minus_dise_pro = np.where(dise_pro-0.3 > 0, dise_pro-0.3, 0)
        # action_mask[:, dise_start:sym_start] = np.where(dise_pro > 0.3, dise_pro, minus_dise_pro) # only save dise with pro > 0.5

        # tmp_dise_pro = np.where(tmp_dise_pro == 0, np.nan, tmp_dise_pro) # nan multiply any number is nan
        # multiply each sym with dise pro
        tmp_sym_pro = np.repeat(tmp_dise_pro, self.slot_cardinality, axis=0).reshape(batch_size, self.slot_cardinality,
                                                                                     -1) * (
                      self.dise_sym_matrix.transpose())
        # tmp_sym_pro = np.where(tmp_sym_pro == 0, np.inf, tmp_sym_pro)  # replace zeros with inf
        # tmp_sym_pro = np.where(tmp_sym_pro == np.nan, 0, tmp_sym_pro)  # replace nan back with zeros
        sym_pro = np.max(tmp_sym_pro, 2)
        # action_mask[:, sym_start:] = sym_pro*sym_flag
        # action_mask[:, sym_start:] = softmax_2d(sym_pro*sym_flag)
        action_mask[:, sym_start:] = softmax_2d(sym_pro) * sym_flag_1
        return action_mask

    def run_policy(self, representation, state):
        """ epsilon-greedy policy """

        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            # if warm_start, pick action by rule
            if self.warm_start == 1:
                # if pool size > defined size, exit warm start
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy(state)
            else:
                if self.mask == 1:
                    Xs = representation[0].reshape(1, -1)
                    action_mask = self.search_batch_action_mask(Xs)
                else:
                    action_mask = np.ones((1, self.num_actions))
                return self.dqn.predict(representation, action_mask, {}, predict_model=True)

    def disease_from_dict(self, current_slots):
        flag = ""
        for dise in self.req_dise_sym_dict:
            flag = dise
            for sym in self.req_dise_sym_dict[dise]:
                if sym not in current_slots['inform_slots'] or current_slots['inform_slots'][sym] != True:
                    flag = ""
                    break
        return flag

    def disease_from_dict_rate(self, current_slots):
        flag = ""
        # if all slot informed and choose dise with largest sym rate, flag = "" when no
        max_sym_rate = 0.0
        for dise in self.dise_sym_num_dict:
            tmp = [v for v in self.dise_sym_num_dict[dise].keys() if v in current_slots['inform_slots'].keys()]
            tmp_sum = 0
            dise_sym_sum = 0
            for sym in tmp:
                tmp_sum += dise_sym_num_dict[dise][sym]
            for sym in dise_sym_num_dict[dise]:
                dise_sym_sum += dise_sym_num_dict[dise][sym]
            # tmp_rate = float(len(tmp))/float(len(self.req_dise_sym_dict[dise]))
            tmp_rate = float(tmp_sum) / float(dise_sym_num)
            if tmp_rate > max_sym_rate:
                max_sym_rate = tmp_rate
                flag = dise
        return flag

    def rule_policy_1(self, state):
        """ Rule Policy """
        current_slots = state['current_slots']
        dise = self.disease_from_dict(current_slots)
        act_slot_response = {}
        # if disease can be decided
        sym_flag = 1
        for sym in self.request_set:
            if sym not in current_slots['inform_slots'].keys():
                sym_flag = 0

        if dise != "":
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {'disease': dise, 'taskcomplete': "PLACEHOLDER"}
            act_slot_response['request_slots'] = {}

        # if no dise has been satisfied, choose on request

        elif sym_flag == 0:
            # if still has sym in request set not asked
            # choose the most related dise's sym to ask
            left_set = [v for v in self.request_set if v not in current_slots['inform_slots'].keys()]
            slot = random.choice(left_set)
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}

        # if sym in request set has been asked but no disease detect all its syms
        # choose the one with most rate
        elif self.phase == 0:
            dise = self.disease_from_dict_rate(current_slots)
            if dise != "":
                act_slot_response['diaact'] = "inform"
                act_slot_response['inform_slots'] = {'disease': dise, 'taskcomplete': "PLACEHOLDER"}
                act_slot_response['request_slots'] = {}
            else:
                act_slot_response = {'diaact': "inform",
                                     'inform_slots': {'disease': 'UNK', 'taskcomplete': "PLACEHOLDER"},
                                     'request_slots': {}}
            self.phase += 1
        # last thanks
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}

        return self.action_index(act_slot_response)

    def rule_policy(self, state):
        """ Rule Policy """
        current_slots = state['current_slots']
        dise = self.disease_from_dict(current_slots)
        act_slot_response = {}
        # if disease can be decided
        sym_flag = 1
        for sym in self.request_set:
            if sym not in current_slots['inform_slots'].keys():
                sym_flag = 0

        if dise != "":
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {'disease': dise, 'taskcomplete': "PLACEHOLDER"}
            act_slot_response['request_slots'] = {}

        # if no dise has been satisfied, choose on request

        elif sym_flag == 0:
            # if still has sym in request set not asked
            # choose the most related dise's sym to ask
            dise_sym_rate = {}
            for dise in self.dise_sym_num_dict:
                if dise not in dise_sym_rate:
                    dise_sym_rate[dise] = 0
                tmp = [v for v in self.dise_sym_num_dict[dise].keys() if v in current_slots['inform_slots'].keys()]
                tmp_sum = 0
                dise_sym_sum = 0
                for sym in tmp:
                    tmp_sum += self.dise_sym_num_dict[dise][sym]
                for sym in self.dise_sym_num_dict[dise]:
                    dise_sym_sum += self.dise_sym_num_dict[dise][sym]
                # dise_sym_rate[dise] = float(len(tmp))/float(len(self.dise_sym_num_dict[dise]))
                dise_sym_rate[dise] = float(tmp_sum) / float(dise_sym_sum)

            sorted_dise = list(dict(sorted(dise_sym_rate.items(), key=lambda d: d[1], reverse=True)).keys())
            left_set = []
            for i in range(len(sorted_dise)):
                max_dise = sorted_dise[i]
                left_set = [v for v in self.req_dise_sym_dict[max_dise] if
                            v not in current_slots['inform_slots'].keys()]
                if len(left_set) > 0: break
            # if syms in request set of all disease have been asked, choose one sym in request set
            if len(left_set) == 0:
                left_set = [v for v in self.request_set if v not in current_slots['inform_slots'].keys()]
            slot = random.choice(left_set)
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}

        # if sym in request set has been asked but no disease detect all its syms
        # choose the one with most rate
        elif self.phase == 0:
            dise = self.disease_from_dict_rate(current_slots)
            if dise != "":
                act_slot_response['diaact'] = "inform"
                act_slot_response['inform_slots'] = {'disease': dise, 'taskcomplete': "PLACEHOLDER"}
                act_slot_response['request_slots'] = {}
            else:
                act_slot_response = {'diaact': "inform",
                                     'inform_slots': {'disease': 'UNK', 'taskcomplete': "PLACEHOLDER"},
                                     'request_slots': {}}
            self.phase += 1
        # last thanks
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}

        return self.action_index(act_slot_response)

    def action_index(self, act_slot_response):
        """ Return the index of action """

        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print(act_slot_response)
        raise Exception("action index not found")
        return None

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        """ Register feedback from the environment, to be stored as future training data """

        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)
        # only record experience of dqn train, and warm start
        if self.predict_mode == False:  # Training Mode
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
        else:  # Prediction Mode
            self.experience_replay_pool.append(training_example)

    def train(self, batch_size=1, num_batches=100):
        """ Train DQN with experience replay """

        for iter_batch in range(num_batches):
            self.cur_bellman_err = 0
            for iter in range(int(len(self.experience_replay_pool) / (batch_size))):
                batch = [random.choice(self.experience_replay_pool) for i in range(batch_size)]
                batch_struct = self.dqn.singleBatch(batch, {'gamma': self.gamma, 'activation_func': 'relu'},
                                                    self.clone_dqn)
                self.cur_bellman_err += batch_struct['cost']['total_cost']

            print("cur bellman err %.4f, experience replay pool %s" % (
            float(self.cur_bellman_err) / len(self.experience_replay_pool), len(self.experience_replay_pool)))

    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """

        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print('saved model in %s' % (path,))
        except Exception as e:
            print('Error: Writing model fails: %s' % (path,))
            print(e)

    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""

        self.experience_replay_pool = pickle.load(open(path, 'rb'))

    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """

        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']

        # print("trained DQN Parameters:", json.dumps(trained_file[], indent=2))
        return model

