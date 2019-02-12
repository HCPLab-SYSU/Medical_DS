import argparse, json, copy, os
import pickle as pickle
from utils import *
from dialog_system.dialog_manager import DialogManager
from agents.agent import AgentDQN
from usersim.usersim_rule import RuleSimulator
from usersim.usersim_test import TestRuleSimulator
from utils.utils import *

import dialog_config
from dialog_config import *

""" 
Launch a dialog simulation per the command line arguments
This function instantiates a user_simulator, an agent, and a dialog system.
Next, it triggers the simulator to run for the specified number of episodes.
"""
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', dest='data_folder', type=str, default='ad_data', help='folder to all data')
parser.add_argument('--max_turn', dest='max_turn', default=22, type=int, help='maximum ength of each dialog (default=20, 0=no maximum length)')
parser.add_argument('--episodes', dest='episodes', default=1, type=int, help='Total number of episodes to run (default=1)')
parser.add_argument('--slot_err_prob', dest='slot_err_prob', default=0.05, type=float, help='the slot err probability')
parser.add_argument('--slot_err_mode', dest='slot_err_mode', default=0, type=int, help='slot_err_mode: 0 for slot_val only; 1 for three errs')
parser.add_argument('--intent_err_prob', dest='intent_err_prob', default=0.05, type=float, help='the intent err probability')
parser.add_argument('--priority_replay', dest='priority_replay', default=0, type=int, help='')
parser.add_argument('--fix_buffer', dest='fix_buffer', default=0, type=int, help='')

parser.add_argument('--agt', dest='agt', default=0, type=int, help='Select an agent: 0 for a command line input, 1-6 for rule based agents')
parser.add_argument('--usr', dest='usr', default=0, type=int, help='Select a user simulator. 0 is a Frozen user simulator.')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.1, help='Epsilon to determine stochasticity of epsilon-greedy agent policies')

parser.add_argument('--origin_model', dest='origin_model', default=0, type=int, help='0 for not mask')

# load NLG & NLU model
parser.add_argument('--nlg_model_path', dest='nlg_model_path', type=str, default='./deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p', help='path to model file')
parser.add_argument('--nlu_model_path', dest='nlu_model_path', type=str, default='./deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p', help='path to the NLU model file')
parser.add_argument('--act_level', dest='act_level', type=int, default=0, help='0 for dia_act level; 1 for NL level')
parser.add_argument('--run_mode', dest='run_mode', type=int, default=0, help='run_mode: 0 for default NL; 1 for dia_act; 2 for both')
parser.add_argument('--auto_suggest', dest='auto_suggest', type=int, default=0, help='0 for no auto_suggest; 1 for auto_suggest')
parser.add_argument('--cmd_input_mode', dest='cmd_input_mode', type=int, default=0, help='run_mode: 0 for NL; 1 for dia_act')

# RL agent parameters
parser.add_argument('--experience_replay_pool_size', dest='experience_replay_pool_size', type=int, default=1000, help='the size for experience replay')
parser.add_argument('--dqn_hidden_size', dest='dqn_hidden_size', type=int, default=60, help='the hidden size for DQN')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='batch size')
parser.add_argument('--gamma', dest='gamma', type=float, default=0.9, help='gamma for DQN')
parser.add_argument('--predict_mode', dest='predict_mode', type=bool, default=False, help='predict model for DQN')
parser.add_argument('--simulation_epoch_size', dest='simulation_epoch_size', type=int, default=50, help='the size of validation set')
parser.add_argument('--warm_start', dest='warm_start', type=int, default=1, help='0: no warm start; 1: warm start for training')
parser.add_argument('--warm_start_epochs', dest='warm_start_epochs', type=int, default=100, help='the number of epochs for warm start')
parser.add_argument('--supervise', dest='supervise', type=int, default=1, help='0: no supervise; 1: supervise for training')
parser.add_argument('--supervise_epochs', dest='supervise_epochs', type=int, default=100, help='the number of epochs for supervise')

parser.add_argument('--trained_model_path', dest='trained_model_path', type=str, default=None, help='the path for trained model')
parser.add_argument('-o', '--write_model_dir', dest='write_model_dir', type=str, default='./deep_dialog/checkpoints/', help='write model to disk')
parser.add_argument('--save_check_point', dest='save_check_point', type=int, default=10, help='number of epochs for saving model')
parser.add_argument('--success_rate_threshold', dest='success_rate_threshold', type=float, default=0.3, help='the threshold for success rate')
parser.add_argument('--learning_phase', dest='learning_phase', default='train', type=str, help='train/test; default is all')
parser.add_argument('--train_set', dest='train_set', default='all', type=str, help='train/test/all; default is all')
parser.add_argument('--test_set', dest='test_set', default='all', type=str, help='train/test/all; default is all')
parser.add_argument('--predict_method', dest='predict_method', default=0, type=int, help='0 for not test all, 1 for test 5000 epoch')


args = parser.parse_args()
params = vars(args)

print('Dialog Parameters: ')
print(json.dumps(params, indent=2))

train_set = 'train'
test_set = 'test'
data_folder = params['data_folder']
fix_buffer = False
priority_replay = False
goal_set = load_pickle('{}/goal_dict_original.p'.format(data_folder))
act_set = text_to_dict('{}/dia_acts.txt'.format(data_folder))  # all acts
slot_set = text_to_dict('{}/slot_set.txt'.format(data_folder))
sym_dict = text_to_dict('{}/symptoms.txt'.format(data_folder))  # all symptoms
dise_dict = text_to_dict('{}/diseases.txt'.format(data_folder))
req_dise_sym_dict = load_pickle('{}/req_dise_sym_dict.p'.format(data_folder))
dise_sym_num_dict = load_pickle('{}/dise_sym_num_dict.p'.format(data_folder))
tran_mat = np.loadtxt('{}/action_mat.txt'.format(data_folder)) 
dise_sym_pro = np.loadtxt('{}/dise_sym_pro.txt'.format(data_folder))
sym_dise_pro = np.loadtxt('{}/sym_dise_pro.txt'.format(data_folder))
sp = np.loadtxt('{}/sym_prio.txt'.format(data_folder))

max_turn = params['max_turn']
num_episodes = params['episodes']

agt = params['agt']
usr = params['usr']

dialog_config.run_mode = params['run_mode']
dialog_config.auto_suggest = params['auto_suggest']

################################################################################
#   Parameters for Agents
################################################################################
agent_params = {}
agent_params['max_turn'] = max_turn
agent_params['epsilon'] = params['epsilon']
agent_params['agent_run_mode'] = params['run_mode']
agent_params['agent_act_level'] = params['act_level']
agent_params['experience_replay_pool_size'] = params['experience_replay_pool_size']
agent_params['dqn_hidden_size'] = params['dqn_hidden_size']
agent_params['batch_size'] = params['batch_size']
agent_params['gamma'] = params['gamma']
agent_params['predict_mode'] = params['predict_mode']
agent_params['trained_model_path'] = params['trained_model_path']
agent_params['warm_start'] = params['warm_start']
agent_params['supervise'] = params['supervise']
agent_params['fix_buffer'] = fix_buffer
agent_params['priority_replay'] = priority_replay
agent_params['origin_model'] = params['origin_model']

fix_buffer = False
if params['fix_buffer'] == 1:
    fix_buffer = True
priority_replay = False
if params['priority_replay'] == 1:
    priority_replay = True
print('fix_buffer', fix_buffer)
print('priority_replay', priority_replay)
agent_params['fix_buffer'] = fix_buffer
agent_params['priority_replay'] = priority_replay
agent = AgentDQN(sym_dict,dise_dict,req_dise_sym_dict,dise_sym_num_dict,tran_mat,dise_sym_pro,sym_dise_pro,sp,act_set,slot_set,agent_params)

################################################################################
#   Parameters for User Simulators
################################################################################
usersim_params = {}
usersim_params['max_turn'] = max_turn
usersim_params['slot_err_probability'] = params['slot_err_prob']
usersim_params['slot_err_mode'] = params['slot_err_mode']
usersim_params['intent_err_probability'] = params['intent_err_prob']
usersim_params['simulator_run_mode'] = params['run_mode']
usersim_params['simulator_act_level'] = params['act_level']
usersim_params['data_split'] = params['learning_phase']
test_user_sim = TestRuleSimulator(sym_dict, act_set, slot_set, goal_set, usersim_params)
user_sim = RuleSimulator(sym_dict, act_set, slot_set, goal_set, usersim_params)
################################################################################
# Dialog Manager
################################################################################
dm_params = {}
dialog_manager = DialogManager(agent, user_sim, act_set, slot_set, dm_params)
test_dialog_manager = DialogManager(agent, test_user_sim, act_set, slot_set, dm_params)

run_mode = params['run_mode']
output = False
if run_mode < 3: output = True


def test(test_size, dialog_manager,  output=False):
    print(output)
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    div_success = {}
    div_reward = {}
    div_turns = {}
    div_num = {}
    user_sim.data_split = test_set
    print("data split len " + str(len(user_sim.start_set[user_sim.data_split])))
    res = {}
    ave_hit_rate = 0.0
    agent.predict_mode = True
    # print(len(user_sim.start_set[user_sim.learning_phase]))
    for episode in range(test_size):
        dialog_manager.initialize_episode()
        dise = dialog_manager.goal['disease_tag']
        if dise not in div_success:
            div_success[dise] = 0
            div_num[dise] = 0
            div_reward[dise] = 0
            div_turns[dise] = 0
        episode_over = False
        while not episode_over:
            episode_over, reward, dialog_status, hit_rate = dialog_manager.next_turn()
            cumulative_reward += reward
            div_reward[dise] += reward
            if episode_over:
                ave_hit_rate += hit_rate
                if dialog_status == dialog_config.SUCCESS_DIALOG:
                    div_success[dise] += 1
                    successes += 1
                    if output: print("test simulation episode %s: Success" % episode)
                else:
                    if output: print("test simulation episode %s: Fail" % episode)
                cumulative_turns += dialog_manager.state_tracker.turn_count
                div_turns[dise] += dialog_manager.state_tracker.turn_count
                div_num[dise] += 1
    res['success_rate'] = float(successes) / test_size
    res['ave_reward'] = float(cumulative_reward) / test_size
    res['ave_turns'] = float(cumulative_turns) / test_size
    ave_hit_rate = ave_hit_rate / test_size
    # print(successes)
    print("Test hit rate %.4f, success rate %.4f, ave reward %.4f, ave turns %.4f" % (ave_hit_rate, res['success_rate'], res['ave_reward'], res['ave_turns']))
    for dise in div_num:
        print("disease: %s, number:%s, sucess_rate: %.4f,  ave reward %.4f, ave turns %.4f" % (dise,div_num[dise], div_success[dise]/div_num[dise], div_reward[dise]/div_num[dise], div_turns[dise]/div_num[dise]))
    return res


avg_success = 0.0
avg_reward = 0.0
avg_turn = 0.0
if params['predict_method'] == 1:
    test_num = 10
    for i in range(test_num):
        res = test(num_episodes, dialog_manager,  output=output)
        avg_success += res['success_rate']
        avg_reward += res['ave_reward']
        avg_turn += res['ave_turns']
    print('success rate: %3f, average reward: %3f, average turns: %3f' % (avg_success / test_num, avg_reward / test_num, avg_turn / test_num))
else:
    test(len(goal_set['test']), test_dialog_manager, output=output)
'''
test_num = 10
for i in range(test_num):
    res = test(num_episodes, output=output)
    avg_success += res['success_rate']
    avg_reward += res['ave_reward']
    avg_turn += res['ave_turns']
print('success rate: %3f, average reward: %3f, average turns: %3f' % (avg_success / test_num, avg_reward / test_num, avg_turn / test_num))
'''
#test(142, output=output)

