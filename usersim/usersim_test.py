from .usersim import UserSimulator
import argparse, json, random, copy

import dialog_config


class TestRuleSimulator(UserSimulator):
    """ A rule-based user simulator for testing dialog policy """

    def __init__(self, sym_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        """ Constructor shared by all user simulators """

        self.sym_dict = sym_dict # all symptoms
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set

        self.max_turn = params['max_turn']
        self.slot_err_probability = 0
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = 0

        self.simulator_run_mode = params['simulator_run_mode']
        self.simulator_act_level = params['simulator_act_level']

        self.data_split = params['data_split']
        self.hit = 0

        self.left_goal = copy.deepcopy(start_set[self.data_split])

    def initialize_episode(self):
        """ Initialize a new episode (dialog)
        state['history_slots']: keeps all the informed_slots
        state['rest_slots']: keep all the slots (which is still in the stack yet)
        """

        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0
        # self.state['hit_slots'] = 0
       
        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        # self.goal =  random.choice(self.start_set)
        self.goal = self._sample_goal()

        self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE

        """ Debug: build a fake goal mannually """
        # self.debug_falk_goal()

        # sample first action
        user_action = self.start_action()
        assert (self.episode_over != 1), ' but we just started'
        return user_action, self.goal

    def start_action(self):
        self.state['diaact'] = "request"
        self.state['request_slots']['disease'] = 'UNK'
        if len(self.goal['explicit_inform_slots']) > 0:
            for slot in self.goal['explicit_inform_slots']:
                if self.goal['explicit_inform_slots'][slot] == True:
                    self.state['inform_slots'][slot] = dialog_config.TRUE
                if self.goal['explicit_inform_slots'][slot] == False:
                    self.state['inform_slots'][slot] = dialog_config.FALSE
        start_action = {}
        start_action['diaact'] = self.state['diaact']
        start_action['inform_slots'] = self.state['inform_slots']
        start_action['request_slots'] = self.state['request_slots']
        start_action['turn'] = self.state['turn']
        return start_action

    def _sample_goal(self):
        """ sample a user goal  """

        sample_goal = random.choice(self.left_goal)
        self.left_goal.remove(sample_goal)
        return sample_goal

    def corrupt(self, user_action):
        """ Randomly corrupt an action with error probs (slot_err_probability and slot_err_mode) on Slot and Intent (intent_err_probability). """

        for slot in user_action['inform_slots'].keys():
            slot_err_prob_sample = random.random()
            if slot_err_prob_sample < self.slot_err_probability:  # add noise for slot level
                if self.slot_err_mode == 0:  # replace the slot_value only
                    choice = [dialog_config.TRUE, dialog_config.FALSE, dialog_config.NOT_SURE]
                    choice.remove(user_action['inform_slots'][slot])
                    user_action['inform_slots'][slot] = random.choice(choice)
                elif self.slot_err_mode == 1:  # combined
                    slot_err_random = random.random()
                    if slot_err_random <= 0.33:
                        choice = [dialog_config.TRUE, dialog_config.FALSE, dialog_config.NOT_SURE]
                        choice.remove(user_action['inform_slots'][slot])
                        user_action['inform_slots'][slot] = random.choice(choice)

                        #user_action['inform_slots'][slot] = not user_action['inform_slots'][slot]
                    elif slot_err_random > 0.33 and slot_err_random <= 0.66:
                        del user_action['inform_slots'][slot]
                        random_slot = random.choice(list(self.sym_dict.keys()))
                        user_action[random_slot] = random.choice([dialog_config.TRUE, dialog_config.FALSE, dialog_config.NOT_SURE])
                    else:
                        del user_action['inform_slots'][slot]
                elif self.slot_err_mode == 2:  # replace slot and its values
                    del user_action['inform_slots'][slot]
                    random_slot = random.choice((self.sym_dict.keys()))
                    user_action[random_slot] = random.choice([dialog_config.TRUE, dialog_config.FALSE, dialog_config.NOT_SURE])
                elif self.slot_err_mode == 3:  # delete the slot
                    del user_action['inform_slots'][slot]

        intent_err_sample = random.random()
        if intent_err_sample < self.intent_err_probability:  # add noise for intent level
            user_action['diaact'] = random.choice(list(self.act_set.keys()))

    def next(self, system_action):
        """ Generate next User Action based on last System Action """
        self.hit = 0
        self.state['turn'] += 2
        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        
        sys_act = system_action['diaact']
        
        if 0 < self.max_turn < self.state['turn']:
            self.dialog_status = dialog_config.FAILED_DIALOG
            self.episode_over = True
            self.state['diaact'] = "closing"
        else:
            self.state['history_slots'].update(self.state['inform_slots']) # add inform slot to history
            self.state['inform_slots'].clear()
            
            if sys_act == "inform":
                self.response_inform(system_action)
            #elif sys_act == "multiple_choice":
            #    self.response_multiple_choice(system_action)
            elif sys_act == "request":
                self.response_request(system_action)

            elif sys_act == "thanks":
                self.response_thanks(system_action)
            # elif sys_act == "confirm_answer":
            #     self.response_confirm_answer(system_action)
            #elif sys_act == "closing":
            #     self.episode_over = True
            #     self.state['diaact'] = "thanks"

        self.corrupt(self.state)

        response_action = {}
        response_action['diaact'] = self.state['diaact']
        response_action['inform_slots'] = self.state['inform_slots']
        response_action['request_slots'] = self.state['request_slots']
        response_action['turn'] = self.state['turn']

        # add NL to dia_act
        # self.add_nl_to_action(response_action)

        # if len(self.goal['implicit_inform_slots'].keys()) == 0:
        #     hit_rate = 0.0
        # else:
        #     hit_rate = float(self.state['hit_slots'])/len(self.goal['implicit_inform_slots'].keys())
        # print(self.hit)

        return response_action, self.episode_over, self.dialog_status, self.hit



    def response_thanks(self, system_action):
        """ Response for Thanks (System Action) """

        self.episode_over = True
        self.dialog_status = dialog_config.SUCCESS_DIALOG
        # fail if no diagnosis or wrong diagnosis
        if self.state['request_slots']['disease'] == 'UNK' or self.state['request_slots']['disease'] != self.goal['disease_tag']:
            self.dialog_status = dialog_config.FAILED_DIALOG
        self.state['diaact'] = "closing"
        

    def response_request(self, system_action):
        """ Response for Request (System Action) """
        if len(system_action['request_slots'].keys()) > 0:
            slot = list(system_action['request_slots'].keys())[0]
            # answer slot in the goal
            if slot in self.goal['implicit_inform_slots'].keys():
                self.hit = 1
                # self.state['hit_slots'] += 1
                if self.goal['implicit_inform_slots'][slot] == True:
                    self.state['diaact'] = "confirm"
                    self.state['inform_slots'][slot] = dialog_config.TRUE
                elif self.goal['implicit_inform_slots'][slot] == False:
                    self.state['diaact'] = "deny"
                    self.state['inform_slots'][slot] = dialog_config.FALSE
            else:
                self.state['diaact'] = "not_sure"
                self.state['inform_slots'][slot] = dialog_config.NOT_SURE
        #         self.state['inform_slots'][slot] = self.goal['implicit_inform_slots'][slot]
        #         self.state['diaact'] = "inform"
        #         if slot in self.state['rest_slots']:
        #             self.state['rest_slots'].remove(slot)
        #         if slot in self.state['request_slots'].keys():
        #             del self.state['request_slots'][slot]
        #             self.state[request_slots].clear()
        #     # slot in request slots and been answered before, not appear in current setting
        #     elif slot in self.goal['request_slots'].keys() and slot not in self.state['rest_slots'] and slot in \
        #             self.state['history_slots'].keys():  # the requested slot has been answered
        #         self.state['inform_slots'][slot] = self.state['history_slots'][slot]
        #         self.state['request_slots'].clear()
        #         self.state['diaact'] = "inform"
        #     # slot in request slots and not answered, not appear in current setting, still need modification
        #     elif slot in self.goal['request_slots'].keys() and slot in self.state['rest_slots']:
        #         self.state['diaact'] = "request"  # "confirm_question"
        #         self.state['request_slots'][slot] = "UNK"
        #         ########################################################################
        #         # Inform the rest of informable slots
        #         ########################################################################
        #         for info_slot in self.state['rest_slots']:
        #             if info_slot in self.goal['inform_slots'].keys():
        #                 self.state['inform_slots'][info_slot] = self.goal['implicit_inform_slots'][info_slot]
        #         for info_slot in self.state['inform_slots'].keys():
        #             if info_slot in self.state['rest_slots']:
        #                 self.state['rest_slots'].remove(info_slot)
        #     else:
        #         # all request slots filled and inform slots informed, this will not appear,
        #         if len(self.state['request_slots']) == 0 and len(self.state['rest_slots']) == 0: 
        #             self.state['diaact'] = "thanks"
        #         # slot not in informable slots and not in request slot, inform notsure
        #         else:
        #             self.state['diaact'] = "inform"
        #         self.state['inform_slots'][slot] = -2
        # else:  # this case should not appear
        #     if len(self.state['rest_slots']) > 0:
        #         random_slot = random.choice(self.state['rest_slots'])
        #         if random_slot in self.goal['inform_slots'].keys():
        #             self.state['inform_slots'][random_slot] = self.goal['inform_slots'][random_slot]
        #             self.state['rest_slots'].remove(random_slot)
        #             self.state['diaact'] = "inform"
        #         elif random_slot in self.goal['request_slots'].keys():
        #             self.state['request_slots'][random_slot] = self.goal['request_slots'][random_slot]
        #             self.state['diaact'] = "request"
    # response to diagnosis
    def response_inform(self, system_action):
        #self.state['diaact'] = "thanks"
        self.episode_over = True
        self.dialog_status = dialog_config.SUCCESS_DIALOG
        # fail if no diagnosis or wrong diagnosis
        self.state['request_slots']['disease'] = system_action['inform_slots']['disease']
        if self.state['request_slots']['disease'] == 'UNK' or self.state['request_slots']['disease'] != self.goal['disease_tag']:
            self.dialog_status = dialog_config.FAILED_DIALOG
        self.state['diaact'] = "thanks"

def main(params):
    user_sim = RuleSimulator()
    user_sim.initialize_episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    params = vars(args)

    print("User Simulator Parameters:")
    print(json.dumps(params, indent=2))

    main(params)

