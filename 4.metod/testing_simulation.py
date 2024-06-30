import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_01_GREEN = 0
PHASE_01_YELLOW = 1
PHASE_01_RED = 2

PHASE_02_GREEN = 0
PHASE_02_YELLOW = 1
PHASE_02_RED = 2

PHASE_03_GREEN = 0
PHASE_03_YELLOW = 1
PHASE_03_RED = 2


class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []


    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        old_action = -1 # dummy init

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait

            self._reward_episode.append(reward)

        #print("Total reward:", np.sum(self._reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time


    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length() 
            self._queue_length_episode.append(queue_length)


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2", "E3", "E4"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            car_type = traci.vehicle.getTypeID(car_id)
            if car_type == "bus":
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
                if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                    self._waiting_times[car_id] = wait_time * 2
                else:
                    if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                        del self._waiting_times[car_id]
            elif car_type == "taxi":
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                road_id = traci.vehicle.getRoadID(car_id)
                if road_id in incoming_roads:
                    self._waiting_times[car_id] = wait_time * 1
                else:
                    if car_id in self._waiting_times:
                        del self._waiting_times[car_id]
        total_waiting_time_for_reward = sum(self._waiting_times.values())
        return total_waiting_time_for_reward


    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        return np.argmax(self._Model.predict_one(state))


    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 0 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        if old_action == 0:
            traci.trafficlight.setPhase("tl_01", yellow_phase_code)
#            if action_number == 1:
#                traci.trafficlight.setPhase("tl_02", yellow_phase_code)
#            elif action_number == 2:
#                traci.trafficlight.setPhase("tl_03", yellow_phase_code)
        elif old_action == 1:
            traci.trafficlight.setPhase("tl_02", yellow_phase_code)
#            if action_number == 0:
#                traci.trafficlight.setPhase("tl_01", yellow_phase_code)
#            elif action_number == 2:
#                traci.trafficlight.setPhase("tl_03", yellow_phase_code)
        elif old_action == 2:
            traci.trafficlight.setPhase("tl_03", yellow_phase_code)
#            if action_number == 0:
#                traci.trafficlight.setPhase("tl_01", yellow_phase_code)
#            elif action_number == 1:
#                traci.trafficlight.setPhase("tl_02", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """


        if action_number == 0:
            traci.trafficlight.setPhase("tl_01", PHASE_01_GREEN)
            traci.trafficlight.setPhase("tl_02", PHASE_02_RED)
            traci.trafficlight.setPhase("tl_03", PHASE_02_RED)
        elif action_number == 1:
            traci.trafficlight.setPhase("tl_01", PHASE_01_RED)
            traci.trafficlight.setPhase("tl_02", PHASE_02_GREEN)
            traci.trafficlight.setPhase("tl_03", PHASE_03_RED)
        elif action_number == 2:
            traci.trafficlight.setPhase("tl_01", PHASE_01_RED)
            traci.trafficlight.setPhase("tl_02", PHASE_02_RED)
            traci.trafficlight.setPhase("tl_03", PHASE_03_GREEN)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_dogu = traci.edge.getLastStepHaltingNumber("E2")
        halt_kuzey = traci.edge.getLastStepHaltingNumber("E3")
        halt_batı = traci.edge.getLastStepHaltingNumber("E4")
        queue_length = halt_dogu + halt_kuzey + halt_batı
        return queue_length


    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
       
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            
            car_type = traci.vehicle.getTypeID(car_id)
            
            if (car_type == "bus") or (car_type == "taxi"):
                lane_pos = traci.vehicle.getLanePosition(car_id)
                road_id = traci.vehicle.getRoadID(car_id)
                lane_pos = 300 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 300 = max len of a road

                # distance in meters from the traffic light -> mapping into cells
                if lane_pos < 5:
                    lane_cell = 0
                elif lane_pos < 10:
                    lane_cell = 1
                elif lane_pos < 15:
                    lane_cell = 2
                elif lane_pos < 20:
                    lane_cell = 3
                elif lane_pos < 30:
                    lane_cell = 4
                elif lane_pos < 50:
                    lane_cell = 5
                elif lane_pos < 100:
                    lane_cell = 6
                elif lane_pos < 150:
                    lane_cell = 7
                elif lane_pos < 200:
                    lane_cell = 8
                elif lane_pos <= 300:
                    lane_cell = 9

                # finding the road where the car is located 
            
                if road_id == "E2":
                    road_group = 0
                elif road_id == "E3":
                    road_group = 1
                elif road_id == "E4":
                    road_group = 2
                else:
                    road_group = -1

                if road_group >= 1 and road_group <= 2:
                    car_position = int(str(road_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-29
                    valid_car = True
                elif road_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

                if valid_car:
                    #car_speed = float(traci.vehicle.getSpeed(car_id) / traci.vehicle.getMaxSpeed(car_id))
                    #a = str(car_speed)
                    #x = len(a.split(".")[0]) + 1
                    #y = 1
                    #z = x + y
                    if car_type == "bus":
                        state[car_position] = 2  # write the position of the car car_id in the state array in the form of "cell occupied"
                    elif car_type == "taxi":
                        state[car_position] = 1
                    #state[car_position] = a[0:z]
                    
#            else:
#                pass

        return state


    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode



