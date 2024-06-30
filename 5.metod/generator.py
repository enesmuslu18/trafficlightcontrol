import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/sehrekustu.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" vClass= "passenger" id="standart_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" color= "cyan" />
            <vType accel="1.0" decel="4.5" vClass= "bus" id="bus" length="10.0" minGap="2.5" maxSpeed="15" sigma="0.5" color= "red" />
            <vType accel="1.0" decel="4.5" vClass= "taxi" id="taxi" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" color= "yellow" />
            

            <route id="D_B" edges="E2 1011411824 724597257 26430484#0 26430484#1 26430484#2"/>
            <route id="D_K" edges="E2 724597254#0 724597254#1 724597253#0 724597253#1"/>
            <route id="K_G" edges="E3 1011411822 724597252 1099844081#0 1099844081#1 1099844081#2"/>
            <route id="K_B" edges="E3 724597257 26430484#0 26430484#1 26430484#2"/>
            <route id="B_D" edges="E4 724597252 1099844081#0 1099844081#1 1099844081#2"/>
            <route id="B_K" edges="E4 724597252 1011411823 724597254#0 724597254#1 724597253#0 724597253#1"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                personal_or_public = np.random.uniform()
                if personal_or_public < 0.75:  # choose car type: personal or public - 75% of times personal car
                    personal_car = np.random.randint(1, 6)  
                    if personal_car == 1:
                        print('    <vehicle id="car_D_B_%i" type="standart_car" route="D_B" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                    elif personal_car == 2:
                        print('    <vehicle id="car_D_K_%i" type="standart_car" route="D_K" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                    elif personal_car == 3:
                        print('    <vehicle id="car_K_G_%i" type="standart_car" route="K_G" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                    elif personal_car == 4:
                        print('    <vehicle id="car_K_B_%i" type="standart_car" route="K_B" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                    elif personal_car == 5:
                        print('    <vehicle id="car_B_D_%i" type="standart_car" route="B_D" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                    elif personal_car == 6:
                        print('    <vehicle id="car_B_K_%i" type="standart_car" route="B_K" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                else:  # -25% of the time public vehicles
                    bus_or_taxi = np.random.uniform()
                    if bus_or_taxi < 0.75:   # -75% of times bus
                        public_bus = np.random.randint(1, 6)  
                        if public_bus == 1:
                            print('    <vehicle id="bus_D_B_%i" type="bus" route="D_B" depart="%s" departLane="random" departSpeed="5" />' % (car_counter, step), file=routes)
                        elif public_bus == 2:
                            print('    <vehicle id="bus_D_K_%i" type="bus" route="D_K" depart="%s" departLane="random" departSpeed="5" />' % (car_counter, step), file=routes)
                        elif public_bus == 3:
                            print('    <vehicle id="bus_K_G_%i" type="bus" route="K_G" depart="%s" departLane="random" departSpeed="5" />' % (car_counter, step), file=routes)
                        elif public_bus == 4:
                            print('    <vehicle id="bus_K_B_%i" type="bus" route="K_B" depart="%s" departLane="random" departSpeed="5" />' % (car_counter, step), file=routes)
                        elif public_bus == 5:
                            print('    <vehicle id="bus_B_D_%i" type="bus" route="B_D" depart="%s" departLane="random" departSpeed="5" />' % (car_counter, step), file=routes)
                        elif public_bus == 6:
                            print('    <vehicle id="bus_B_K_%i" type="bus" route="B_K" depart="%s" departLane="random" departSpeed="5" />' % (car_counter, step), file=routes)
                    else:   # -25% of times taxi
                        public_taxi = np.random.randint(1, 6)
                        if public_taxi == 1:
                            print('    <vehicle id="taxi_D_B_%i" type="taxi" route="D_B" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                        elif public_taxi == 2:
                            print('    <vehicle id="taxi_D_K_%i" type="taxi" route="D_K" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                        elif public_taxi == 3:
                            print('    <vehicle id="taxi_K_G_%i" type="taxi" route="K_G" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                        elif public_taxi == 4:
                            print('    <vehicle id="taxi_K_B_%i" type="taxi" route="K_B" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                        elif public_taxi == 5:
                            print('    <vehicle id="taxi_B_D_%i" type="taxi" route="B_D" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                        elif public_taxi == 6:
                            print('    <vehicle id="taxi_B_K_%i" type="taxi" route="B_K" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)

            print("</routes>", file=routes)
