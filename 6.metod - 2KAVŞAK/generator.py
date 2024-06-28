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
        with open("intersection/bursa.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" vClass= "passenger" id="standart_car" length="4.0" minGap="2.5" maxSpeed="13" sigma="0.5" color= "cyan" />
            <vType accel="1.0" decel="4.5" vClass= "bus" id="bus" length="8.0" minGap="2.5" maxSpeed="8" sigma="0.5" color= "red" />
            <vType accel="1.0" decel="4.5" vClass= "taxi" id="taxi" length="4.0" minGap="2.5" maxSpeed="13" sigma="0.5" color= "yellow" />
            

            
            <route id="sk1" edges="E20 E2 E19"/>
            <route id="sk2" edges="E20 E2 E32 E33 E13 E15 E16 E5 E35 E8"/>
            
            <route id="sd1" edges="E14 E7 E37 E11 E12 E1 E31 E19"/>
            <route id="sd2" edges="E14 E7 E37 E11 E12 E1 E21"/> 
            
            <route id="sb1" edges="E18 E3 E33 E13 E15 E16 E5 E35 E8"/>
            <route id="sb2" edges="E18 E3 E33 E30 E21"/>
            
            <route id="gk1" edges="E10 E4 E34 E9"/>
            <route id="gk2" edges="E10 E4 E11 E12 E1 E21"/>
            <route id="gk3" edges="E10 E4 E34 E35 E8"/>
            
            <route id="gb1" edges="E18 E3 E33 E13 E15 E16 E5 E35 E8"/>
            <route id="gb2" edges="E20 E2 E32 E33 E13 E15 E16 E5 E9"/>
            <route id="gb3" edges="E20 E2 E32 E33 E13 E15 E16 E5 E35 E36 E0"/>
            
            <route id="gg1" edges="E17 E6 E8"/>
            <route id="gg2" edges="E17 E6 E36 E0"/>
            <route id="gg3" edges="E17 E6 E36 E37 E11 E12 E1 E31 E19"/>
            
            <route id="gd1" edges="E14 E7 E37 E11 E12 E1 E21"/>          
            <route id="gd2" edges="E14 E7 E37 E34 E9"/>
            <route id="gd3" edges="E14 E7 E0"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                personal_or_public = np.random.uniform()
                if personal_or_public < 0.75:  # choose car type: personal or public - 75% of times personal car
                    sehrekustu_or_gokdere = np.random.uniform()
                    if sehrekustu_or_gokdere < 0.50:
                        sehrekustu_direction = np.random.uniform()
                        if sehrekustu_direction < 0.31:
                            sk_rota = np.random.uniform()
                            if sk_rota < 0.70:
                                print('    <vehicle id="car_sk1_%i" type="standart_car" route="sk1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            else:
                                print('    <vehicle id="car_sk2_%i" type="standart_car" route="sk2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                        elif sehrekustu_direction > 0.59:
                            sb_rota = np.random.uniform()
                            if sb_rota < 0.70:
                                print('    <vehicle id="car_sb1_%i" type="standart_car" route="sb1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            else:
                                print('    <vehicle id="car_sb2_%i" type="standart_car" route="sb2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                        else:
                            sd_rota = np.random.uniform()
                            if sd_rota < 0.70:
                                print('    <vehicle id="car_sd1_%i" type="standart_car" route="sd1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            else:
                                print('    <vehicle id="car_sd2_%i" type="standart_car" route="sd2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                    else:
                        gokdere_direction = np.random.uniform()
                        if gokdere_direction < 0.25:
                            gk_rota = np.random.uniform()
                            if gk_rota < 0.10:
                                print('    <vehicle id="car_gk3_%i" type="standart_car" route="gk3" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            elif gk_rota > 0.50:
                                print('    <vehicle id="car_gk1_%i" type="standart_car" route="gk1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            else:
                                print('    <vehicle id="car_gk2_%i" type="standart_car" route="gk2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                        elif gokdere_direction >= 0.25 and gokdere_direction < 0.50:
                            gb_rota = np.random.uniform()
                            if gb_rota < 0.25:
                                print('    <vehicle id="car_gb3_%i" type="standart_car" route="gb3" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            elif gb_rota > 0.75:
                                print('    <vehicle id="car_gb2_%i" type="standart_car" route="gb2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            else:
                                print('    <vehicle id="car_gb1_%i" type="standart_car" route="gb1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                        elif gokdere_direction >= 0.50 and gokdere_direction < 0.75:
                            gg_rota = np.random.uniform()
                            if gg_rota < 0.50:
                                print('    <vehicle id="car_gg1_%i" type="standart_car" route="gg1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            elif gg_rota > 0.80:
                                print('    <vehicle id="car_gg3_%i" type="standart_car" route="gg3" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            else:
                                print('    <vehicle id="car_gg2_%i" type="standart_car" route="gg2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                        else:
                            gd_rota = np.random.uniform()
                            if gd_rota < 0.25:
                                print('    <vehicle id="car_gd3_%i" type="standart_car" route="gd3" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            elif gd_rota > 0.75:
                                print('    <vehicle id="car_gd2_%i" type="standart_car" route="gd2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            else:
                                print('    <vehicle id="car_gd1_%i" type="standart_car" route="gd1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                else:  # -25% of the time public vehicles
                    bus_or_taxi = np.random.uniform()
                    if bus_or_taxi < 0.75:   # -75% of times bus
                        sehrekustu_or_gokdere = np.random.uniform()
                        if sehrekustu_or_gokdere < 0.50:
                            sehrekustu_direction = np.random.uniform()
                            if sehrekustu_direction < 0.31:
                                sk_rota = np.random.uniform()
                                if sk_rota < 0.70:
                                    print('    <vehicle id="bus_sk1_%i" type="bus" route="sk1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="bus_sk2_%i" type="bus" route="sk2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            elif sehrekustu_direction > 0.59:
                                sb_rota = np.random.uniform()
                                if sb_rota < 0.70:
                                    print('    <vehicle id="bus_sb1_%i" type="bus" route="sb1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="bus_sb2_%i" type="bus" route="sb2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            else:
                                sd_rota = np.random.uniform()
                                if sd_rota < 0.70:
                                    print('    <vehicle id="bus_sd1_%i" type="bus" route="sd1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="bus_sd2_%i" type="bus" route="sd2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                        else:
                            gokdere_direction = np.random.uniform()
                            if gokdere_direction < 0.25:
                                gk_rota = np.random.uniform()
                                if gk_rota < 0.10:
                                    print('    <vehicle id="bus_gk3_%i" type="bus" route="gk3" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                elif gk_rota > 0.50:
                                    print('    <vehicle id="bus_gk1_%i" type="bus" route="gk1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="bus_gk2_%i" type="bus" route="gk2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            elif gokdere_direction >= 0.25 and gokdere_direction < 0.50:
                                gb_rota = np.random.uniform()
                                if gb_rota < 0.25:
                                    print('    <vehicle id="bus_gb3_%i" type="bus" route="gb3" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                elif gb_rota > 0.75:
                                    print('    <vehicle id="bus_gb2_%i" type="bus" route="gb2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="bus_gb1_%i" type="bus" route="gb1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            elif gokdere_direction >= 0.50 and gokdere_direction < 0.75:
                                gg_rota = np.random.uniform()
                                if gg_rota < 0.50:
                                    print('    <vehicle id="bus_gg1_%i" type="bus" route="gg1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                elif gg_rota > 0.80:
                                    print('    <vehicle id="bus_gg3_%i" type="bus" route="gg3" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="bus_gg2_%i" type="bus" route="gg2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            else:
                                gd_rota = np.random.uniform()
                                if gd_rota < 0.25:
                                    print('    <vehicle id="bus_gd3_%i" type="bus" route="gd3" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                elif gd_rota > 0.75:
                                    print('    <vehicle id="bus_gd2_%i" type="bus" route="gd2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="bus_gd1_%i" type="bus" route="gd1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)       
                    else:   # -25% of times taxi
                        sehrekustu_or_gokdere = np.random.uniform()
                        if sehrekustu_or_gokdere < 0.50:
                            sehrekustu_direction = np.random.uniform()
                            if sehrekustu_direction < 0.31:
                                sk_rota = np.random.uniform()
                                if sk_rota < 0.70:
                                    print('    <vehicle id="taxi_sk1_%i" type="taxi" route="sk1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="taxi_sk2_%i" type="taxi" route="sk2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            elif sehrekustu_direction > 0.59:
                                sb_rota = np.random.uniform()
                                if sb_rota < 0.70:
                                    print('    <vehicle id="taxi_sb1_%i" type="taxi" route="sb1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="taxi_sb2_%i" type="taxi" route="sb2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            else:
                                sd_rota = np.random.uniform()
                                if sd_rota < 0.70:
                                    print('    <vehicle id="taxi_sd1_%i" type="taxi" route="sd1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="taxi_sd2_%i" type="taxi" route="sd2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                        else:
                            gokdere_direction = np.random.uniform()
                            if gokdere_direction < 0.25:
                                gk_rota = np.random.uniform()
                                if gk_rota < 0.10:
                                    print('    <vehicle id="taxi_gk3_%i" type="taxi" route="gk3" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                elif gk_rota > 0.50:
                                    print('    <vehicle id="taxi_gk1_%i" type="taxi" route="gk1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="taxi_gk2_%i" type="taxi" route="gk2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            elif gokdere_direction >= 0.25 and gokdere_direction < 0.50:
                                gb_rota = np.random.uniform()
                                if gb_rota < 0.25:
                                    print('    <vehicle id="taxi_gb3_%i" type="taxi" route="gb3" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                elif gb_rota > 0.75:
                                    print('    <vehicle id="taxi_gb2_%i" type="taxi" route="gb2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="taxi_gb1_%i" type="taxi" route="gb1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            elif gokdere_direction >= 0.50 and gokdere_direction < 0.75:
                                gg_rota = np.random.uniform()
                                if gg_rota < 0.50:
                                    print('    <vehicle id="taxi_gg1_%i" type="taxi" route="gg1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                elif gg_rota > 0.80:
                                    print('    <vehicle id="taxi_gg3_%i" type="taxi" route="gg3" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="taxi_gg2_%i" type="taxi" route="gg2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                            else:
                                gd_rota = np.random.uniform()
                                if gd_rota < 0.25:
                                    print('    <vehicle id="taxi_gd3_%i" type="taxi" route="gd3" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                elif gd_rota > 0.75:
                                    print('    <vehicle id="taxi_gd2_%i" type="taxi" route="gd2" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)
                                else:
                                    print('    <vehicle id="taxi_gd1_%i" type="taxi" route="gd1" depart="%s" departLane="random" departSpeed="8" />' % (car_counter, step), file=routes)

            print("</routes>", file=routes)
