
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import sys


class FuzzyPilot:
        def __init__(self):
                self.simulation = self._build_fuzzy_system()
        def _build_fuzzy_system(self) -> ctrl.ControlSystemSimulation:

                # 1. Define Universes (Antecedents & Consequents)
                # Speed Error: -100 (Too Fast) to 100 (Too Slow) knots
                speed_err = ctrl.Antecedent(np.arange(-100, 101, 1), 'speed_error')
                
                # Altitude: 0 to 40,000 feet
                altitude = ctrl.Antecedent(np.arange(0, 40001, 500), 'altitude')
                
                # Throttle: -100% (Full Decel) to 100% (Full Accel)
                throttle = ctrl.Consequent(np.arange(-100, 101, 1), 'throttle')

                # 2. Define Membership Functions
                # Speed Error
                speed_err['negative_large'] = fuzz.trapmf(speed_err.universe, [-100, -100, -50, -10])
                speed_err['zero'] = fuzz.trimf(speed_err.universe, [-20, 0, 20])
                speed_err['positive_large'] = fuzz.trapmf(speed_err.universe, [10, 50, 100, 100])

                # Altitude
                altitude['low'] = fuzz.sigmf(altitude.universe, 10000, -0.0005) # S-shape falling
                altitude['high'] = fuzz.sigmf(altitude.universe, 20000, 0.0005) # S-shape rising

                # Throttle Output
                throttle['brake'] = fuzz.trimf(throttle.universe, [-100, -100, -10])
                throttle['maintain'] = fuzz.trimf(throttle.universe, [-20, 0, 20])
                throttle['boost'] = fuzz.trimf(throttle.universe, [10, 100, 100])

                # 3. Define Rules
                # Rule 1: If we are going too fast (negative error), brake.
                rule1 = ctrl.Rule(speed_err['negative_large'], throttle['brake'])
                
                # Rule 2: If we are going too slow (positive error), boost.
                rule2 = ctrl.Rule(speed_err['positive_large'], throttle['boost'])
                
                # Rule 3: If speed is good, just maintain.
                rule3 = ctrl.Rule(speed_err['zero'], throttle['maintain'])
                
                # Rule 4: If altitude is high (thin air) and speed is slow, apply EXTRA boost.
                rule4 = ctrl.Rule(altitude['high'] & speed_err['positive_large'], throttle['boost'])

                # 4. Compile System
                control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
                return ctrl.ControlSystemSimulation(control_system) 
                

