
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import sys


class FuzzyPilot:
        def __init__(self):
                self.simulation = self._build_fuzzy_system()
        def _build_fuzzy_system(self) -> ctrl.ControlSystemSimulation:

                #Define Universes (Antecedents & Consequents)
                # -100 (Too Fast) to 100 (Too Slow) knots
                speed_err = ctrl.Antecedent(np.arange(-100, 101, 1), 'speed_error')
                # 0 to 40,000 feet
                altitude = ctrl.Antecedent(np.arange(0, 40001, 500), 'altitude')
                # -100% (Decelaration) to 100% (Acceleration)
                throttle = ctrl.Consequent(np.arange(-100, 101, 1), 'throttle')

                # 2. Define Membership Functions
                # Speed Error 
                speed_err['negative_large'] = fuzz.trapmf(speed_err.universe, [-100, -100, -50, -10])
                speed_err['zero'] = fuzz.trimf(speed_err.universe, [-20, 0, 20])
                speed_err['positive_large'] = fuzz.trapmf(speed_err.universe, [10, 50, 100, 100])

                altitude['low'] = fuzz.sigmf(altitude.universe, 10000, -0.0005) # S-shape falling
                altitude['high'] = fuzz.sigmf(altitude.universe, 20000, 0.0005) # S-shape rising

                throttle['strong_brake'] = fuzz.trimf(throttle.universe, [-100,-100,-60,])
                throttle['brake'] = fuzz.trimf(throttle.universe, [-60, -30, 0])
                throttle['maintain'] = fuzz.trimf(throttle.universe, [-10, 0, 10])
                throttle['boost'] = fuzz.trimf(throttle.universe, [0, 30, 60])
                throttle['maxboost'] = fuzz.trimf(throttle.universe, [60 , 100, 100])


                # 3. Define Rules
                rule1 = ctrl.Rule(speed_err['negative_large'] & altitude['low'], throttle['brake'])
                rule2 = ctrl.Rule(speed_err['negative_large'] & altitude['high'], throttle['strong_brake'])
                rule3 = ctrl.Rule(speed_err['zero'], throttle['maintain'])
                rule4 = ctrl.Rule(speed_err['positive_large'] & altitude['low'], throttle['boost'])
                rule5 = ctrl.Rule(speed_err['positive_large'] & altitude['high'], throttle['maxboost'])

                # 4. Compile System
                control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
                return ctrl.ControlSystemSimulation(control_system) 

        def compute_throttle(self, current_speed_error: float, current_altitude: float) -> float:
                """
                Calculates the required throttle adjustment.
                
                Args:
                        current_speed_error (float): Target Speed - Current Speed
                        current_altitude (float): Current Altitude in feet
                
                Returns:
                        float: Recommended throttle adjustment percentage (-100 to 100)
                """
                try:
                        self.simulation.input['speed_error'] = np.clip(current_speed_error, -100, 100)
                        self.simulation.input['altitude'] = np.clip(current_altitude, 0, 40000)
                        self.simulation.compute()
                        return self.simulation.output['throttle']
                except Exception as e:
                        print(f"Error in fuzzy computation: {e}")
                        return 0.0
class KnapsackOptimizer:

    def __init__(self, weights: list, values: list, capacity: float, num_particles=30, iterations=100):
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.n_items = len(weights)
        self.n_particles = num_particles
        self.iterations = iterations
        
        # PSO Hyperparameters
        self.w = 0.7       # Inertia weight
        self.c1 = 1.4      # Cognitive (personal best) weight
        self.c2 = 1.4      # Social (global best) weight

    def _sigmoid(self, x):

        return 1 / (1 + np.exp(-x))

    def _fitness(self, binary_position):
        total_weight = np.sum(binary_position * self.weights)
        if total_weight > self.capacity:
            return 0 # Penalty: Invalid solution
        return np.sum(binary_position * self.values)

    def optimize(self):
        # Initialize particles (random 0s and 1s)
        particles = np.random.randint(2, size=(self.n_particles, self.n_items))
        
        # Initialize velocities
        velocities = np.random.uniform(-1, 1, size=(self.n_particles, self.n_items))

        # Track Personal Bests
        p_best_pos = particles.copy()
        p_best_scores = np.array([self._fitness(p) for p in particles])

        # Track Global Best
        g_best_index = np.argmax(p_best_scores)
        g_best_pos = p_best_pos[g_best_index].copy()
        g_best_score = p_best_scores[g_best_index]

        print(f"\n[SWARM] Initializing Swarm with {self.n_particles} particles...")

        # Optimization Loop
        for it in range(self.iterations):
            for i in range(self.n_particles):
                # 1. Update Velocity
                r1, r2 = np.random.rand(2)
                
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (p_best_pos[i] - particles[i]) +
                                 self.c2 * r2 * (g_best_pos - particles[i]))

                # 2. Update Position (Binary mapping using Sigmoid)
                # If sigmoid(velocity) > random, set to 1, else 0
                probs = self._sigmoid(velocities[i])
                particles[i] = (np.random.rand(self.n_items) < probs).astype(int)

                # 3. Evaluate Fitness
                current_score = self._fitness(particles[i])

                # 4. Update Personal Best
                if current_score > p_best_scores[i]:
                    p_best_scores[i] = current_score
                    p_best_pos[i] = particles[i].copy()

                    # 5. Update Global Best
                    if current_score > g_best_score:
                        g_best_score = current_score
                        g_best_pos = particles[i].copy()

            if it % 20 == 0:
                print(f"Iteration {it}/{self.iterations} | Current Best Value: {g_best_score}")

        return g_best_pos, g_best_score


# input validation
def get_valid_input(prompt, type_func):
    while True:
        try:
            return type_func(input(prompt))
        except ValueError:
            print(f"Invalid input. Please enter a valid {type_func.__name__}.")



####### MAIN ########
def main():
    print("    MACHINE LEARNING ASSIGNMENT: FUZZY & SWARM SYSTEM    ")
    print("1. Run Fuzzy Logic Airplane Controller")
    print("2. Run Swarm Intelligence Knapsack Solver")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ")

    if choice == '1':
        print("\n--- Part A: Fuzzy Controller ---")
        pilot = FuzzyPilot()
        
        try:
            # Taking user input for demonstration
            speed_input = get_valid_input("Enter Speed Error (-100 to 100): ", float)
            alt_input = get_valid_input("Enter Altitude (0 to 40000): ", float)
            
            result = pilot.compute_throttle(speed_input, alt_input)
            
            print("-" * 30)
            print(f"Inputs -> Speed Err: {speed_input} | Altitude: {alt_input}")
            print(f"Fuzzy Output -> Throttle Adjustment: {result:.2f}%")
            print("-" * 30)
            if result > 0:
                print("Action: PUSH THROTTLE")
            elif result < 0:
                print("Action: REDUCE THROTTLE")
            else:
                print("Action: MAINTAIN")

        except Exception as e:
            print(f"System Error: {e}")

    elif choice == '2':
        print("\n--- Part B: Knapsack Swarm Optimization ---")
        try:
            # Collecting Problem Data
            n = get_valid_input("How many items available? ", int)
            
            print(f"Enter {n} weights (space separated): ")
            weights = list(map(float, input().strip().split()))
            
            print(f"Enter {n} values (space separated): ")
            values = list(map(float, input().strip().split()))
            
            capacity = get_valid_input("Enter Knapsack Capacity: ", float)

            # Validation
            if len(weights) != n or len(values) != n:
                print("Error: The number of weights/values must match the number of items.")
                return

            # Execution
            optimizer = KnapsackOptimizer(weights, values, capacity)
            best_config, best_val = optimizer.optimize()

            print("\n" + "="*30)
            print(f"OPTIMIZATION COMPLETE")
            print("="*30)
            print(f"Best Configuration (Binary): {best_config}")
            print(f"Total Value: {best_val}")
            print(f"Total Weight: {np.sum(best_config * np.array(weights))}/{capacity}")
            
            # Show which items were picked
            print("\nItems Selected:")
            for i, selected in enumerate(best_config):
                if selected == 1:
                    print(f"- Item {i+1} (Weight: {weights[i]}, Value: {values[i]})")

        except Exception as e:
            print(f"Input Error: {e}")

    elif choice == '3':
        print("Exiting.")
        sys.exit()

if __name__ == "__main__":
    main()
