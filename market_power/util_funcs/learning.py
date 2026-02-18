
from assume.world import World
from assume.scenario.loader_csv import load_scenario_folder, setup_world, run_learning
from assume.common.base import LearningConfig
from itertools import product
import optuna
import random
import numpy as np
import torch as th
import copy


def seed_everything(seed:int):
    """
    Helper function to make training
    and simulation replicable.

    Input:
        seed(int): the seed for all random generators.
    """

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    

class HyperparameterTuner:

    def __init__(self, 
                 world: World,
                 scenario: str,
                 db_pardir: str, 
                 trial_params:dict={},
                 seed:int=42):
        
        self.world = world
        self.scenario = scenario
        self.db_pardir = db_pardir
        self.optuna_seed = seed
        self.world_seed = seed
        self._init_world_args(world)
        self.trial_params = trial_params


    def _init_world_args(self, world: World):
        """Saves copies of world arguments for backup and resetting after trials.
        """
        self.save_path = world.learning_config.trained_policies_save_path
        self.load_path = world.learning_config.trained_policies_load_path
        self.scenario_data = world.scenario_data
        self.train_freq = world.learning_config.train_freq
        self.id = world.simulation_id
    
    
        
    def _update_world(self, trial:optuna.trial.Trial):
        """        
        Updates the simulation id and learning config in the scenario data
        based on the trial parameters. Updates the save and load paths
        for trained policies accordingly.
        """
        
        scenario_data = copy.deepcopy(self.scenario_data)
        trial_id = "-".join(f"{k}_{v}" for k, v in trial.params.items())
        scenario_data["simulation_id"] = f"{trial_id}" 
        scenario_data["config"]["learning_config"]["trained_policies_save_path"] = (
            f"{self.save_path}/{trial_id}")
        scenario_data["config"]["learning_config"]["trained_policies_load_path"] = (
            f"{self.save_path}/{trial_id}")
  
        
        for param in trial.params:

            if param in scenario_data["config"]["learning_config"]:
                scenario_data["config"]["learning_config"][param] = trial.params[param]
            
            elif param in scenario_data["config"]["bidding_strategy_params"]:
                scenario_data["config"]["bidding_strategy_params"][param] = trial.params[param]
            
            elif param == "seed":
                self.world_seed = int(trial.params[param])
            
            else:  
                raise Exception(f"Parameter {param} is not in 'learning_config' or 'bidding_strategy_params'.")

        self.world.scenario_data = scenario_data
        self.world.learning_config = LearningConfig(scenario_data["config"]["learning_config"])
        self.world.simulation_id = trial_id

        

    def evaluate_trial(self, trial:optuna.trial.Trial) -> float:
        """
        Evaluate the trial based on the profits in the final  
        simulation.
        
        Note: 
        expect the db name to be "{example}.db" and the 
        "simulation" column to contain entries with trial id.
        """

        self.world.run()
        df = read_market_orders(self.scenario,
                                self.db_pardir,
                                simulation_id=self.world.simulation_id)
        df = df.drop_duplicates()
        df = df[df["unit_operator"] == "Operator-RL"]
        profit = df.groupby("datetime")["profit"].sum()       
        print(f"Simulation has length {len(profit)}")

        return profit.sum()



    def objective(self, trial:optuna.trial.Trial) -> float:
        """
        "Objective function for the optuna trial.
        """

        for param, param_list in self.trial_params.items():
            trial.suggest_categorical(param, param_list)        

        self._update_world(trial)
        setup_world(self.world)
        seed_everything(self.world_seed)

        if self.world.learning_config.learning_mode:
            run_learning(self.world)
        
        profits = self.evaluate_trial(trial)
        print(f"Study {self.id} with profits: {profits:,.1f} terminated.")
        return profits


    def run_trials(self, n_trials:int) -> optuna.study.Study:
        """
        Create and run an optuna study for hyperparameter tuning
        using grid search.
        """


        study = optuna.create_study(
            study_name=self.id,
            direction="maximize", 
            sampler=optuna.samplers.GridSampler(search_space=self.trial_params, seed=self.optuna_seed)
        )

        study.optimize(self.objective, n_trials=n_trials)
        return study
    

if __name__ == "__main__":
    from db_read import *
    from market_power_index import *
    example = "base_op" # germany_op
    pardir = "sqlite:///temp_db"
    db_uri = f"{pardir}/{example}.db"
    world = World(database_uri=db_uri)
    
    load_scenario_folder(
    world,
    inputs_path="market_power/inputs",
    scenario=example,
    study_case=False,
    )
    np.random.seed(42)
    seeds = np.random.choice(range(1000), size=100, replace=False)
    trial_params = {"seed": seeds.tolist()}
    #                 "gradient_steps":[1,10,100]}
    # # trial_params = {"seed": [7,21,42,2002,999],
    # #                 "nbins":[4,8,12,16,20,24,28,32,36]}
    hypertuner = HyperparameterTuner(world, 
                                     example, 
                                     pardir, 
                                     trial_params=trial_params)
    
    study = hypertuner.run_trials(n_trials=20)
    summary = study.trials_dataframe()
    summary.to_csv(f"trials_{example}.csv", index=False)

    all_runs = pd.DataFrame()
    combinations = [
    "-".join(f"{k}_{v}" for k, v in zip(list(trial_params), combo))
    for combo in product(*(trial_params[k] for k in trial_params))]
    
    for combo in combinations:
        #df = read_market_orders(example, pardir, simulation_id=f"seed_{combo}")
        df = read_market_orders(example, pardir, simulation_id=combo)
        pf = df.groupby(["datetime", "unit_operator"])["profit"].sum().unstack()["Operator-RL"]
        gen = df.groupby(["datetime", "unit_operator"])["accepted_volume"].sum().unstack()["Operator-RL"]
        all_runs.loc[combo, "profit (Mn)"] = pf.sum() / 10**6
        all_runs.loc[combo, "gen (TWh)"] = gen.sum() / 10**3
        all_runs.loc[combo, "RSI"] = residual_supply_index(df)["Operator-RL"].corr(pf)
        all_runs.loc[combo, "MI"] = marginal_share(df)["Operator-RL"]
        all_runs.loc[combo, "LI"] = lerner_index(df)["Operator-RL"].mean()
        all_runs.loc[combo, "OG"] = output_gap(df)["Operator-RL"].mean()
    
    all_runs.to_csv(f"all_runs_{example}.csv")

    
    


