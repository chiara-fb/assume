#TODO: here the manual research

from assume.world import World
from assume.units.powerplant import PowerPlant
from tqdm import tqdm
import numpy as np
import pyomo.environ as pyo
import copy
import numpy as np
import os



class BilevelSolution:
    """
    Finds the optimal bids for the units of a world operator as a function
    of the residual demand in a given market, by iterating over a given range 
    of possible bids. Bids are formulated as lambda*marginal_cost for each
    group with the same marginal cost.

    Input: 
        world(World): the simulation world
        operator_id(str): identifies the operator in the world
        market_id(str): identifies the operator in the world
        min_lambda(int): lower bound on multiplier for marginal cost (incl.)
        max_lambda(int): upper bound on multiplier for marginal cost (incl.)
    ----
    
    Note: 
    
    This class assumes that:    
        1) only powerplants are considered supply
        2) marginal costs remain constant over the simulation time
        3) strategic prices, non-strategic quantities
                    
    """
    def __init__(self, 
                 world: World, 
                 operator_id:str,
                 market_id:str, 
                 min_lambda:int=1,
                 max_lambda:int=3):

        self.supply = self.calculate_supply(world, operator_id, market_id)
        self.op_units = self.get_operator_units(world, operator_id, market_id)

        self.residual_loads = self.get_residual_loads(world, operator_id, market_id)
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.min_price = world.markets[market_id].minimum_bid_price
        self.max_price = world.markets[market_id].maximum_bid_price

    
    def get_residual_loads(self, 
                           world:World,
                           operator_id:str,
                           market_id:str) -> set:
        """
        Using the forecaster of one of the generation unit in the market, 
        obtains the set of residual loads which occur in the simulation.
        """
        op = world.unit_operators[operator_id]
        unit,*_ = [u for u in op.units.values() 
                   if isinstance(u, PowerPlant)
                   if market_id in u.bidding_strategies]

        res_loads = unit.forecaster.residual_load[market_id]
        res_loads = set(res_loads)
        return res_loads
    

    def get_operator_units(self, 
                           world:World,
                           operator_id:str,
                           market_id:str) -> list:
        """
        Returns all the generation unit of operator_id in market_id.
        """

        op = world.unit_operators[operator_id]
        op_units = [u_id for u_id, u in op.units.items() 
                     if isinstance(u, PowerPlant)
                     if market_id in u.bidding_strategies]
        
        return op_units

    
    def calculate_supply(self, 
                         world:World, 
                         operator_id:str,
                         market_id:str) -> dict[tuple]:
        """
        Calculates the total supply of the operator and the residual
        supply in the market, considering only dispatchable power plants. 

        Returns a dictionary of tuples such that:
        {unit0: (mc0, q0), unit1: (mc1, q1), ....] 
        with mc0 < mc1.
        """
        
        supply = {}
        
        for op_id, operator in world.unit_operators.items():
            for unit_id, unit in operator.units.items():
                
                if not isinstance(unit, PowerPlant):
                    continue

                if unit.technology in ["solar", "hydro", "biomass", "wind_offshore", "wind_offshore"]:
                    continue
                
                if unit.bidding_strategies.get(market_id, "False"):
                    mc = unit.calculate_marginal_cost(world.start, 0)
                    supply[unit_id] = mc, unit.max_power
                    
        return dict(sorted(supply.items(), key=lambda x: x[0]))


    def create_model(self,
                     res_load:float,
                     tie_break_rule:bool=False,
                     precision:float=0.0) -> pyo.ConcreteModel:
        """
        For each residual load level in self.residual_loads, 
        iterates over all possible combinations of lambdas 
        with a user-defined precision and for residual load level
        returns one (of the possible many) set of lambda which 
        obtain the highest profits. 
        
        """
        
        m = pyo.ConcreteModel()     
        
        ### Sets (all generators, stratgic generators) ###
        m.allUnits = pyo.Set(initialize=list(self.supply.keys()))
        m.opUnits = pyo.Set(initialize=self.op_units)
        
        ### Parameters (marginal costs, max generation, residual load)
        m.Costs = pyo.Param(m.allUnits, initialize={k:v[0] for k,v in self.supply.items()})
        m.MaxPower = pyo.Param(m.allUnits, initialize={k:v[1] for k,v in self.supply.items()})
        m.ResLoad = pyo.Param(initialize=res_load)

        ### Upper level variables ###
        m.opBids = pyo.Var(
            m.opUnits,
            domain=pyo.Reals,        
            doc="markup on marginal cost"
        )

        m.bid_min = pyo.Constraint(m.opUnits, 
                                   rule=lambda m,j: m.opBids[j] >= self.min_lambda * m.Costs[j])
        
        m.bid_max = pyo.Constraint(m.opUnits, 
                                   rule=lambda m,j: m.opBids[j] <= self.max_lambda * m.Costs[j])
        

        ### Lower level variables and constraints ### 

        m.Price = pyo.Var(domain=pyo.Reals,
                bounds=(self.min_price, self.max_price))
        m.Power = pyo.Var(
            m.allUnits,
            domain=pyo.NonNegativeReals,
            doc="share of accepted generation for Units"
        )

        m.power_max = pyo.Constraint(m.allUnits, 
                                    rule=lambda m,i: m.Power[i] <= m.MaxPower[i])
        
        ## Duals of generation constraints must be non-negative for KKT
        m.Low = pyo.Var(m.allUnits, domain=pyo.NonNegativeReals)  
        m.High = pyo.Var(m.allUnits, domain=pyo.NonNegativeReals) 

        
        ## Upper-level objective ###
        def max_profits(m):
            return sum(
                    (m.Price - m.Costs[j]) * m.Power[j]
                for j in m.opUnits)
        
        m.obj = pyo.Objective(rule=max_profits, sense=pyo.maximize)   
        
        ## Demand constraint ###
        def demand_constraint(m):
            return sum(m.Power[i] for i in m.allUnits) == m.ResLoad - precision
        
        m.demand_eq = pyo.Constraint(rule=demand_constraint)



        ### KKT conditions (stationarity and complementarity)
        ## first-derivative of the lower-level optimization problem
        def stationarity(m, i):
            if tie_break_rule: 
                return ((m.opBids[i] + precision if i in m.opUnits else m.Costs[i]) -
                        m.Price - m.Low[i] + m.High[i]) == 0
            
            else:
                return ((m.opBids[i] if i in m.opUnits else m.Costs[i]) -
                        m.Price - m.Low[i] + m.High[i]) == 0

        m.stat_eq  = pyo.Constraint(m.allUnits, rule=stationarity)
        m.low_compl = pyo.Constraint(m.allUnits, rule=lambda m,i: m.Power[i] * m.Low[i] == 0)
        m.high_compl = pyo.Constraint(m.allUnits, rule=lambda m,i: (m.Power[i]-m.MaxPower[i]) * m.High[i] == 0) 

        return m


    def find_optimal_response(self, solver_name='gurobi', precision=1e-4, tie_break_rule=False, **solver_kwargs) -> dict:
        """
        Iterates over all the possible values for residual load and finds the optimal response
        in terms of bid prices. Note: bid[j] is none if j was not committed in the lower-level.
        """

        optimal_responses = {}
        solver = pyo.SolverFactory(solver_name)
        
        for res_load in tqdm(self.residual_loads): 
            model = self.create_model(res_load, tie_break_rule, precision)
            res = solver.solve(model, **solver_kwargs)
            try:
                lambdas = {j: model.opBids[j].value 
                           if model.Power[j].value > 0 else None 
                           for j in model.opUnits}
                optimal_responses[float(res_load)] = lambdas
            except Exception as e:
                print(f"Exception for ResLoad {res_load}: {e}")
        
        return optimal_responses
        
        

