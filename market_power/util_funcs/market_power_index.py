import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import lilliefors



def residual_supply_index(outputs_path:str,
                          market:str="EOM",
                          deduct_renewables:bool=True,
                          quantity:str="volume",
                          lower_bound:float=0.0,
                          upper_bound:float=2.0):
    
    """This function reads the demand and supply from input files 
    of a scenario to compute the Residual Supply Index (RSI). 
    
    Formula:
    
    RSI[o,t] = (Tot Supply[t] - Supply[o,t]) / Load[t]

    for operator o at time t.

    Inputs:

        outputs_path(str): path to scenario outputs
        market(str): market ID 
        deduct_renewables(bool): if True, remove RES generation from load
        quantity(str): if "accepted_volume", uses accepted bids only, else use all bids
        lower_bound(float): Lower bound for clipping. Default: 0.0
        upper_bound(float): Upper bound for clipping. Default: 2.0

    ----------------------------------------------
    Notes:

    A lower RSI implies a higher degree of market power.
    RSI is unbounded and can be negative (e.g. 
    if renewable generation > load). 

    """

    assert quantity in ["volume", "accepted_volume"], "Quantity should be 'volume' or 'accepted_volume'"
    
    orders = pd.read_csv(f"{outputs_path}/market_orders.csv", index_col=0, parse_dates=["start_time", "end_time"])
    orders = orders[orders["market_id"] == market]

    demand_orders = orders[orders["volume"] < 0]
    demand = demand_orders.groupby("start_time")[quantity].sum()

    supply_orders = orders[orders["volume"] > 0]
    supply = supply_orders.pivot_table(index="start_time", 
                                       values=quantity, 
                                       columns="unit_id", 
                                       aggfunc="sum")

    units = pd.read_csv(f"{outputs_path}/power_plant_meta.csv", index_col=0)

    if deduct_renewables: 
        # move renewables to demand
        res = [c for c in supply.columns if c in 
                ["Biomass", "Hydro", "Solar", "Wind Onshore", "Wind Offshore"]
                ]
        demand += supply[res].sum(axis=1)
        supply = supply.drop(columns=res, errors='ignore')

    rsi = supply.T.rename(index=units["unit_operator"])
    rsi = rsi.groupby(level=0).sum(min_count=1)
    rsi = (rsi.sum() - rsi) / (-1 * demand)

    rsi = rsi.T.clip(lower_bound, upper_bound)
    return rsi

def lerner_index(outputs_path:str,
                 market:str="EOM"):
    

    """This function compute the Lerner Index of bidding
    generation units for a given output scenario.  
    
    Formula:
    
    LI[u,t] = (MarketPrice[t] - MarginalCost[u,t]) / MarketPrice[t]

    for unit u at time t.

    Inputs:

        outputs_path(str): path to scenario outputs
        market(str): market ID 
    ----------------------------------------------
    Note:
    Lerner Index is only defined for the price-setting unit!
    A higher LI implies a higher degree of market power. 

    """

    orders = pd.read_csv(f"{outputs_path}/market_orders.csv", parse_dates=["start_time", "end_time"])
    dispatch = pd.read_csv(f"{outputs_path}/unit_dispatch.csv", parse_dates=["time"])
    powerplants = pd.read_csv(f"{outputs_path}/power_plant_meta.csv", index_col=0)

    # dispatch = dispatch[(dispatch["power"].notna()) & (dispatch["power"] > 0)]
    dispatch = dispatch.rename(columns={"time":"start_time", "unit":"unit_id"})
    dispatch["marginal_cost"] = dispatch["energy_generation_costs"] / dispatch["power"]

    # Lerner Index is only defined for the price-setting unit
    orders = orders[orders["market_id"] == market]
    orders = orders[orders["accepted_price"]  == orders["price"]]
    orders = orders[orders["accepted_volume"] > 0]
    
    orders = orders.merge(dispatch, on=["start_time", "unit_id"], how="left")
    orders["lerner_index"] = (orders["price"] - orders["marginal_cost"]) / orders["price"]
    orders["unit_operator"] = orders["unit_id"].map(powerplants["unit_operator"])
    li = orders.pivot_table(index="start_time", 
                            values="lerner_index", 
                            columns="unit_operator", 
                            # if > 1 marginal units, keeps the one with highest LI
                            aggfunc="max") 
    for op_id in powerplants["unit_operator"].unique():
        if op_id not in li:
            li[op_id] = None
        
    return li


def output_gap(outputs_path:str, market:str="EOM"):
    """This function compute the Output Gap of a unit operator
    for a given output scenario.  
    
    Formula:
    
    OG[o,t] = (TotCompetitiveGeneration[o,t] - RealizedGeneration[o,t]) / InstalledCapacity[o]

    for operator o at time t.

    Inputs:

        outputs_path(str): path to scenario outputs
        market(str): market ID 
    ----------------------------------------------
    Note:
    Lerner Index is only defined for the price-setting unit!
    A higher LI implies a higher degree of market power. 

    """

    orders = pd.read_csv(f"{outputs_path}/market_orders.csv", parse_dates=["start_time", "end_time"])
    dispatch = pd.read_csv(f"{outputs_path}/unit_dispatch.csv", parse_dates=["time"])
    powerplants = pd.read_csv(f"{outputs_path}/power_plant_meta.csv", index_col=0)
    installed_capacity = powerplants.groupby("unit_operator")["max_power"].sum()

    dispatch = dispatch[dispatch["power"] > 0]
    dispatch = dispatch.rename(columns={"time":"start_time", "unit":"unit_id"})
    dispatch["marginal_cost"] = dispatch["energy_generation_costs"] / dispatch["power"]

    # Lerner Index is only defined for the price-setting unit
    orders = orders.merge(dispatch, on=["start_time", "unit_id"], how="left")
    orders["unit_operator"] = orders["unit_id"].map(powerplants["unit_operator"])
    
    output_gap = lambda x: (x["volume"] - x["accepted_volume"]) if x["marginal_cost"] <= x["accepted_price"] else 0
    orders["output_gap"] = orders.apply(output_gap, axis=1)

    gap = orders.groupby(["start_time", "unit_operator"])["output_gap"].sum()
    gap = gap.unstack() / installed_capacity
        
    return gap



def marginal_share(outputs_path:str,
                   market:str="EOM"):
    
    """Returns the share of the hours in the simulations in which the operator is price-setting.
    """

    orders = pd.read_csv(f"{outputs_path}/market_orders.csv", index_col=0, parse_dates=["start_time", "end_time"])
    orders = orders[orders["market_id"] == market]
    orders = orders[orders["accepted_volume"] > 0]
    orders = orders[orders["accepted_price"] == orders["price"]]

    powerplants = pd.read_csv(f"{outputs_path}/power_plant_meta.csv", index_col=0)
    orders["unit_operator"] = orders["unit_id"].map(powerplants["unit_operator"])

    ms = orders.groupby(["start_time", "unit_operator"]).size().unstack()
    ms = (ms > 0).mean()
    
    for op_id in powerplants["unit_operator"].unique():
        if op_id not in ms:
            ms[op_id] = None
            
    return ms






def nw_mean_test(profits_a, profits_b, max_lags=None):
    """
    One-sided test: mean(a - b) > 0 using Newey–West standard errors.
    The test can be used to assert whether the profits of one simulation
    are statistically significant higher than those of another one.
    """
    d = np.asarray(profits_a) - np.asarray(profits_b)
    stat, p_value = lilliefors(d, dist='norm')
    print(f"Lilliefors statistic: {stat}")
    print(f"p-value: {p_value}")
    X = np.ones(len(d))  # intercept-only regression
    model = sm.OLS(d, X)

    results = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max_lags}
    )

    mean_diff = float(results.params[0])
    t_stat = float(results.tvalues[0])
    p_value = float(results.pvalues[0] / 2)  # one-sided

    return mean_diff, t_stat, p_value


