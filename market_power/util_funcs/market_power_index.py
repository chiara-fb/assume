

def residual_supply_index(bids,
                          market:str="EOM",
                          deduct_renewables:bool=True,
                          quantity:str="volume",
                          lower_bound:float=0.0,
                          upper_bound:float=None):
    
    """This function reads the demand and supply from input files 
    of a scenario to compute the Residual Supply Index (RSI). 
    
    Formula:
    
    RSI[o,t] = (Tot Supply[t] - Supply[o,t]) / Load[t]

    for operator o at time t.

    Inputs:

        bids(pd.DataFrame): df of market orders
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
    df = bids.copy()
    df = df[df["market_id"] == market]

    demand_orders = df[df["volume"] < 0]
    demand = demand_orders.groupby("datetime")[quantity].sum()

    supply_orders = df[df["volume"] > 0]
    supply = supply_orders.pivot_table(index="datetime", 
                                       values=quantity, 
                                       columns="unit_operator", 
                                       aggfunc="sum")

    if deduct_renewables: 
        # move renewables to demand
        demand += supply["renewables_operator"].fillna(0)
        supply = supply.drop(columns="renewables_operator", errors='ignore')

    rsi = supply.apply(lambda x: (x - supply.sum(axis=1)) / demand, axis=0)
    rsi = rsi.clip(lower_bound, upper_bound)
    return rsi

def lerner_index(bids,
                 market:str="EOM"):
    

    """This function compute the Lerner Index of bidding
    generation units for a given output scenario.  
    
    Formula:
    
    LI[u,t] = (MarketPrice[t] - MarginalCost[u,t]) / MarketPrice[t]

    for unit u at time t.

    ----------------------------------------------
    Note:
    Lerner Index is only defined for the price-setting unit!
    A higher LI implies a higher degree of market power. 

    """
    df = bids.copy()
    operators = df["unit_operator"].unique()

    # Lerner Index is only defined for the price-setting unit
    df = df[df["market_id"] == market]
    df = df[df["accepted_price"]  == df["price"]]
    df = df[df["accepted_volume"] > 0]
    
    df["lerner_index"] = (df["price"] - df["marginal_cost"]) / df["price"]
    li = df.pivot_table(index="datetime", 
                        values="lerner_index", 
                        columns="unit_operator", 
                        # if > 1 marginal units, keeps the one with highest LI
                        aggfunc="max") 
    for op_id in operators:
        if op_id not in li:
            li[op_id] = None
        
    return li


def output_gap(bids:str, market:str="EOM"):
    """This function compute the Output Gap of a unit operator
    for a given output scenario.  
    
    Formula:
    
    OG[o,t] = (TotCompetitiveGeneration[o,t] - RealizedGeneration[o,t]) / InstalledCapacity[o]

    for operator o at time t.

    """
    df = bids.copy()
    df = df[df["market_id"] == market]

    def fill_marginal_cost(bid_df):
        bid_df["marginal_cost"] = bid_df["marginal_cost"].where(bid_df["power"] > 0) 
        bid_df["marginal_cost"] = bid_df["marginal_cost"].bfill().ffill()
        return bid_df

    df = df.groupby("bid_id").apply(fill_marginal_cost, include_groups=False)
    output_gap = lambda x: ((x["volume"] - x["accepted_volume"]) 
                            if x["marginal_cost"] < x["accepted_price"] else 0)
    df["output_gap"] = df.apply(output_gap, axis=1)
    relative_gap = lambda x: x["output_gap"].sum() / x["max_power"].sum()
    gap = df.groupby(["datetime", "unit_operator"]).apply(relative_gap, include_groups=False)
    gap = gap.unstack() 

    return gap



def marginal_share(bids,
                   market:str="EOM"):
    
    """Returns the share of the hours in the simulations in which the operator is price-setting.
    """
    df = bids.copy()
    df = df[df["market_id"] == market]
    operators = df["unit_operator"].unique()
    df = df[df["accepted_volume"] > 0]
    df = df[df["accepted_price"] == df["price"]]

    ms = df.groupby(["datetime", "unit_operator"]).size().unstack()
    ms = (ms > 0).mean()
    
    for op_id in operators:
        if op_id not in ms:
            ms[op_id] = None
            
    return ms



