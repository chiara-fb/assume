from sqlalchemy import create_engine, inspect
import pandas as pd
import os

def calculate_startup_costs(df_by_unit:pd.DataFrame):
      # construct a boolean: TRUE if the unit was started in that time step, FALSE else
    is_on = lambda x: (x["accepted_volume"] > 0).any() * 1
    state = df_by_unit.groupby("datetime").apply(is_on, include_groups=False)
    # Count only start-up costs, not shutdowns
    startup = state.diff().clip(0,1) 
    # Divide the startup cost over all the bids for that time step
    size = df_by_unit.groupby("datetime").size()
    shared_startup = (startup / size).rename("startup_cost")
    # multiply the indicator for the startup cost of the unit
    df_by_unit = df_by_unit.join(shared_startup.fillna(0), on="datetime")
    df_by_unit["startup_cost"]*=df_by_unit["hot_start_cost"]

    return df_by_unit


def read_rl_params(example:str, pardir:str="sqlite:///local_db", study_case:str="base", simulation_id:str=None) -> pd.DataFrame: 
  """This function reads the profits
  from the RL-params table of the simulation 
  database for a given example.
  """

  # Connect to the simulation database
  try:
    engine = create_engine(f"{pardir}/{example}.db")

    if simulation_id is None:
      simulation_id = f"{example}_{study_case}"

    # Query rewards for specific simulation and unit
    sql = f"""
            SELECT *
            FROM rl_params
            WHERE simulation = '{simulation_id}' 
            ORDER BY datetime
            """

    reward_df = pd.read_sql(sql, engine)
    reward_df["datetime"] = pd.to_datetime(reward_df["datetime"])
    reward_df.rename(columns={"evaluation_mode": "eval"}, inplace=True)
    return reward_df
  
  except Exception as e: 
    print(f"Exception: {e}")

  


def read_market_orders(example:str, pardir:str="sqlite:///local_db", study_case:str="base", simulation_id:str=None, unit_operators:list=None) -> pd.DataFrame: 
  """This function reads the unit bids
  from the market_orders table of the simulation 
  database for a given example.
  """
  # Connect to the simulation database
  
  try:
    engine = create_engine(f"{pardir}/{example}.db")
    
    # Check if start-up costs exist in the simulation
    cols = inspect(engine).get_columns("power_plant_meta")
    col_names = [col['name'] for col in cols]
    startup_costs = False
    pp_query = "pp.unit_operator, pp.max_power, pp.technology, "

    if 'hot_start_cost' in col_names:
      pp_query+= "pp.hot_start_cost, "
      startup_costs = True
      

    if simulation_id is None:
      simulation_id = f"{example}_{study_case}"

    # Query rewards for specific simulation and unit
    sql = f"""
            SELECT mo.*, 
            {pp_query}
            ud.energy_generation_costs, ud.power
            FROM market_orders mo
            LEFT JOIN power_plant_meta pp
            ON mo.unit_id = pp."index"
            LEFT JOIN unit_dispatch ud
            ON mo.unit_id = ud.unit
            AND mo.start_time = ud.time
            WHERE mo.simulation = '{simulation_id}'
            AND ud.simulation = '{simulation_id}'
            ORDER BY mo.start_time;
            """
    df = pd.read_sql(sql, engine)
    if unit_operators is not None:
      df = df[df["unit_operator"].isin(unit_operators)]
    
    df = df.drop_duplicates()

    df = df.rename(columns={"start_time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(by=["unit_id", "datetime"])
    df["marginal_cost"] = (
                      df["energy_generation_costs"] /
                      df["power"].where(df["power"] > 0)
                      ).fillna(0)
    df["profit"] = ((df["accepted_price"] - df["marginal_cost"]) * 
                    df["accepted_volume"])
    
    # if they exist, add startup costs
    if startup_costs:
      group = df.groupby("unit_id", group_keys=True)
      df = group.apply(calculate_startup_costs, include_groups=False)
      df = df.reset_index("unit_id")
    
    df.set_index("datetime", inplace=True)
    print("Market orders read.")
    
    return df

  except Exception as e: 
    print(f"Exception: {e}")


def read_dispatch(example:str, pardir:str="sqlite:///local_db", study_case:str="base", simulation_id:str=None) -> pd.DataFrame: 
  """This function reads the unit generation
  from the market_dispatch table of the simulation 
  database for a given example.
  """

  # Connect to the simulation database
  try:
    engine = create_engine(f"{pardir}/{example}.db")

    if simulation_id is None:
      simulation_id = f"{example}_{study_case}"

    # Query rewards for specific simulation and unit
    sql = f"""
            SELECT ud.*, pp.unit_operator, pp.max_power, pp.technology
            FROM unit_dispatch ud
            JOIN power_plant_meta pp
            ON ud.unit = pp."index"
            WHERE ud.simulation = '{simulation_id}'
            ORDER BY ud.time;
            """
    gen_df = pd.read_sql(sql, engine)
    gen_df["datetime"] = pd.to_datetime(gen_df["time"])
    return gen_df
  
  except Exception as e: 
    print(f"Exception: {e}")
  
  



def read_market_price(example, pardir:str="sqlite:///local_db", study_case:str="base", simulation_id:str=None, inputs_path="../inputs"): 
  """This function reads the profits
  from the RL-params table of the simulation 
  database for a given example and simulation id.
  """

  # Connect to the simulation database
  try:
    engine = create_engine(f"{pardir}/{example}.db")

    if simulation_id is None:
      simulation_id = f"{example}_{study_case}"
    
    # Query rewards for specific simulation and unit
    sql = f"""
            SELECT price, product_start
            FROM market_meta
            WHERE simulation = '{simulation_id}'
            """
    price_df = pd.read_sql(sql, engine)
    price_df = price_df.rename(columns={"product_start": "datetime"})
    price_df["datetime"] = pd.to_datetime(price_df["datetime"])
    price_df.set_index("datetime", inplace=True)
    
    real_price_path = os.path.join(inputs_path, 
                                  f"{example}", 
                                  "DE_LU_day_ahead_prices_2019.csv")
    if os.path.exists(real_price_path):
      real_prices = pd.read_csv(real_price_path)
      real_prices["datetime"] = pd.to_datetime(real_prices["datetime"])
      real_prices.set_index("datetime", inplace=True)
      price_df = price_df.join(real_prices)
    
    return price_df

  except Exception as e: 
      print(f"Exception: {e}")
  