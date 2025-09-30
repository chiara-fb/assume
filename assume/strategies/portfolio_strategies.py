# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Order, Orderbook, Product
from assume.strategies.naive_strategies import NaiveSingleBidStrategy
from datetime import datetime, timedelta
import numpy as np

# this is for users that install ASSUME without learning strategy and related dependencies
try: 
    import torch as th
    from assume.strategies.learning_strategies import BaseLearningStrategy

except ImportError: 
    pass

# had to change type hint to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from assume.common.units_operator import UnitsOperator 


class BasePortfolioStrategy:
    """
    The base portfolio strategy

    Methods
    -------
    """

    def __init__(self, *args, **kwargs):
        pass

    def calculate_bids(
        self,
        operator: "UnitsOperator",
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a units operator and
        defines how the units managed by it should bid.

        This gives a lot of flexibility to the market bids.

        Args:
            operator (UnitsOperator): The operator handling the units.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """
        return []

    
    def calculate_tot_capacity(
            self,
            operator: "UnitsOperator", 
    ) -> dict[str,float]:
        """
        Computes the total capacity of the units owned by a unit operator by market.
        
        Args: 
            operator (UnitsOperator): The operator that bids on the market(s).
        Returns: 
            dict: a dictionary indexed by market_id.
        """

        tot_capacity = {}

        for unit in operator.units.values():
            for market_id in unit.bidding_strategies.keys():
                tot_capacity[market_id] = tot_capacity.get(market_id, 0) + unit.max_power

        return tot_capacity


class SimplePortfolioStrategy(BasePortfolioStrategy):
    """
    A naive strategy that bids the marginal cost of the unit on the market.

    Methods
    -------
    """

    def calculate_bids(
        self,
        operator: "UnitsOperator",
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
) -> Orderbook:
        # TODO this should be adjusted
        start = product_tuples[0][0]  # start time of the first product
        end_all = product_tuples[-1][1]  # end time of the last product

        for unit_id, unit in operator.units.items():
            pass

        bids = []
        return bids
    

class CournotStrategy(BasePortfolioStrategy): 
    """
    A Cournot strategy that adds a markup to the marginal cost of each unit of 
    the units operator. The marginal cost is computed with NaiveSingleBidStrategy, 
    and the markup depends on the total capacity of the unit operator.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.markup = kwargs.get("markup", 0.01)


    def calculate_bids(
        self,
        operator: "UnitsOperator", 
        market_config: MarketConfig,
        product_tuples: list[Product],
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            units_operator (UnitsOperator): The operator that bids on the market.
            market_config (MarketConfig): The configuration of the market.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """

        tot_capacity = self.calculate_tot_capacity(operator)[market_config.market_id]
        #TODO: divide by total available capacity in the market 

        operator_bids = Orderbook()
        ## Compute marginal costs ###
        for unit_id, unit in operator.units.items():
            bids = NaiveSingleBidStrategy().calculate_bids(
                    unit,
                    market_config,
                    product_tuples,
                )
            
            ### Apply Cournot mark-up ###
            for bid in bids:
                bid["price"] += self.markup * tot_capacity
                bid["unit_id"] = unit_id
                
                operator_bids.append(bid)

        return operator_bids


class PortfolioRLStrategy(BaseLearningStrategy, BasePortfolioStrategy):
    """
    Reinforcement Learning Strategy that enables the agent to learn optimal bidding strategies for
    the portfolio of a units operator on an Energy-Only Market.

    The agent submits a discrete price curve of price, quantity pairs for 6 technology types (OCGT, CCGT,
    lignite, hard coal, nuclear, oil) according to their available capacity in the portfolio.
    This strategy utilizes a set of 52 observations to generate actions, which are then transformed into 
    market bids. The observation space for this strategy consists of 36 global observations and
    and 3 unique values for each technology type (i.e. 18 in total) in the portfolio. 
    
    Observations include the following components:

    - **Forecasted Residual Load**: Forecasted load over the foresight period, scaled by the maximum
      demand, indicating anticipated grid conditions.
    - **Forecasted Price**: Price forecast over the foresight period, scaled by the maximum bid price,
      providing a sense of expected market prices.
      For each technology:
        - **Total min capacity, Total max capacity and Avg marginal cost**: The last 3x6 elements of the 
        observation vector, representing the unique state of the portfolio. Here, `total min capacity`  and
        `total max capacity` are scaled by the total installed capacity of the technology for the units
        operator, while `average marginal cost` is scaled by the maximum bid price. 

    Actions are formulated as 2 values (price, quantity) for each curve_split. Actions are rescaled from a range of 
    [-1, 1] to real bid prices in the `calculate_bids` method, then translate into unit-specific bids.

    Rewards are based on profit from transactions, minus operational and opportunity costs. Key components include:

    - **Profit**: Determined from the income generated by accepted bids, calculated as the product of
      accepted price, volume, and duration.
    - **Operational Costs**: Includes marginal costs and start-up costs when a unit transitions between
      on and off states.

    Attributes
    ----------
    foresight : int
        Number of time steps for which the agent forecasts market conditions. Defaults to 24.
    max_bid_price : float
        Maximum allowable bid price. Defaults to 100.
    min_bid_price : float
        Maximum allowable bid price. Defaults to  -100.
    max_demand : float
        Maximum demand capacity of the unit. Defaults to 10e3.
    device : str
        Device for computation, such as "cpu" or "cuda". Defaults to "cpu".
    float_type : str
        Data type for floating-point calculations, typically "float32". Defaults to "float32".
    learning_mode : bool
        Indicates whether the agent is in learning mode. Defaults to False.
    algorithm : str
        Name of the RL algorithm in use. Defaults to "matd3".
    actor_architecture_class : type[torch.nn.Module]
        Class of the neural network architecture used for the actor network. Defaults to MLPActor.
    actor : torch.nn.Module
        Actor network for determining actions.
    order_types : list[str]
        Types of market orders supported by the strategy. Defaults to ["SB"].
    action_noise : NormalActionNoise
        Noise model added to actions during learning to encourage exploration. Defaults to None.
    collect_initial_experience_mode : bool
        Whether the agent is collecting initial experience through exploration. Defaults to True.

    Args
    ----
    *args : Variable length argument list.
    **kwargs : Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):

        obs_dim = kwargs.pop("obs_dim", 43) # 36 shared observations + 7 unique_observations  
        act_dim = kwargs.pop("act_dim", 7) # bids and quantities for flexible generation, price for inflexible generation
        unique_obs_dim = kwargs.pop("unique_obs_dim", 7) # tot inflexible generation, flexible gen quantiles and marginal cost quanties
        self.curve_splits = act_dim // 2 # the number of (price, quantity) tuples to offer (default: 3)
        
        kwargs["unit_id"] = 'rl_operator'
        
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            unique_obs_dim=unique_obs_dim,
            *args,
            **kwargs,
        )

        # 'foresight' represents the number of time steps into the future that we will consider
        # when constructing the observations. This value is fixed for each strategy, as the
        # neural network architecture is predefined, and the size of the observations must remain consistent.
        # If you wish to modify the foresight length, remember to also update the 'obs_dim' parameter above,
        # as the observation dimension depends on the foresight value.
        self.foresight = 12

        # define allowed order types
        self.order_types = kwargs.get("order_types", ["SB"])
        # needed to scale actions and observations back and forth between [-1,1]
        self.scale = lambda x, m, M: -1 + 2 * (x - m) / (M - m)
        self.rescale = lambda x, m, M: ((x + 1) / 2) * (M - m) + m
    
    


    def calculate_bids(
        self,
        operator: "UnitsOperator",
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Calculates bids based on the current observations and actions derived from the actor network.

        Args
        ----
            operator (UnitsOperator): The operator that bids on the market.
            market_config (MarketConfig): The configuration of the market.
            product_tuples (list[Product]): List of products with start and end times for bidding.
            **kwargs : Additional keyword arguments.

        Returns
        -------
        Orderbook
            Contains bid entries for each product, including start time, end time, price, and volume.

        Notes
        -----
        This method obtains actions as a tensor of quantities and bid prices, where the first
        bid is the price for inflex generation, the next self.curve_splits are bid quantities
        for flex generation, followed by bid prices for flex generation. Unit-specific bids 
        are submitted accordingly by sorting the units by their marginal cost, and bidding 
        their inflex generation to the inflex bid price, the flex generation to the lowest 
        curve_splits whose capacity is still not fully bid.
        """

        start = product_tuples[0][0]
        end = product_tuples[0][1]

        # assign forecaster, outputs dict and technology_max_power dict to units_operator
        if not hasattr(operator, 'installed_capacity'):
            raise ValueError("Operator is not a RL-Operator.")
        
        # =============================================================================
        # 1. Get the Observations, which are the basis of the action decision
        # =============================================================================
        next_observation = self.create_observation(
            unit=operator,
            market_id=market_config.market_id,
            start=start,
            end=end,
        )

        # =============================================================================
        # 2. Get the Actions, based on the observations
        # =============================================================================
        actions, noise = self.get_actions(next_observation)
    
        # =============================================================================
        # 3. Transform Actions into bids
        # =============================================================================
        # actions are in the range [-1,1], we need to transform them into actual bids
        # we can use our domain knowledge to guide the bid formulation
        
        # first curve_splits actions are quantities
        scaled_price_inflex = actions[0].item()
        scaled_quant = actions[1:self.curve_splits+1].numpy()
        scaled_price = actions[self.curve_splits+1:].numpy()
        
        bid_quant = self.rescale(scaled_quant, 0, operator.installed_capacity)
        bid_price = self.rescale(scaled_price, self.min_bid_price, self.max_bid_price)
        price_inflex = self.rescale(scaled_price_inflex, self.min_bid_price, self.max_bid_price)
        # sorted is needed because we are not enforcing prices to be increasing
        sorted_index = np.argsort(bid_price)

        units = {}
        bids = []
        
        for unit_id, unit in operator.units.items():
            min_power, max_power = unit.calculate_min_max_power(start,end)
            min_mw, max_mw = min_power[0], max_power[0]
            mc = unit.calculate_marginal_cost(start, max_mw)
            # unit tuple of flex and inflex avail capacity, marginal cost
            units[unit_id] = (min_mw, max_mw-min_mw, mc)

        # sort unit tuples by by marginal cost
        sorted_units = sorted(units, key=lambda x: x[-1]) 

        for ix in range(self.curve_splits):
            
            # bid the maximum available capacity of units until bid_quantity is covered
            # or there are no further units
            for unit_id in sorted_units:
                inflex_gen, flex_gen, mc = units.get(unit_id, (None,None,None))

                #if the quantity is fully offered move to next curve split
                if bid_quant[ix] == 0: 
                    break
                
                # if unit capacity was fully used, move to next unit
                elif mc is None:
                    continue
                
                else: 
                    volume = min(flex_gen, bid_quant[sorted_index[ix]])

                    # we assume all units to bid their inflexible generation at the same price
                    if inflex_gen > 0:
                        bids.append({           
                            "start_time": start,
                            "end_time": end,
                            "only_hours": None,
                            "price": float(price_inflex),
                            "volume": float(inflex_gen),
                            "unit_id": unit_id,
                            "bid_id": f"{operator.id}_{unit_id}_inflex", # units_operator.id_unit.id
                            "node": operator.units[unit_id].node,
                            })
                    
                    bids.append({           
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": float(bid_price[sorted_index[ix]]),
                    # min() for partial bid if avail capacity is greater than bid_quantity left
                    "volume": float(volume),
                    "unit_id": unit_id,
                    "bid_id": f"{operator.id}_{unit_id}_{ix}", # units_operator.id_unit.id
                    "node": operator.units[unit_id].node,
                    })

                    # if unit capacity was fully used, drop it; else, reduce its capacity
                    if volume == flex_gen:
                        units.pop(unit_id)
                    else: 
                        units[unit_id] = (0, flex_gen - bid_quant[ix], mc)
                    
                    bid_quant[ix] -= volume                     

        # store results in unit outputs as lists to be written to the buffer for learning
        operator.outputs["rl_observations"].append(next_observation)
        operator.outputs["rl_actions"].append(actions)

        # store results in unit outputs as series to be written to the database by the unit operator
        operator.outputs["actions"].at[start] = actions
        operator.outputs["exploration_noise"].at[start] = noise

        return bids


    def get_actions(self, next_observation:th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Compute actions based on the current observation.

        Args
        ----
        next_observation (torch.Tensor): The current observation, where the last element is assumed to be the marginal cost.

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing: Actions to be taken (with or without noise). The noise component (if any), useful for diagnostics.
            The action to be taken represent the price for inflex generation, and self.curve_splits bid volumes and prices for
            flexible generation, scaled to the range [-1, 1].

        Notes
        -----
        During learning, exploratory noise is applied and already part of the curr_action unless in evaluation mode.
        In initial exploration mode, flex price and quantities actions are sampled around the respective quantiles, and inflex
        price around zero to explore its vicinity. This assumes the last 2*self.curve_splits+1 elements of`next_observation`
        have the following structure: [inflex_gen, flex_quant, flex_mc]
        """

        # Get the base action and associated noise from the parent implementation
        curr_action, noise = super().get_actions(next_observation)

        if self.learning_mode and not self.evaluation_mode:
            if self.collect_initial_experience_mode:
                # Assumes last dimensions of the observation are the individual obs
                individual_obs = next_observation[-2*self.curve_splits+1:]
                marginal_cost = individual_obs[
                    self.curve_splits+1:
                ].detach()  # ensure no gradients flow through
                # Add marginal cost to the bid action directly for initial random exploration
                curr_action[self.curve_splits+1:] = curr_action[self.curve_splits+1:] + marginal_cost 
                
                # bid price for inflexible capacity 
                inflex_quant = self.scale(0, self.min_bid_price, self.max_bid_price) # if min_bid_price = -1 * max_bid_price, this is 0
                curr_action[0] = inflex_quant

                flex_quant = individual_obs[
                    1: self.curve_splits+1
                ].detach() # ensure no gradients flow through
                # Assumes to bid all flexible capacity
                curr_action[1:self.curve_splits+1] = curr_action[1:self.curve_splits+1] + flex_quant 
                        
        return curr_action, noise


    def get_individual_observations(
        self, operator: "UnitsOperator", start: datetime, end: datetime
    ):
        """
        Retrieves the observations specific to the units_operator. Returns a scaled array
        in range [-1,1] with the structure: [inflex_gen, flex_quant, flex_mc],
        where:
            inflex_gen: portfolio min generation capacity (must-run)
            flex_quant: self.curve_splits quantiles of flex generation
            flex_mc: self.curve_splits quantiles of flex marginal costs.


        Args
        ----
            operator (UnitsOperator): The operator that bids on the market.
            start (datetime.datetime): Start time for the observation period.
            end (datetime.datetime): End time for the observation period

        Returns
        -------
        individual_observations (np.array): must-run generation, quantiles of avail max generation,
        weighted quantiles of marginal costs.

        Notes
        -----
            Generation is scaled by self.installed_capacity[self.market_id].
            Avg marginal cost is min-max scaled by [self.min_bid_price, self.max_bid_price].
        """

        units_dict = {}
        min_gen = 0 # track total must-run quantities if any

        #iteratively computes the total dispatch volume and volume-weighted marginal cost
        for unit_id, unit in operator.units.items():
            min_power, max_power = unit.calculate_min_max_power(start, end)
            min_mw, max_mw = min_power[0], max_power[0]
            mc = unit.calculate_marginal_cost(start, max_mw)
            # unit tuple of flex avail capacity, marginal cost
            units_dict[unit_id] = (max_mw - min_mw, mc)
            min_gen+= min_mw 
        
        # sort unit tuples by marginal cost
        sorted_units = sorted(units_dict, key=lambda x: x[-1]) 
        flex_quant = [units_dict[unit][0] for unit in sorted_units] 
        flex_mc = [units_dict[unit][1] for unit in sorted_units] 

        q = np.linspace(0,1,self.curve_splits)
        quant_q = np.quantile(np.cumsum(flex_quant), q=q)
        cost_q = np.quantile(flex_mc, q=q, weights=flex_quant, method='inverted_cdf')
        #creates a list of scaled volumes and average marginal costs for each technology (alphabetically sorted)
        scaled_quant = quant_q / operator.installed_capacity
        scaled_cost = self.scale(cost_q, self.min_bid_price, self.max_bid_price)

        individual_observations = np.array(
            # total inflexible generation, supply curve of flexible generation and marginal costs
            [min_gen / operator.installed_capacity, *scaled_quant, *scaled_cost]
        )

        return individual_observations


    def calculate_reward(
        self,
        operator: "UnitsOperator",
        market_config: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculates the reward for the unit based on profits, costs, and opportunity costs from market transactions.

        Args
        ----
            operator (UnitsOperator): The operator for which to calculate the reward.
            market_config (MarketConfig): The configuration of the market.
            orderbook (Orderbook): Orderbook containing executed bids and details.

        Notes
        -----
        The reward is computed by multiplying the following:
        **Profit**: Income from accepted bids minus marginal and start-up costs.
        **Scaling**: A scaling factor to normalize the reward to the range [-1,1]

        The reward is scaled and stored along with other outputs in the data to support learning.
        """
        # Function is called after the market is cleared, and we get the market feedback,
        # allowing us to calculate profit based on the realized transactions.
        print("Does the reward ever get calculated")
        product_type = market_config.product_type
        start = orderbook[0]["start_time"]
        end = orderbook[0]["end_time"]

        # Depending on how the unit calculates marginal costs, retrieve cost values.
    
        market_clearing_price = orderbook[0]["accepted_price"]
        duration = (end - start) / timedelta(hours=1)

        income = 0.0
        operational_cost = 0.0

        accepted_volume_total = 0
        offered_volume_total = 0

        # Iterate over all orders in the orderbook to calculate order-specific profit.
        for order in orderbook:
            unit_id = order["unit_id"]
            unit = operator.units[unit_id]

            accepted_volume = order.get("accepted_volume", 0)
            accepted_volume_total += accepted_volume
            offered_volume_total += order["volume"]

            # Calculate profit as income minus operational cost for this event.
            order_income = market_clearing_price * accepted_volume * duration
            marginal_cost = marginal_cost = unit.calculate_marginal_cost(
                start, unit.outputs[product_type].at[start])
            order_cost = marginal_cost * accepted_volume * duration

            # Consideration of start-up costs, divided evenly between upward and downward regulation events.
            if (
                unit.outputs[product_type].at[start] != 0
                and unit.outputs[product_type].at[start - unit.index.freq] == 0
            ):
                operational_cost += unit.hot_start_cost / 2
            elif (
                unit.outputs[product_type].at[start] == 0
                and unit.outputs[product_type].at[start - unit.index.freq] != 0
            ):
                operational_cost += unit.hot_start_cost / 2

            # Accumulate income and operational cost for all orders.
            income += order_income
            operational_cost += order_cost

        profit = income - operational_cost
        # scaling factor to normalize the reward to the range [-1,1]
        scaling = 1 / (self.max_bid_price * operator.installed_capacity)
        reward = scaling * profit

        # Store results in unit outputs, which are later written to the database by the unit operator.
        # `end_excl` marks the last product's start time by subtracting one frequency interval.
        end_excl = end - unit.index.freq 
        operator.outputs["profit"].loc[start:end_excl] += profit
        operator.outputs["reward"].loc[start:end_excl] = reward
        #units_operator.outputs["regret"].loc[start:end_excl] = regret_scale * opportunity_cost
        operator.outputs["total_costs"].loc[start:end_excl] = operational_cost
        operator.outputs["rl_rewards"].append(reward)

