# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.strategies.naive_strategies import NaiveSingleBidStrategy
from datetime import datetime, timedelta
import numpy as np


class UnitOperatorStrategy:
    """
    The UnitOperatorStrategy is similar to the UnitStrategy.
    A UnitOperatorStrategy calculates the bids for all units of a units operator.
    """

    def __init__(self, *args, **kwargs):
        pass

    def calculate_bids(
        self,
        units_operator,  # type: UnitsOperator
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a units operator and
        defines how the units managed by it should bid.

        This gives a lot of flexibility to the market bids.

        Args:
            units_operator (UnitsOperator): The operator handling the units.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """
        return []

    def tot_capacity(
            self,
            units_operator,  # type: UnitsOperator
    ) -> dict[str,float]:
        """
        Computes the total capacity of the units owned by a unit operator by market.
        
        Args: 
            units_operator (UnitsOperator): The operator that bids on the market(s).
        Returns: 
            dict: a dictionary indexed by market_id.
        """

        tot_capacity = {}

        for unit in units_operator.units.values():
            for market_id in unit.bidding_strategies.keys():
                tot_capacity[market_id] = tot_capacity.get(market_id, 0) + unit.max_power

        return tot_capacity
    


class DirectUnitOperatorStrategy(UnitOperatorStrategy):
    def calculate_bids(
        self,
        units_operator,  # type: UnitsOperator
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Formulates the bids to the market according to the bidding strategy of the each unit individually.
        This calls calculate_bids of each unit and returns the aggregated list of all individual bids of all units.

        Args:
            units_operator: The units operator whose units are queried
            market_config (MarketConfig): The market to formulate bids for.
            product_tuples (list[tuple]): The products to formulate bids for.

        Returns:
            OrderBook: The orderbook that is submitted as a bid to the market.
        """
        bids: Orderbook = []

        for unit_id, unit in units_operator.units.items():
            product_bids = unit.calculate_bids(
                market_config=market_config,
                product_tuples=product_tuples,
            )
            for i, order in enumerate(product_bids):
                order["agent_addr"] = units_operator.context.addr
                if market_config.volume_tick:
                    order["volume"] = round(order["volume"] / market_config.volume_tick)
                if market_config.price_tick:
                    order["price"] = round(order["price"] / market_config.price_tick)
                if "bid_id" not in order.keys() or order["bid_id"] is None:
                    order["bid_id"] = f"{unit_id}_{i+1}"
                order["unit_id"] = unit_id
                bids.append(order)

        return bids


class CournotPortfolioStrategy(UnitOperatorStrategy):
    """
    A Cournot strategy that adds a markup to the marginal cost of each unit of
    the units operator. The marginal cost is computed with NaiveSingleBidStrategy,
    and the markup depends on the total capacity of the unit operator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.markup = kwargs.get("markup", 0)

    def calculate_bids(
        self,
        units_operator,  # type: UnitsOperator
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            units_operator (UnitsOperator): The units operator that bids on the market.
            market_config (MarketConfig): The configuration of the market.
            product_tuples (list[Product]): The list of all products open for bidding.

        Returns:
            Orderbook: The bids consisting of the start time, end time, price and volume.
        """

        max_power = self.tot_capacity(units_operator)[market_config.market_id]
        operator_bids = Orderbook()

        for unit_id, unit in units_operator.units.items():
            # Compute bids from marginal costs of a unit
            bids = NaiveSingleBidStrategy().calculate_bids(
                unit,
                market_config,
                product_tuples,
            )
            # Apply Cournot mark-up
            for bid in bids:
                bid["price"] += self.markup * max_power
                bid["unit_id"] = unit_id

            operator_bids.extend(bids)

        return operator_bids


class PortfolioRLStrategy(BaseLearningStrategy, UnitOperatorStrategy):
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
        Maximum allowable bid price. Defaults to -100.
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
        act_dim = kwargs.pop("act_dim", 6) # price for flexible generation, price for inflexible generation
        unique_obs_dim = kwargs.pop("unique_obs_dim", 7) # tot inflexible capacity, tot flexible capacity, flexible marginal cost quantiles
        self.n_prices = act_dim - 1 # the number of prices to offer for flexible generation (default: 5)
        
        kwargs["bidder_id"] = 'Operator-RL'
        
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
        units_operator,  # type: UnitsOperator
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Calculates bids based on the current observations and actions derived from the actor network.

        Args
        ----
            units_operator (UnitsOperator): The operator that bids on the market.
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
        bid is the price for inflex generation, the next self.flex_prices are bid prices for 
        flex generation. Unit-specific bids are submitted accordingly by sorting the units 
        by their marginal cost, and bidding their inflex generation for the inflex bid price, 
        their flex generation for the next price quantile whose capacity is still not fully bid.
        """

        start = product_tuples[0][0]
        end = product_tuples[0][1]

        # assign forecaster, outputs dict and technology_max_power dict to units_operator
        if not hasattr(units_operator, 'installed_capacity'):
            market_id = market_config.market_id
            tot_capacity = self.tot_capacity(units_operator)
            units_operator.installed_capacity = tot_capacity[market_id]
        
        # =============================================================================
        # 1. Get the Observations, which are the basis of the action decision
        # =============================================================================
        next_observation = self.create_observation(
            unit=units_operator,
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
        
        # first n_prices actions are quantities
        scaled_quant_flex = next_observation[-self.n_prices+1]
        scaled_price_inflex = actions[0].item()
        scaled_price_flex = actions[1:].numpy()
        
        tot_quant_flex = self.rescale(scaled_quant_flex, 0, units_operator.installed_capacity)
        price_flex = self.rescale(scaled_price_flex, self.min_bid_price, self.max_bid_price)
        price_inflex = self.rescale(scaled_price_inflex, self.min_bid_price, self.max_bid_price)
        # sorted is needed because we are not enforcing prices to be increasing
        sorted_index = np.argsort(price_flex)

        units = {}
        bids = []
        
        for unit_id, unit in units_operator.units.items():
            min_power, max_power = unit.calculate_min_max_power(start,end)
            min_mw, max_mw = min_power[0], max_power[0]
            mc = unit.calculate_marginal_cost(start, max_mw)
            # unit tuple of flex and inflex avail capacity, marginal cost
            units[unit_id] = (min_mw, max_mw-min_mw, mc)

        # sort unit tuples by by marginal cost
        sorted_units = sorted(units, key=lambda x: x[-1]) 

        for ix in range(self.n_prices):
            # each quantile has the same size, given by tot_quant_flex / self.n_prices
            quant_flex_ix = tot_quant_flex / self.n_prices
            # bid unit max capacity until quant_flex_ix is covered or no more units to bid
            for unit_id in sorted_units:
                inflex_gen, flex_gen, mc = units.get(unit_id, (None,None,None))

                #if the quantity is fully offered move to next curve split
                if quant_flex_ix == 0: 
                    break
                
                # if unit capacity was fully used, move to next unit
                elif mc is None:
                    continue
                
                else: 
                    volume = min(flex_gen, quant_flex_ix)

                    # we assume all units to bid their inflexible generation at the same price
                    if inflex_gen > 0:
                        bids.append({           
                            "start_time": start,
                            "end_time": end,
                            "only_hours": None,
                            "price": float(price_inflex),
                            "volume": float(inflex_gen),
                            "unit_id": unit_id,
                            "bid_id": f"{units_operator.id}_{unit_id}_inflex", # units_operator.id_unit.id
                            "node": units_operator.units[unit_id].node,
                            })
                    
                    bids.append({           
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": float(price_flex[sorted_index[ix]]),
                    # min() for partial bid if avail capacity is greater than bid_quantity left
                    "volume": float(volume),
                    "unit_id": unit_id,
                    "bid_id": f"{units_operator.id}_{unit_id}_{ix}", # units_operator.id_unit.id
                    "node": units_operator.units[unit_id].node,
                    })

                    # if unit capacity was fully used, drop it; else, reduce its capacity
                    if volume == flex_gen:
                        units.pop(unit_id)
                    else: 
                        units[unit_id] = (0, flex_gen - quant_flex_ix, mc)
                    
                    quant_flex_ix -= volume                     

        # store results in unit outputs as lists to be written to the buffer for learning
        units_operator.outputs["rl_observations"].append(next_observation)
        units_operator.outputs["rl_actions"].append(actions)

        # store results in unit outputs as series to be written to the database by the unit operator
        units_operator.outputs["actions"].at[start] = actions
        units_operator.outputs["exploration_noise"].at[start] = noise

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
            The output action has the structure: [inflex_price, flex_price ] where inflex_price is an int representing a single
            price for inflexible generation, and flex_price a list of self.n_prices price flexible generation. 
            Output is scaled to the range [-1, 1].

        Notes
        -----
        During learning, exploratory noise is applied and already part of the curr_action unless in evaluation mode.
        In initial exploration mode, flex price and quantities actions are sampled around the respective quantiles, and inflex
        price around zero to explore its vicinity. This assumes the last self.n_prices+2 elements of`next_observation`
        have the following structure: [inflex_quantity, flex_quantity, flex_mc]
        """

        # Get the base action and associated noise from the parent implementation
        curr_action, noise = super().get_actions(next_observation)

        if self.learning_mode and not self.evaluation_mode:
            if self.collect_initial_experience_mode:
                # Assumes last dimensions of the observation are the individual obs
                individual_obs = next_observation[-self.n_prices+2:]
                marginal_costs = individual_obs[2:].detach()  # ensure no gradients flow through
                # Add marginal cost to the bid action directly for initial random exploration
                curr_action[1:] = curr_action[1:] + marginal_costs 
                
                # bid price for inflexible capacity 
                inflex_quant = self.scale(0, self.min_bid_price, self.max_bid_price) # if min_bid_price = -1 * max_bid_price, this is 0
                curr_action[0] = curr_action[0] + inflex_quant
                        
        return curr_action, noise


    def get_individual_observations(
        self, 
        units_operator,  # type: UnitsOperator
        start: datetime, end: datetime
    ):
        """
        Retrieves the observations specific to the units_operator. Returns a scaled array
        in range [-1,1] with the structure: [inflex_gen, flex_quant, flex_mc],
        where:
            inflex_gen: portfolio min generation capacity (must-run)
            flex_quant: self.n_prices quantiles of flex generation
            flex_mc: self.n_prices quantiles of flex marginal costs.


        Args
        ----
            units_operator (UnitsOperator): The operator that bids on the market.
            start (datetime.datetime): Start time for the observation period.
            end (datetime.datetime): End time for the observation period

        Returns
        -------
        individual_observations (np.array): total inflexible capacity, total flexible capacity,
        weighted quantiles of marginal costs.

        Notes
        -----
            Generation is scaled by self.installed_capacity[self.market_id].
            Avg marginal cost is min-max scaled by [self.min_bid_price, self.max_bid_price].
        """

        unit_tuples = []
        tot_quant_inflex = 0 
        tot_quant_flex = 0

        #iteratively computes the total dispatch volume and volume-weighted marginal cost
        for unit in units_operator.units.values():
            min_power, max_power = unit.calculate_min_max_power(start, end)
            min_mw, max_mw = min_power[0], max_power[0]
            mc = unit.calculate_marginal_cost(start, max_mw)
            # unit tuple of flex capacity, marginal cost
            unit_tuples.append((max_mw - min_mw, mc))
            tot_quant_inflex+= min_mw 
            tot_quant_flex+= max_mw - min_mw

        # Total flexible and inflexible generation
        scaled_quant_flex = tot_quant_flex / units_operator.installed_capacity
        scaled_quant_inflex = tot_quant_inflex / units_operator.installed_capacity
        
        # Sort unit tuples by marginal cost
        sorted_tuples = sorted(unit_tuples, key=lambda x: x[-1]) 
        flex_q, flex_mc = zip(*sorted_tuples)
    
        # Average marginal costs for each quantile
        q = np.linspace(0,1,self.n_prices)
        cost_q = np.quantile(flex_mc, q=q, weights=flex_q, method='inverted_cdf')
        scaled_cost = self.scale(cost_q, self.min_bid_price, self.max_bid_price)

        individual_observations = np.array(
            # total inflexible generation, total flexible generation and marginal costs
            [scaled_quant_inflex, scaled_quant_flex, *scaled_cost]
        )

        return individual_observations


    def calculate_reward(
        self,
        units_operator,  # type: UnitsOperator
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculates the reward for the unit based on profits, costs, and opportunity costs from market transactions.

        Args
        ----
            units_operator (UnitsOperator): The operator for which to calculate the reward.
            marketconfig (MarketConfig): The configuration of the market.
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
        product_type = marketconfig.product_type
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
            unit = units_operator.units[unit_id]

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
        scaling = 1 / (self.max_bid_price * units_operator.installed_capacity)
        reward = scaling * profit

        # Store results in unit outputs, which are later written to the database by the unit operator.
        # `end_excl` marks the last product's start time by subtracting one frequency interval.
        end_excl = end - unit.index.freq 
        units_operator.outputs["profit"].loc[start:end_excl] += profit
        units_operator.outputs["reward"].loc[start:end_excl] = reward
        #units_operator.outputs["regret"].loc[start:end_excl] = regret_scale * opportunity_cost
        units_operator.outputs["total_costs"].loc[start:end_excl] = operational_cost
        units_operator.outputs["rl_rewards"].append(reward)

