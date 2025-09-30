# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import numpy as np
import torch as th

from assume.common import UnitsOperator
from assume.common.market_objects import (
    MarketConfig,
    Orderbook,
)
from assume.common.utils import convert_tensors, create_rrule, get_products_index
from assume.strategies import BaseStrategy, LearningStrategy
from assume.units import BaseUnit
from assume.common.fast_pandas import TensorFastSeries, FastSeries
from collections import defaultdict

logger = logging.getLogger(__name__)


class RLUnitsOperator(UnitsOperator):
    def __init__(
        self,
        available_markets: list[MarketConfig],
        portfolio_strategies: dict[str, BaseStrategy] = {},
    ):
        super().__init__(available_markets, portfolio_strategies)

        self.rl_bidders = []
        self.learning_strategies = {
            "obs_dim": 0,
            "act_dim": 0,
            "device": "cpu",
        }

        

    def on_ready(self):
        super().on_ready()

        # todo
        recurrency_task = create_rrule(
            start=self.context.data["train_start"],
            end=self.context.data["train_end"],
            freq=self.context.data.get("train_freq", "24h"),
        )

        self.context.schedule_recurrent_task(
            self.write_to_learning_role, recurrency_task
        )

    def add_unit(
        self,
        unit: BaseUnit,
    ) -> None:
        """
        Create a unit.

        Args:
            unit (BaseUnit): The unit to be added.
        """
        self.units[unit.id] = unit

        # check if unit or operator have a learning strategy for any of the available markets
        # bidder_type stores type of learning bidder: either unit or unit_operator
        # rl_bidders stores the rl_bidders (either one operator or multiple units)
        self.bidder_type = "unit"
        for market in self.available_markets:
            portfolio_strategy = self.portfolio_strategies.get(market.market_id)
            bidding_strategy = unit.bidding_strategies.get(market.market_id)

            if isinstance(portfolio_strategy, LearningStrategy):
                strategy = portfolio_strategy
                self.bidder_type = "units_operator"
                self.rl_bidders.append(self)
                # equips operator with forecaster, outputs and installed market capacity
                self.init_portfolio_learning(market.market_id)
                
            elif isinstance(strategy, LearningStrategy):
                strategy = bidding_strategy
                self.bidder_type = "unit"
                self.rl_bidders.append(unit)

            else:
                continue
                
            self.learning_strategies.update(
                {
                    "obs_dim": strategy.obs_dim,
                    "act_dim": strategy.act_dim,
                    "device": strategy.device,
                }
            )


    def write_learning_to_output(self, orderbook: Orderbook, market_id: str) -> None:
        """
        Sends the current rl_strategy update to the output agent.

        Args:
            products_index (pandas.DatetimeIndex): The index of all products.
            marketconfig (MarketConfig): The market configuration.
        """

        products_index = get_products_index(orderbook)

        # should write learning results if at least one bidding_strategy is a learning strategy
        # or if the portfolio strategy is a learning strategy
        
        if not (len(self.rl_bidders) > 0 and orderbook):
            return

        output_agent_list = []
        start = products_index[0]

        for bidder in self.rl_bidders:
            strategies = getattr(bidder, 
                                 "bidding_strategies" if self.bidder_type == "unit" 
                                 else "portfolio_strategies")
            
            strategy = strategies.get(market_id)

            # rl only for energy market for now!
            if isinstance(strategy, LearningStrategy):
                output_dict = {
                    "datetime": start,
                    self.bidder_type: bidder.id,
                }

                output_dict.update(
                    {
                        "profit": bidder.outputs["profit"].at[start],
                        "reward": bidder.outputs["reward"].at[start],
                    }
                )

                if "regret" in bidder.outputs:
                    output_dict.update({"regret": bidder.outputs["regret"].at[start]})

                action_tuple = bidder.outputs["actions"].at[start]
                noise_tuple = bidder.outputs["exploration_noise"].at[start]
                action_dim = action_tuple.numel()

                for i in range(action_dim):
                    output_dict[f"exploration_noise_{i}"] = noise_tuple[i]
                    output_dict[f"actions_{i}"] = action_tuple[i]

                output_agent_list.append(output_dict)

        db_addr = self.context.data.get("learning_output_agent_addr")

        if db_addr and output_agent_list:
            self.context.schedule_instant_message(
                receiver_addr=db_addr,
                content={
                    "context": "write_results",
                    "type": "rl_params",
                    "data": output_agent_list,
                },
            )

    async def write_to_learning_role(
        self,
    ) -> None:
        """
        Writes learning results to the learning agent.

        """
        if len(self.rl_bidders) == 0:
            return

        obs_dim = self.learning_strategies["obs_dim"]
        act_dim = self.learning_strategies["act_dim"]
        device = self.learning_strategies["device"]

        lr_bidders_count = len(self.rl_bidders)

        # Collect the number of reward values for each unit.
        # This represents how many complete transitions we have for each unit.
        # Using a set ensures we capture only unique lengths across all units.
        values_len_set = {len(bidder.outputs["rl_rewards"]) for bidder in self.rl_bidders}

        # Check if all units have the same number of reward values.
        # If the set contains more than one unique length, it means at least one unit
        # has a different number of rewards, indicating an inconsistency.
        # This is considered an error condition, so we raise an exception.
        if len(values_len_set) > 1:
            raise ValueError(
                "Mismatch in reward value lengths: All units must have the same number of rewards."
            )

        # Since all units have the same length, extract the common length
        values_len = values_len_set.pop()

        # return if no data is available
        if values_len == 0:
            return

        all_observations = th.zeros(
            (values_len, lr_bidders_count, obs_dim), device=device
        )
        all_actions = th.zeros(
            (values_len, lr_bidders_count, act_dim), device=device
        )
        all_rewards = []

        # Iterate through each RL unit and collect all of their observations, actions, and rewards
        # making it dependent on values_len ensures that data is not stored away for which the reward was not calculated yet
        for i, bidder in enumerate(self.rl_bidders):
            # Convert pandas Series to torch Tensor
            obs_tensor = th.stack(bidder.outputs["rl_observations"][:values_len], dim=0)
            actions_tensor = th.stack(bidder.outputs["rl_actions"][:values_len], dim=0)

            all_observations[:, i, :] = obs_tensor
            all_actions[:, i, :] = actions_tensor
            all_rewards.append(bidder.outputs["rl_rewards"])

            # reset the outputs
            bidder.reset_saved_rl_data()

        all_observations = all_observations.numpy(force=True)
        all_actions = all_actions.numpy(force=True)

        all_rewards = np.array(all_rewards).T

        rl_agent_data = (all_observations, all_actions, all_rewards)

        learning_role_addr = self.context.data.get("learning_agent_addr")

        if learning_role_addr:
            self.context.schedule_instant_message(
                content={
                    "context": "rl_training",
                    "type": "save_buffer_and_update",
                    "data": rl_agent_data,
                },
                receiver_addr=learning_role_addr,
            )

    async def formulate_bids(
        self, market: MarketConfig, products: list[tuple]
    ) -> Orderbook:
        """
        Formulates the bid to the market according to the bidding strategy of the each unit individually.

        Args:
            market (MarketConfig): The market to formulate bids for.
            products (list[tuple]): The products to formulate bids for.

        Returns:
            OrderBook: The orderbook that is submitted as a bid to the market.
        """

        orderbook: Orderbook = []

        for unit_id, unit in self.units.items():
            product_bids = unit.calculate_bids(
                market,
                product_tuples=products,
            )
            for i, order in enumerate(product_bids):
                order["agent_addr"] = self.context.addr

                if market.volume_tick:
                    order["volume"] = round(order["volume"] / market.volume_tick)
                if market.price_tick:
                    order["price"] = round(order["price"] / market.price_tick)
                if "bid_id" not in order.keys() or order["bid_id"] is None:
                    order["bid_id"] = f"{unit_id}_{i+1}"
                order["unit_id"] = unit_id
                orderbook.append(order)

        # Convert all CUDA tensors to CPU in one pass
        return convert_tensors(orderbook)
    


    async def formulate_bids_portfolio(
        self, market: MarketConfig, products: list[tuple]
    ) -> Orderbook:
        """
        Formulates the bid to the market according to the bidding strategy of the unit operator.

        Args:
            market (MarketConfig): The market to formulate bids for.
            products (list[tuple]): The products to formulate bids for.

        Returns:
            OrderBook: The orderbook that is submitted as a bid to the market.
        """
        orderbook: Orderbook = []
        portfolio_strategy = self.portfolio_strategies[market.market_id]
        product_bids = portfolio_strategy.calculate_bids(self, market, products)

        for i, order in enumerate(product_bids):
            order["agent_addr"] = self.context.addr
            if market.volume_tick:
                order["volume"] = round(order["volume"] / market.volume_tick)
            if market.price_tick:
                order["price"] = round(order["price"] / market.price_tick)
            if "bid_id" not in order.keys() or order["bid_id"] is None:
                order["bid_id"] = f"{self.id}_{i+1}"
            
            orderbook.append(order)
            
        # Convert all CUDA tensors to CPU in one pass
        return convert_tensors(orderbook)
    

    def init_portfolio_learning(self, market_id:str):
        """
        If the portfolio strategy of the units operator in a market is a learning strategy,
        add required features to the units operator (forecaster, outputs dictionary, installed
        capacity for the given market).
        """
        portfolio_strategy = self.portfolio_strategies[market_id]
        tot_capacity = portfolio_strategy.calculate_tot_capacity(self)
        self.installed_capacity = tot_capacity[market_id]

        for unit in self.units.values():
            if hasattr(unit, "forecaster"):
                self.forecaster = unit.forecaster
                break

        self.outputs = defaultdict(lambda: FastSeries(value=0.0, index=unit.index))
        
        self.outputs["actions"] = TensorFastSeries(value=0.0, index=unit.index, name='actions')
        self.outputs["exploration_noise"] = TensorFastSeries(
            value=0.0,
            index=unit.index,
            name='exploration_noise',
        )
        self.outputs["reward"] = FastSeries(value=0.0, index=unit.index, name='reward')
        # RL data stored as lists to simplify storing to the buffer
        self.outputs["rl_observations"] = []
        self.outputs["rl_actions"] = []
        self.outputs["rl_rewards"] = []   