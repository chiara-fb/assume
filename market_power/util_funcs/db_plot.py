import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import seaborn as sns
from IPython.display import clear_output, display
from matplotlib.patches import Patch


COLOR_DICT = {'open cycle gas turbine': 'pink', 
              'oil': 'red', 
              'lignite': 'brown', 
              'nuclear': 'grey', 
              'hard coal': 'black',
              'combined cycle gas turbine': 'olive', 
              'wind_offshore': 'blue', 
              'hydro': 'cyan',
              'wind_onshore': 'blue', 
              'biomass': 'purple', 
              'solar': 'gold'}



def supply_curve_ax(ax:plt.Axes, ax_params:dict):
            # standard supply curve: volume on x, price on y (step)
    #ax.step(np.concatenate(([0], cum_vol)), np.concatenate(([prices[0]], prices)), where='post', linewidth=2)
    ax.set_xlabel('Cumulative Volume')
    ax.set_ylabel('Price')
    ax.set_xlim(left=0, right=ax_params["max_vol"] + 50)
    ax.set_ylim(bottom=ax_params["min_bid"] - 50, 
                top=ax_params["max_bid"] + 50)

    # draw vertical line at that cumulative volume and annotate accepted price
    x, y = ax_params["intersect_x"], ax_params["intersect_y"]
    ax.axvline(x=x, color='red', linestyle='--', 
               linewidth=1.5, label=f'accepted_price={y:.2f}')
    ax.plot([x], [y], 'ro')  # marker at intersection
    ax.annotate(f"{y:.1f} EUR/MWh", (x,y), (x-1000, y+0.5), color="black")

    title = ax_params['name']

    if 'profits' in ax_params:
       title+= f" profits: {ax_params['profits']:,.1f} EUR"
    
    ax.set_title(title)

    return ax    



def plot_supply_curves(bids_dfs:dict,  
                       color_dict:dict=COLOR_DICT, 
                       time_sleep:float=0.1, 
                       only_hours:list=None, 
                       strategic_operator:str="Operator-RL",
                       only_operators:list=None):
    """
    Iterate over datetimes in `bids_df` and plot a supply curve (cumulative volume vs price)
    for each time. A vertical line marks the intersection corresponding to the accepted_price
    (i.e. the cumulative volume up to the accepted_price).

    Args:
        bids_df tuple[pd.DataFrame]: tuple of bids dataframes indexed by datetime.
        Must contain 'price', 'volume' and 'accepted_price' columns.
    """


    
    fig, axes = plt.subplots(1, len(bids_dfs), figsize=(7.5*len(bids_dfs), 8))
    b0, *_ = bids_dfs.values()
    hours = sorted(b0.index.unique())

    if only_hours is not None: 
        hours = only_hours
    
    demand, supply = b0[b0["volume"] < 0], b0[b0["volume"] > 0]
    min_bid, max_bid = supply['price'].min(), supply['price'].max()
    max_vol = -1 * demand["volume"].min()

    for t in hours:

        for ax, name in zip(axes, bids_dfs):
            ax.clear()

            plot_df = bids_dfs[name].copy()
            plot_df = plot_df[plot_df["volume"] > 0]
            plot_df["marginal_cost"] = plot_df.groupby("unit_id")["marginal_cost"].ffill().bfill()
            # drop rows without price/volume
            slice_df = plot_df.loc[t]
            intersect_y = slice_df['accepted_price'].unique()[0]
            intersect_x = slice_df["accepted_volume"].sum() 
            
            # sort ascending price and compute cumulative volume
            sort_df = slice_df.sort_values('price')
            sort_df = sort_df.reset_index()
            if only_operators is not None:
                sort_df = sort_df[sort_df["unit_operator"].isin(only_operators)]
            sort_df["cumvol"] = sort_df["volume"].cumsum()
            sort_df["cmap"] = sort_df["technology"].map(color_dict)

            ax_params = {
                'name': name,
                'min_bid': min_bid,
                'max_bid': max_bid,
                'max_vol': max_vol,
                'intersect_x': intersect_x,
                'intersect_y': intersect_y,
            }

            prev_x = 0.0

            for i, row in sort_df.iterrows():
                x0 = prev_x
                x1 = float(row['cumvol'])
                y = float(row['price'])
                op_to_col = {strategic_operator: "tab:red", 
                             "renewables_operator": "tab:blue"}
                col = op_to_col.get(row["unit_operator"], "tab:grey")

                # horizontal segment for this bid
                ax.hlines(y, x0, x1, colors=col)
                ax.hlines(row['marginal_cost'], x0, x1, colors='black', linestyles='dashed')

                if row["unit_operator"] == strategic_operator:                   
                    ax_params['profits'] = ax_params.get('profits', 0) + row["profit"]

                prev_x = x1

                ax.fill_betweenx([0, y], x0, x1, color=row["cmap"], alpha=0.3)
            

            ax = supply_curve_ax(ax, ax_params)
            handles = [Patch(color=c, label=l, alpha=0.3) for l, c in COLOR_DICT.items()]
            fig.suptitle(f'Supply curve @ {pd.to_datetime(t)}')
            fig.subplots_adjust(bottom=0.15)
            fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
            # Refresh the display
        clear_output(wait=True)    
        display(fig)
        time.sleep(time_sleep)
    


def plot_reward_heatmaps(reward_df, marginal_costs, metric="reward", bins=10):
    
    plot_df = reward_df.copy()
    
    for i, mc in enumerate(marginal_costs):
        price = (plot_df[f"actions_{i}"] + 2) * mc 
        plot_df[f"bin_{i}"] = pd.cut(price, bins=bins) 
    
    ncols = len(marginal_costs) - 1
    fig, axes = plt.subplots(ncols=ncols, figsize=(10*ncols,7))
    if ncols == 1: 
        axes = (axes,)
    
    for i, ax in enumerate(axes):
        heatmap_data = (
            plot_df
            .groupby([f'bin_{ncols}',f'bin_{i}'], observed=False)[metric]
            .mean()
            .unstack()
        )
        sns.heatmap(
            heatmap_data,
            cmap='viridis',
            ax=ax,
            #annot=True,
            cbar_kws={'label': f'mean {metric}'}
        )

        Z = heatmap_data.values
        # index of global max (row, col)
        max_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
        max_row, max_col = max_idx
        circle = plt.Circle(
        (max_col + 0.5, max_row + 0.5),  # center of cell
        radius=1,
        fill=False,
        edgecolor='red',
        linewidth=5
        )
        ax.add_patch(circle)
        ax.set_xlabel(f'Bid price {i}')
        ax.set_ylabel(f'Bid price {ncols}')
        ax.invert_yaxis()
        ax.set_title(f"Reward as a function of price {i} and {ncols}")
    
    return fig, axes