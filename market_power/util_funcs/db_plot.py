import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from IPython.display import clear_output, display
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib as mpl
import yaml

try:
    with open("util_funcs/plot_config.yaml", "r") as f:
    
        config = yaml.safe_load(f)

    mpl.rcParams.update(config)

except Exception as e: 
    print("Loading plot config failed")



COLOR_DICT = {'open cycle gas turbine': 'olive', 
              'oil': 'red', 
              'lignite': 'purple', 
              'nuclear': 'grey', 
              'hard coal': 'brown',
              'combined cycle gas turbine': 'olive', 
              'wind_offshore': 'blue', 
              'hydro': 'cyan',
              'wind_onshore': 'blue', 
              'biomass': 'purple', 
              'solar': 'gold'}



def supply_curve_ax(ax:plt.Axes, ax_params:dict):
            # standard supply curve: volume on x, price on y (step)
    #ax.step(np.concatenate(([0], cum_vol)), np.concatenate(([prices[0]], prices)), where='post', linewidth=2)
    # ax.set_xlabel('Cumulative Volume')
    # ax.set_ylabel('Price')
    #ax.set_xlim(left=0, right=ax_params["max_vol"] + 10000)
    ax.set_ylim(bottom=0, 
                top=ax_params["max_bid"] + 5)

    # draw vertical line at that cumulative volume and annotate accepted price
    x, y = ax_params["intersect_x"], ax_params["intersect_y"]
    ax.axvline(x=x, color='red', linestyle='--', 
               linewidth=3, label=f'accepted_price={y:.2f}')
    ax.plot([x], [y], 'ro')  # marker at intersection
    ax.annotate(f"{y:.1f} EUR/MWh", (x,y), (x-1000, y+0.5), color="black")

    title = ax_params['name'].capitalize()

    if 'profits' in ax_params:
       title+= f" profits: {ax_params['profits']:,.1f} EUR"
    
    ax.set_title(title)

    return ax    



def plot_supply_curves(bids_dfs:dict,
                       nrows:int=1,  
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

    ncols = int(np.ceil(len(bids_dfs) / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10*ncols, 8*nrows), tight_layout=True)
    b0, *_ = bids_dfs.values()
    hours = sorted(b0.index.unique())

    if only_hours is not None: 
        hours = only_hours
    
    demand, supply = b0[b0["volume"] < 0], b0[b0["volume"] > 0]
    min_bid, max_bid = supply['price'].min(), supply['price'].max()
    max_vol = -1 * demand["volume"].min() 
    
    #labels = set(b0["technology"].dropna())
    #handles = [Patch(color=color_dict[l], label=l.replace("_", " ").capitalize(), alpha=0.3) for l in labels]
    fig.supxlabel("Cumulative volume", y=0.08)
    fig.supylabel("Price", x=0.05)
    fig.subplots_adjust(bottom=0.22)

    labels = b0[b0["unit_operator"] == strategic_operator]["technology"].drop_duplicates()
    handles = [
        *[Patch(color=color_dict[l], label=f"Unit {j+1} ({l})", alpha=0.3) for j,l in enumerate(labels)],
        Patch(color="grey", label="Non-strat. units", alpha=0.3),
        Line2D([0], [0], color='red', ls='--', lw=3, label='Demand')
    ]
    #_ = fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))

    for t in hours:
        #fig.suptitle(f'Supply curve @ {pd.to_datetime(t)}')
        for n, name in enumerate(bids_dfs):            
            i, j = n // ncols, n % ncols

            ax = axes[n] if nrows == 1 or ncols == 1 else axes[i,j]
            ax.clear()
            ax.set_axisbelow(True)
            ax.grid(True)

            try:
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

                ax_params = {
                    'name': name,
                    'min_bid': min_bid,
                    'max_bid': max_bid,
                    'max_vol': max_vol,
                    'intersect_x': intersect_x,
                    'intersect_y': intersect_y,
                }

                prev_x = 0.0

                for _, row in sort_df.iterrows():
                    x0 = prev_x
                    x1 = float(row['cumvol'])
                    y0 = float(row["marginal_cost"])
                    y1 = float(row['price'])
                    label = row['technology']
                    color = "grey"

                    if row["unit_operator"] == strategic_operator:                   
                        ax_params['profits'] = ax_params.get('profits', 0) + row["profit"]
                        color = color_dict[label]

                    # horizontal segment for this bid
                    ax.hlines(y1, x0, x1, colors='black')
                    ax.fill_betweenx([0, y1], x0, x1, color=color, alpha=0.3)

                    if y0 != 0 and y0 < y1:
                        ax.hlines(y0, x0, x1, colors='black', ls="dashed")
                        ax.fill_betweenx([y0, y1], x0, x1, color=color)

                    prev_x = x1

                ax = supply_curve_ax(ax, ax_params)
                
                if i != nrows - 1:
                    ax.set_xticklabels("")
                if j != 0:
                    ax.set_yticklabels("")
                if n == 0: 
                    ax.legend(handles=handles)


            
            except Exception as e: 
                print(f"Exception at time {t}: {e}")

            # Refresh the display
        clear_output(wait=True)    
        display(fig)
        time.sleep(time_sleep)
        
    
    return fig, ax
    





