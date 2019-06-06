"""Implement a market simulator that processes a dataframe instead of 
a csv file.
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from analysis import get_portfolio_value, get_portfolio_stats, \
plot_normalized_data
from util import get_data, normalize_data


def compute_portvals_single_symbol(df_orders, symbol, start_val=1000000, 
    commission=9.95, impact=0.005):
    """Compute portfolio values for a single symbol.

    Parameters:
    df_orders: A dataframe with orders for buying or selling stocks. There is
    no value for cash (i.e. 0).
    symbol: The stock symbol whose portfolio values need to be computed
    start_val: The starting value of the portfolio (initial cash available)
    commission: The fixed amount in dollars charged for each transaction
    impact: The amount the price moves against the trader compared to the 
    historical data at each transaction
    
    Returns:
    portvals: A dataframe with one column containing the value of the portfolio
    for each trading day
    """

    # Sort the orders dataframe by date
    df_orders.sort_index(ascending=True, inplace=True)
    
    # Get the start and end dates
    start_date = df_orders.index.min()
    end_date = df_orders.index.max()

    # Create a dataframe with adjusted close prices for the symbol and for cash
    df_prices = get_data([symbol], pd.date_range(start_date, end_date))

    if df_prices.shape[1] > 1:
        del df_prices["SPY"]
    df_prices["cash"] = 1.0

    # Fill NAN values if any
    df_prices.fillna(method="ffill", inplace=True)
    df_prices.fillna(method="bfill", inplace=True)
    df_prices.fillna(1.0, inplace=True)

    # Create a dataframe that represents changes in the number of shares by day
    df_trades = pd.DataFrame(np.zeros((df_prices.shape)), df_prices.index, 
        df_prices.columns)
    for index, row in df_orders.iterrows():
        # Total value of shares purchased or sold
        traded_share_value = df_prices.loc[index, symbol] * row["Shares"]
        # Transaction cost 
        transaction_cost = commission + impact * df_prices.loc[index, symbol] \
                            * abs(row["Shares"])

        # Update the number of shares and cash based on the type of transaction
        # Note: The same asset may be traded more than once on a particular day
        # If the shares were bought
        if row["Shares"] > 0:
            df_trades.loc[index, symbol] = df_trades.loc[index, symbol] \
                                            + row["Shares"]
            df_trades.loc[index, "cash"] = df_trades.loc[index, "cash"] \
                                            - traded_share_value \
                                            - transaction_cost
        # If the shares were sold
        elif row["Shares"] < 0:
            df_trades.loc[index, symbol] = df_trades.loc[index, symbol] \
                                            + row["Shares"]
            df_trades.loc[index, "cash"] = df_trades.loc[index, "cash"] \
                                            - traded_share_value \
                                            - transaction_cost
    # Create a dataframe that represents on each particular day how much of
    # each asset in the portfolio
    df_holdings = pd.DataFrame(np.zeros((df_prices.shape)), df_prices.index, 
        df_prices.columns)
    for row_count in range(len(df_holdings)):
        # In the first row, the number shares are the same as in df_trades, 
        # but start_val must be added to cash
        if row_count == 0:
            df_holdings.iloc[0, :-1] = df_trades.iloc[0, :-1].copy()
            df_holdings.iloc[0, -1] = df_trades.iloc[0, -1] + start_val
        # The rest of the rows show cumulative values
        else:
            df_holdings.iloc[row_count] = df_holdings.iloc[row_count-1] \
                                            + df_trades.iloc[row_count]
        row_count += 1

    # Create a dataframe that represents the monetary value of each asset 
    df_value = df_prices * df_holdings
    
    # Create portvals dataframe
    portvals = pd.DataFrame(df_value.sum(axis=1), df_value.index, ["port_val"])
    return portvals

def market_simulator(df_orders, df_orders_benchmark, symbol, start_val=1000000,
    commission=9.95, impact=0.005, daily_rf=0.0, samples_per_year=252.0, 
    save_fig=False, fig_name="plot.png"):
    """
    This function takes in and executes trades from orders dataframes

    Parameters:
    df_orders: A dataframe that contains portfolio orders
    df_orders_benchmark: A dataframe that contains benchmark orders
    start_val: The starting cash in dollars
    commission: The fixed amount in dollars charged for each transaction
    impact: The amount the price moves against the trader compared to the 
    historical data at each transaction
    daily_rf: Daily risk-free rate, assuming it does not change
    samples_per_year: Sampling frequency per year
    save_fig: Whether to save the plot or not
    fig_name: The name of the saved figure

    Returns:
    Print out final portfolio value of the portfolio, Sharpe ratio, cumulative
    return, average daily return and standard deviation of the portfolio and 
    Benchmark. Plot a chart of the portfolio and benchmark performances.
    """    
    # Process portfolio orders
    portvals = compute_portvals_single_symbol(df_orders=df_orders, symbol=symbol,
        start_val=start_val, commission=commission, impact=impact)

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(
        portvals, daily_rf=daily_rf, samples_per_year=samples_per_year)
    
    # Process benchmark orders
    portvals_bm = compute_portvals_single_symbol(df_orders=df_orders_benchmark, 
        symbol=symbol, start_val=start_val, commission=commission, impact=impact)
    
    # Get benchmark stats
    cum_ret_bm, avg_daily_ret_bm, std_daily_ret_bm, sharpe_ratio_bm = \
    get_portfolio_stats(portvals_bm, daily_rf=daily_rf, 
        samples_per_year=samples_per_year)

    # Compare portfolio against Benchmark
    print ("Sharpe Ratio of Portfolio: {}".format(sharpe_ratio))
    print ("Sharpe Ratio of Benchmark : {}".format(sharpe_ratio_bm))
    print ()
    print ("Cumulative Return of Portfolio: {}".format(cum_ret))
    print ("Cumulative Return of Benchmark : {}".format(cum_ret_bm))
    print ()
    print ("Standard Deviation of Portfolio: {}".format(std_daily_ret))
    print ("Standard Deviation of Benchmark : {}".format(std_daily_ret_bm))
    print ()
    print ("Average Daily Return of Portfolio: {}".format(avg_daily_ret))
    print ("Average Daily Return of Benchmark : {}".format(avg_daily_ret_bm))
    print ()
    print ("Final Portfolio Value: {}".format(portvals.iloc[-1, -1]))
    print ("Final Benchmark Value: {}".format(portvals_bm.iloc[-1, -1]))

    # Rename columns and normalize data to the first date of the date range
    portvals.rename(columns={"port_val": "Portfolio"}, inplace=True)
    portvals_bm.rename(columns={"port_val": "Benchmark"}, inplace=True)
    plot_norm_data_vertical_lines(df_orders, portvals, portvals_bm,
        save_fig=False, fig_name="plot.png")

def plot_norm_data_vertical_lines(df_orders, portvals, portvals_bm, 
    plot_vertical_lines=False, save_fig=False, fig_name="plot.png"):
    """Plots portvals and portvals_bm, showing vertical lines for orderss
    
    Parameters:
    df_orders: A dataframe that contains portfolio orders
    portvals: A dataframe with one column containing daily portfolio value
    portvals_bm: A dataframe with one column containing daily benchmark value
    save_fig: Whether to save the plot or not
    fig_name: The name of the saved figure

    Returns: Plot a chart of the portfolio and benchmark performances
    """
    # Normalize data
    portvals = normalize_data(portvals)
    portvals_bm = normalize_data(portvals_bm)
    df = portvals_bm.join(portvals)

    # Plot the normalized benchmark and portfolio
    plt.plot(df.loc[:, "Benchmark"], label="Benchmark")
    plt.plot(df.loc[:, "Portfolio"], label="Portfolio")

    # Plot the vertical lines for buy and sell signals
    if plot_vertical_lines == True:
        for date in df_orders.index:
            if df_orders.loc[date, "Shares"] > 0:
                plt.axvline(date, color = 'g', linestyle = '--')
            elif df_orders.loc[date, "Shares"] < 0:
                plt.axvline(date, color = 'r', linestyle = '--')

    plt.title("Portfolio vs. Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Normalized prices")
    plt.legend()

    # Set figure size
    fig = plt.gcf()
    fig.set_size_inches(12, 6)

    if save_fig == True:
        plt.savefig(fig_name)
    else:
        plt.show()
