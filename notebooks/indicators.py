"""Implement technical indicators"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import datetime as dt
from util import get_exchange_days, get_data, normalize_data


def get_momentum(price, window=5):
    """Calculate momentum indicator: 
    momentum[t] = (price[t]/price[t-window]) - 1

    Parameters:
    price: Price, typically adjusted close price, series of a symbol
    window: Number of days to look back
    
    Returns: Momentum, series of the same size as input data
    """    
    momentum = pd.Series(np.nan, index=price.index)
    momentum.iloc[window:] = price.iloc[window:] / price.values[:-window] - 1
    return momentum

def get_sma_indicator(price, rolling_mean):
    """Calculate simple moving average indicator, i.e. price / rolling_mean.

    Parameters:
    price: Price, typically adjusted close price, series of a symbol
    rolling_mean: Rolling mean of a series

    Returns: The simple moving average indicator
    """
    return price / rolling_mean - 1

def get_bollinger_bands(rolling_mean, rolling_std, num_std=2):
    """Calculate upper and lower Bollinger Bands.

    Parameters:
    rolling_mean: Rolling mean of a series
    rolling_meanstd: Rolling std of a series
    num_std: Number of standard deviations for the bands

    Returns: Bollinger upper band and lower band
    """
    upper_band = rolling_mean + rolling_std * num_std
    lower_band = rolling_mean - rolling_std * num_std
    return upper_band, lower_band

def compute_bollinger_value(price, rolling_mean, rolling_std):
    """Output a value indicating how many standard deviations 
    a price is from the mean.

    Parameters:
    price: Price, typically adjusted close price, series of a symbol
    rolling_mean: Rolling mean of a series
    rolling_std: Rolling std of a series

    Returns:
    bollinger_val: the number of standard deviations a price is from the mean
    """

    bollinger_val = (price - rolling_mean) / rolling_std
    return bollinger_val

def plot_momentum(sym_price, sym_mom, title="Momentum Indicator",
                  fig_size=(12, 6)):
    """Plot momentum and prices for a symbol.

    Parameters:
    sym_price: Price, typically adjusted close price, series of symbol
    sym_mom: Momentum of symbol
    fig_size: Width and height of the chart in inches
    
    Returns:
    Plot momentum and prices on the sample plot with two scales
    """
    # Create two subplots on the same axes with different left and right scales
    fig, ax1 = plt.subplots()

    # The first subplot with the left scale: prices
    ax1.grid(linestyle='--')
    line1 = ax1.plot(sym_price.index, sym_price, label="Adjusted Close Price",
                     color="b")
    ax1.set_xlabel("Date")
    # Make the y-axis label, ticks and tick labels match the line color
    ax1.set_ylabel("Adjusted Close Price", color="b")
    ax1.tick_params("y", colors="b")

    # The second subplot with the right scale: momentum
    ax2 = ax1.twinx()
    line2 = ax2.plot(sym_mom.index, sym_mom, label="Momentum", color="k",
                     alpha=0.4)
    ax2.set_ylabel("Momentum", color="k")
    ax2.tick_params("y", colors="k")

    # Align gridlines for the two scales
    align_y_axis(ax1, ax2, .1, .1)

    # Show legend with all labels on the same legend
    lines = line1 + line2
    line_labels = [l.get_label() for l in lines]
    ax1.legend(lines, line_labels, loc="upper center")

    #Set figure size
    fig = plt.gcf()
    fig.set_size_inches(fig_size)

    plt.suptitle(title)
    plt.show()

def plot_sma_indicator(sym_price, sma_indicator, rolling_mean, 
                       title="SMA Indicator", fig_size=(12, 6)):
    """Plot SMA indicator, price and rolling_mean for a symbol.

    Parameters:
    sym_price: Price, typically adjusted close price, series of symbol
    sma_indicator: The simple moving average indicator
    rolling_mean (a.k.a SMA): Rolling mean of sym_price
    title: The chart title
    fig_size: Width and height of the chart in inches

    Returns:
    Plot all the three series on the same plot with two scales
    """
    # Create two subplots on the same axes with different left and right scales
    fig, ax1 = plt.subplots()

    # The first subplot with the left scale: prices
    ax1.grid(linestyle='--')
    line1 = ax1.plot(sym_price.index, sym_price, label="Adjusted Close Price",
                     color="b")
    line2 = ax1.plot(rolling_mean.index, rolling_mean, label="SMA", color="g")
    ax1.set_xlabel("Date")
    # Make the y-axis label, ticks and tick labels match the line color
    ax1.set_ylabel("Adjusted Close Price", color="b")
    ax1.tick_params("y", colors="b")

    # The second subplot with the right scale: momentum
    ax2 = ax1.twinx()
    line3 = ax2.plot(sma_indicator.index, sma_indicator, 
        label="SMA Indicator", color="k", alpha=0.4)
    ax2.set_ylabel("SMA indicator", color="k")
    ax2.tick_params("y", colors="k")

    # Align gridlines for the two scales
    align_y_axis(ax1, ax2, .1, .1)

    # Show legend with all labels on the same legend
    lines = line1 + line2 + line3
    line_labels = [l.get_label() for l in lines]
    ax1.legend(lines, line_labels, loc="upper center")

    #Set figure size
    fig = plt.gcf()
    fig.set_size_inches(fig_size)

    plt.title(title)
    plt.show()

def plot_bollinger(sym_price, upper_band, lower_band, bollinger_val, 
                   num_std=1, title="Bollinger Indicator", fig_size=(12, 6)):
    """Plot Bollinger bands and value for a symbol.

    Parameters:
    sym_price: Price, typically adjusted close price, series of symbol
    upper_band: Bollinger upper band
    lower_band: Bollinger lower band
    bollinger_val: The number of standard deviations a price is from the mean
    num_std: Number of standard deviations for the bands
    fig_size: Width and height of the chart in inches

    Returns:
    Plot two subplots, one for the Adjusted Close Price and Bollinger bands,
    the other for the Bollinger value
    """
    # Create 2 subplots
    # Plot symbol's adjusted close price, rolling mean and Bollinger Bands
    f, ax = plt.subplots(2, sharex=True)
    ax[0].fill_between(upper_band.index, upper_band, lower_band, color="gray",
                       alpha=0.4, linewidth=2.0,
                       label="Region btwn Bollinger Bands")
    ax[0].plot(sym_price, label="Adjusted Close Price", color="b")
    ax[0].set_ylabel("Adjusted Close Price")
    ax[0].legend(loc="upper center")

    # Plot the bollinger value
    ax[1].axhspan(-num_std, num_std, color="gray", alpha=0.4, linewidth=2.0,
        label="Region btwn {} & {} std".format(-num_std, num_std))
    ax[1].plot(bollinger_val, label="Bollinger Value", color="b")
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("Bollinger Value")
    ax[1].set_xlim(bollinger_val.index.min(), bollinger_val.index.max())
    ax[1].legend(loc="upper center")

    #Set figure size
    fig = plt.gcf()
    fig.set_size_inches(fig_size)

    plt.suptitle(title)
    plt.show()

def align_y_axis(ax1, ax2, minresax1, minresax2):
    """Set tick marks of twinx axes to line up with 7 total tick marks.

    ax1 and ax2 are matplotlib axes
    Spacing between tick marks will be a factor of minresax1 and minresax2
    from https://stackoverflow.com/questions/26752464/how-do-i-align-gridlines
    -for-two-y-axis-scales-using-matplotlib
    """

    ax1ylims = ax1.get_ybound()
    ax2ylims = ax2.get_ybound()
    ax1factor = minresax1 * 6
    ax2factor = minresax2 * 6
    ax1.set_yticks(np.linspace(ax1ylims[0],
                               ax1ylims[1] + (ax1factor -
                               (ax1ylims[1] - ax1ylims[0]) % ax1factor) %
                               ax1factor, 7))
    ax2.set_yticks(np.linspace(ax2ylims[0],
                               ax2ylims[1] + (ax2factor -
                               (ax2ylims[1] - ax2ylims[0]) % ax2factor) %
                               ax2factor, 7))


if __name__ == "__main__":
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    # Get NYSE trading dates
    dates = get_exchange_days(start_date, end_date, 
                              dirpath="../data/dates_lists", 
                              filename="NYSE_dates.txt")

    symbols = ["AAPL"]
    # Get stock data and normalize it
    df_price = get_data(symbols, dates)
    norm_price = normalize_data(df_price)
    window = 20
    num_std = 2

    for symbol in symbols:
        # Compute rolling mean
        rolling_mean = norm_price[symbol].rolling(window=window).mean()

        # Compute rolling standard deviation
        rolling_std = norm_price[symbol].rolling(window=window).std()

        # Get momentum
        momentum = get_momentum(norm_price[symbol], window)

        # Plot momentum
        plot_momentum(norm_price[symbol], momentum, 
            "Momentum Indicator for {} with lookback={} days \n \
            Prices are normalized to the first date)".format(symbol, window))

        # Get SMA indicator
        sma_indicator = get_sma_indicator(norm_price[symbol], rolling_mean)

        # Plot SMA indicator
        plot_sma_indicator(norm_price[symbol], sma_indicator, rolling_mean, 
            "SMA Indicator for {} with lookback={} days \n \
            (Prices are normalized to the first date)".format(symbol, window))

        # Compute Bollinger bands and value
        upper_band, lower_band = get_bollinger_bands(rolling_mean, rolling_std,
                                                     num_std)
        bollinger_val = compute_bollinger_value(norm_price[symbol], 
                                                rolling_mean, rolling_std)

        # Plot Bollinger bands and values
        plot_bollinger(norm_price[symbol], upper_band, lower_band, 
            bollinger_val, num_std, 
            "Bollinger Indicator for {} with num_std={}, lookback={} days \
            \n(Prices are normalized to the first date)".
            format(symbol, num_std, window))
