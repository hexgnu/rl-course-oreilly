"""Implement a strategy using Q-Learning, or eventually a DQN"""
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

from util import get_data, create_df_benchmark, create_df_trades
import QLearner as ql
from indicators import get_momentum, get_sma_indicator, compute_bollinger_value
from marketsim import compute_portvals_single_symbol, market_simulator
from analysis import get_portfolio_stats

class StrategyLearner(object):
    # Constants for positions and order signals
    LONG = 1
    CASH = 0
    SHORT = -1

    def __init__(self, num_shares=1000, epochs=100, num_steps=10, 
                 impact=0.0, commission=0.00, verbose=False, learner=ql.QLearner(num_states=3000, num_actions=3)):
        """
        
        Instantiate a StrategyLearner that can learn a trading policy.

        num_shares: The number of shares that can be traded in one order
        epochs:     The number of times to train the QLearner
        num_steps:  The number of steps used in getting thresholds for the
        discretization process. It is the number of groups to put data into.
        impact:     The amount the price moves against the trader compared to the
        historical data at each transaction
        commission: The fixed amount in dollars charged for each transaction
        verbose:    If True, print and plot data in add_evidence
        **kwargs:   Arguments for QLearner
        """
        
        self.epochs = epochs
        self.num_steps = num_steps
        self.num_shares = num_shares
        self.impact = impact
        self.commission = commission
        self.verbose = verbose
        self.window_size = 10
        self.q_learner = learner
        # Initialize a QLearner
        # self.q_learner = ql.QLearner(**kwargs)

    def get_features(self, prices):
        """
        Compute the technical features of a position and feed that
        into the Q-Learning process
        
        TODO: Try different parameters to see if you can improve
              the portfolio
              
        Try out different window sizes (right now set to 10 trading days)
        
        prices: Adjusted close prices of the given symbol
        
        df_features: A pandas dataframe of the technical indicators
        """
        
        rolling_mean = prices.rolling(window=self.window_size).mean()
        rolling_std = prices.rolling(window=self.window_size).std()
        
        momentum      = get_momentum(prices, self.window_size)
        sma_indicator = get_sma_indicator(prices, rolling_mean)
        
        bollinger_val = compute_bollinger_value(prices, rolling_mean, rolling_std)
        
        df_features   = pd.concat([momentum, sma_indicator], axis=1)
        df_features   = pd.concat([df_features, bollinger_val], axis=1)
        df_features.columns = ["ind{}".format(i) 
                                for i in range(len(df_features.columns))]
        
        # Perhaps you can come up with a different feature?
        # Try it!
        
        df_features.dropna(inplace=True)
        return df_features

    def get_thresholds(self, df_features, num_steps):
        """
        Compute the thresholds to be used in the discretization of features.
        thresholds is a 2-d numpy array where the first dimesion indicates the 
        indices of features in df_features and the second dimension refers to 
        the value of a feature at a particular threshold.
        """
        
        step_size = round(df_features.shape[0] / num_steps)
        df_copy = df_features.copy()
        thres = np.zeros(shape=(df_features.shape[1], num_steps))
        for i, feat in enumerate(df_features.columns):
            df_copy.sort_values(by=[feat], inplace=True)
            for step in range(num_steps):
                if step < num_steps - 1:
                    thres[i, step] = df_copy[feat].iloc[(step + 1) * step_size]
                # The last threshold must be = the largest value in df_copy
                else:
                    thres[i, step] = df_copy[feat].iloc[-1]
        return thres

    def discretize(self, df_features, non_neg_position, thresholds):
        """
        Discretize features and return a state.

        Parameters:
        df_features: The technical indicators to be discretized. They were  
        computed in get_features()
        non_neg_position: The position at the beginning of a particular day,
        before taking any action on that day. It is >= 0 so that state >= 0

        Returns:
        state: A state in the Q-table from which we will query for an action.
        It indicates an index of the first dimension in the Q-table
        """
        state = non_neg_position * pow(self.num_steps, len(df_features))
        for i in range(len(df_features)):
            thres = thresholds[i][thresholds[i] >= df_features[i]][0]
            thres_i = np.where(thresholds == thres)[1][0]
            state += thres_i * pow(self.num_steps, i)
        return state

    def get_position(self, old_pos, signal):
        """Find a new position based on the old position and the given signal.
        signal = action - 1; action is a result of querying a state, which was
        computed in discretize(), in the Q-table. An action is 0, 1 or 2. It is
        an index of the second dimension in the Q-table. We have to subtract 1
        from action to get a signal of -1, 0 or 1 (short, cash or long).
        """
        new_pos = self.CASH
        # If old_pos is not long and signal is to buy, new_pos will be long
        if old_pos < self.LONG and signal == self.LONG:
            new_pos = self.LONG
        # If old_pos is not short and signal is to sell, new_pos will be short
        elif old_pos > self.SHORT and signal == self.SHORT:
            new_pos = self.SHORT
        return new_pos

    def get_daily_reward(self, prev_price, curr_price, position):
        """Calculate the daily reward as a percentage change in prices: 
        - Position is long: if the price goes up (curr_price > prev_price),
          we get a positive reward; otherwise, we get a negative reward
        - Position is short: if the price goes down, we get a positive reward;
        otherwise, we a negative reward
        - Position is cash: we get no reward
        """
        return position * ((curr_price / prev_price) - 1)

    def has_converged(self, cum_returns, patience=10):
        """Check if the cumulative returns have converged.

        Paramters:
        cum_returns: A list of cumulative returns for respective epochs
        patience: The number of epochs with no improvement in cum_returns

        Returns: True if converged, False otherwise
        """
        # The number of epochs should be at least patience before checking
        # for convergence
        if patience > len(cum_returns):
            return False
        latest_returns = cum_returns[-patience:]
        # If all the latest returns are the same, return True
        if len(set(latest_returns)) == 1:
            return True
        max_return = max(cum_returns)
        if max_return in latest_returns:
            # If one of recent returns improves, not yet converged
            if max_return not in cum_returns[:len(cum_returns) - patience]:
                return False
            else:
                return True
        # If none of recent returns is greater than max_return, it has converged
        return True

    def add_evidence(self, symbol="IBM", start_date=dt.datetime(2008,1,1),
        end_date=dt.datetime(2009,12,31), start_val = 10000):
        """Create a QLearner, and train it for trading.

        Parameters:
        symbol: The stock symbol to act on
        start_date: A datetime object that represents the start date
        end_date: A datetime object that represents the end date
        start_val: Start value of the portfolio which contains only the symbol
        """
        dates = pd.date_range(start_date, end_date)
        # Get adjusted close prices for symbol
        df_prices = get_data([symbol], dates)
        # Get features and thresholds
        df_features = self.get_features(df_prices[symbol])
        thresholds = self.get_thresholds(df_features, self.num_steps)
        cum_returns = []
        for epoch in range(1, self.epochs + 1):
            # Initial position is holding nothing
            position = self.CASH
            # Create a series that captures order signals based on actions taken
            orders = pd.Series(index=df_features.index)
            # Iterate over the data by date

            for day, date in enumerate(df_features.index):
                # Get a state; add 1 to position so that states >= 0
                state = self.discretize(df_features.loc[date], 
                                        position + 1, thresholds)
                # On the first day, get an action without updating the Q-table
                if date == df_features.index[0]:
                    # Get the first action based on nothing
                    # action = self.q_learner.act(state)
                    action = self.q_learner.act(state, 0.0, update=False)

                # On other days, calculate the reward and update the Q-table
                else:
                    prev_price = df_prices[symbol].iloc[day-1]
                    curr_price = df_prices[symbol].loc[date]
                    reward = self.get_daily_reward(prev_price, 
                                                   curr_price, position)
                    action = self.q_learner.act(state, reward, update=True, done=date==df_features.index[-1])
                # On the last day, close any open positions
                if date == df_features.index[-1]:
                    new_pos = -position
                else:
                    new_pos = self.get_position(position, action - 1)

                # Add new_pos to orders
                orders.loc[date] = new_pos
                # Update current position
                position += new_pos
            
            self.q_learner.replay(batch_size=32)

            df_trades = create_df_trades(orders, symbol, self.num_shares)
            portvals = compute_portvals_single_symbol(df_orders=df_trades, 
                                                      symbol=symbol, 
                                                      start_val=start_val, 
                                                      commission=self.commission,
                                                      impact=self.impact)
            cum_return = get_portfolio_stats(portvals)[0]
            cum_returns.append(cum_return)
            if self.verbose: 
                print (epoch, cum_return)
            # Check for convergence after running for at least 20 epochs
            if epoch > 20:
                # Stop if the cum_return doesn't improve for 10 epochs
                if self.has_converged(cum_returns):
                    break
        if self.verbose:
            plt.plot(cum_returns)
            plt.xlabel("Epoch")
            plt.ylabel("Cumulative return (%)")
            plt.show()

    def test_policy(self, symbol="IBM", start_date=dt.datetime(2010,1,1),
        end_date=dt.datetime(2011,12,31), start_val=10000):
        """Use the existing policy and test it against new data.

        Parameters:
        symbol: The stock symbol to act on
        start_date: A datetime object that represents the start date
        end_date: A datetime object that represents the end date
        start_val: Start value of the portfolio which contains only the symbol
        
        Returns:
        df_trades: A dataframe whose values represent trades for each day: 
        +1000 indicating a BUY of 1000 shares, and -1000 indicating a SELL of 
        1000 shares
        """

        dates = pd.date_range(start_date, end_date)
        # Get adjusted close pricess for symbol
        df_prices = get_data([symbol], dates)
        # Get features and thresholds
        df_features = self.get_features(df_prices[symbol])
        thresholds = self.get_thresholds(df_features, self.num_steps)
        # Initial position is holding nothing
        position = self.CASH
        # Create a series that captures order signals based on actions taken
        orders = pd.Series(index=df_features.index)
        # Iterate over the data by date
        for date in df_features.index:
            # Get a state; add 1 to position so that states >= 0
            state = self.discretize(df_features.loc[date], 
                                    position + 1, thresholds)
            action = self.q_learner.act(state, 0.0, update=False)
            # On the last day, close any open positions
            if date == df_features.index[-1]:
                new_pos = -position
            else:
                new_pos = self.get_position(position, action - 1)
            # Add new_pos to orders
            orders.loc[date] = new_pos
            # Update current position
            position += new_pos
        # Create a trade dataframe
        df_trades = create_df_trades(orders, symbol, self.num_shares)
        return df_trades
