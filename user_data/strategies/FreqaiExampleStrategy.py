import logging
import numpy as np
from functools import reduce

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class FreqaiExampleStrategy(IStrategy):
    """
    Production-ready FreqAI strategy with comprehensive feature engineering.
    
    This strategy demonstrates:
    - Advanced feature engineering with automatic expansion
    - Multiple target definitions for regression
    - Comprehensive outlier detection and handling
    - Dynamic threshold calculation using z-scores
    - Production-ready logging and monitoring
    
    Features:
    - Technical indicators (RSI, MFI, ADX, EMA, SMA, Bollinger Bands)
    - Volume analysis and price momentum
    - Time-based features (day of week, hour of day)
    - Correlation pair features
    - Multi-timeframe analysis
    - Shifted candle features for historical context
    
    Targets:
    - Price movement prediction (24 candles ahead)
    - Volatility prediction
    - Trend direction classification
    """

    minimal_roi = {"0": 0.1, "240": -1}
    stoploss = -0.05
    use_exit_signal = True
    can_short = False
    process_only_new_candles = True
    
    # Maximum startup candle count - must be >= max(indicator_periods_candles)
    # This ensures sufficient data for all indicators
    startup_candle_count: int = 100

    plot_config = {
        "main_plot": {
            "close": {"color": "blue"},
            "&-s_close": {"color": "green"},
            "&-s_close_mean": {"color": "orange"},
            "&-s_close_std": {"color": "red"}
        },
        "subplots": {
            "Features": {
                "%-rsi-10": {"color": "purple"},
                "%-rsi-20": {"color": "blue"},
                "%-rsi-50": {"color": "green"},
                "%-bb_width-10": {"color": "orange"},
                "%-bb_width-20": {"color": "red"},
                "%-bb_width-50": {"color": "brown"}
            },
            "Predictions": {
                "do_predict": {"color": "brown"},
                "&-s_close": {"color": "blue"},
                "&-s_volatility": {"color": "green"},
                "&-s_trend": {"color": "purple"}
            },
            "Quality": {
                "DI_values": {"color": "red"},
                "&-s_close_mean": {"color": "orange"},
                "&-s_close_std": {"color": "yellow"}
            }
        },
    }

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Feature engineering function that automatically expands features across:
        - indicator_periods_candles (10, 20, 50)
        - include_timeframes (5m, 15m, 1h, 4h)
        - include_shifted_candles (3)
        - include_corr_pairlist (BTC, ETH, BNB)
        
        Total features per indicator: 3 * 4 * 3 * 3 = 108 features
        
        :param dataframe: Strategy dataframe
        :param period: Current indicator period (10, 20, or 50)
        :param metadata: Pair and timeframe metadata
        :return: DataFrame with added features
        """
        
        # Momentum indicators
        dataframe[f"%-rsi-{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-mfi-{period}"] = ta.MFI(dataframe, timeperiod=period)
        dataframe[f"%-adx-{period}"] = ta.ADX(dataframe, timeperiod=period)
        dataframe[f"%-roc-{period}"] = ta.ROC(dataframe, timeperiod=period)
        dataframe[f"%-willr-{period}"] = ta.WILLR(dataframe, timeperiod=period)
        
        # Moving averages
        dataframe[f"%-sma-{period}"] = ta.SMA(dataframe, timeperiod=period)
        dataframe[f"%-ema-{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe[f"%-wma-{period}"] = ta.WMA(dataframe, timeperiod=period)
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe[f"%-bb_width-{period}"] = (
            bollinger["upper"] - bollinger["lower"]
        ) / bollinger["mid"]
        dataframe[f"%-bb_position-{period}"] = (
            dataframe["close"] - bollinger["lower"]
        ) / (bollinger["upper"] - bollinger["lower"])
        dataframe[f"%-bb_squeeze-{period}"] = (
            bollinger["upper"] - bollinger["lower"]
        ) / dataframe[f"%-sma-{period}"]
        
        # Volume indicators
        dataframe[f"%-volume_sma_ratio-{period}"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )
        dataframe[f"%-volume_ema_ratio-{period}"] = (
            dataframe["volume"] / dataframe["volume"].ewm(span=period).mean()
        )
        
        # Price momentum
        dataframe[f"%-price_change-{period}"] = dataframe["close"].pct_change(period)
        dataframe[f"%-price_volatility-{period}"] = (
            dataframe["close"].rolling(period).std() / dataframe["close"].rolling(period).mean()
        )
        
        # Stochastic oscillator
        stoch = ta.STOCH(dataframe, fastk_period=period, slowk_period=3, slowd_period=3)
        dataframe[f"%-stoch_k-{period}"] = stoch["slowk"]
        dataframe[f"%-stoch_d-{period}"] = stoch["slowd"]
        
        # MACD
        macd = ta.MACD(dataframe, fastperiod=period//2, slowperiod=period, signalperiod=period//3)
        dataframe[f"%-macd-{period}"] = macd["macd"]
        dataframe[f"%-macd_signal-{period}"] = macd["macdsignal"]
        dataframe[f"%-macd_hist-{period}"] = macd["macdhist"]
        
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Basic feature engineering that expands across timeframes and correlation pairs
        but NOT across indicator periods (for features that don't need period variation)
        
        :param dataframe: Strategy dataframe
        :param metadata: Pair and timeframe metadata
        :return: DataFrame with added features
        """
        
        # Raw price and volume features
        dataframe["%-raw_price"] = dataframe["close"]
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]
        
        # Price change features
        dataframe["%-pct_change"] = dataframe["close"].pct_change()
        dataframe["%-pct_change_2"] = dataframe["close"].pct_change(2)
        dataframe["%-pct_change_3"] = dataframe["close"].pct_change(3)
        
        # Long-term moving averages (fixed periods)
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["%-sma-200"] = ta.SMA(dataframe, timeperiod=200)
        dataframe["%-ema-50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["%-sma-50"] = ta.SMA(dataframe, timeperiod=50)
        
        # Volume features
        dataframe["%-volume_ma_ratio"] = (
            dataframe["volume"] / dataframe["volume"].rolling(20).mean()
        )
        dataframe["%-volume_std_ratio"] = (
            dataframe["volume"] / dataframe["volume"].rolling(20).std()
        )
        
        # Price position relative to long-term averages
        dataframe["%-price_vs_ema200"] = dataframe["close"] / dataframe["%-ema-200"] - 1
        dataframe["%-price_vs_sma200"] = dataframe["close"] / dataframe["%-sma-200"] - 1
        dataframe["%-price_vs_ema50"] = dataframe["close"] / dataframe["%-ema-50"] - 1
        
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Standard feature engineering - called once with base timeframe dataframe.
        Features here are NOT auto-expanded and are good for time-based features.
        
        :param dataframe: Strategy dataframe
        :param metadata: Pair metadata
        :return: DataFrame with added features
        """
        
        # Time-based features (normalized)
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25
        dataframe["%-minute_of_hour"] = (dataframe["date"].dt.minute + 1) / 61
        
        # Market session features
        dataframe["%-is_weekend"] = (dataframe["date"].dt.dayofweek >= 5).astype(int)
        dataframe["%-is_london_session"] = (
            (dataframe["date"].dt.hour >= 8) & (dataframe["date"].dt.hour < 16)
        ).astype(int)
        dataframe["%-is_ny_session"] = (
            (dataframe["date"].dt.hour >= 13) & (dataframe["date"].dt.hour < 21)
        ).astype(int)
        dataframe["%-is_asia_session"] = (
            (dataframe["date"].dt.hour >= 0) & (dataframe["date"].dt.hour < 8)
        ).astype(int)
        
        # Candle pattern features
        dataframe["%-doji"] = ta.CDLDOJI(dataframe)
        dataframe["%-hammer"] = ta.CDLHAMMER(dataframe)
        dataframe["%-engulfing"] = ta.CDLENGULFING(dataframe)
        dataframe["%-morning_star"] = ta.CDLMORNINGSTAR(dataframe)
        dataframe["%-evening_star"] = ta.CDLEVENINGSTAR(dataframe)
        
        # Market structure features
        dataframe["%-higher_high"] = (
            (dataframe["high"] > dataframe["high"].shift(1)) & 
            (dataframe["high"].shift(1) > dataframe["high"].shift(2))
        ).astype(int)
        dataframe["%-lower_low"] = (
            (dataframe["low"] < dataframe["low"].shift(1)) & 
            (dataframe["low"].shift(1) < dataframe["low"].shift(2))
        ).astype(int)
        
        # Volatility features
        dataframe["%-atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["%-atr_ratio"] = dataframe["%-atr"] / dataframe["close"]
        
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Define targets for the machine learning model.
        All targets must be prefixed with '&' to be recognized by FreqAI.
        
        :param dataframe: Strategy dataframe
        :param metadata: Pair metadata
        :return: DataFrame with added targets
        """
        
        # Get label period from config
        label_period = self.freqai_info["feature_parameters"]["label_period_candles"]
        
        # Target 1: Price movement prediction (regression)
        # Predict the average price movement over the next N candles
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-label_period)
            .rolling(label_period)
            .mean()
            / dataframe["close"]
            - 1
        )
        
        # Target 2: Volatility prediction (regression)
        # Predict the volatility over the next N candles
        dataframe["&-s_volatility"] = (
            dataframe["close"]
            .shift(-label_period)
            .rolling(label_period)
            .std()
            / dataframe["close"]
        )
        
        # Target 3: Trend direction classification (classification)
        # Classify if the trend will be up, down, or sideways
        future_high = dataframe["high"].shift(-label_period).rolling(label_period).max()
        future_low = dataframe["low"].shift(-label_period).rolling(label_period).min()
        current_price = dataframe["close"]
        
        # Define trend thresholds
        up_threshold = 0.02  # 2% increase
        down_threshold = -0.02  # 2% decrease
        
        # Calculate future price change
        future_change = (future_high + future_low) / 2 / current_price - 1
        
        # Classify trend
        dataframe["&-s_trend"] = np.where(
            future_change > up_threshold, "up",
            np.where(future_change < down_threshold, "down", "sideways")
        )
        
        # Target 4: Maximum drawdown prediction (regression)
        # Predict the maximum drawdown over the next N candles
        dataframe["&-s_max_drawdown"] = (
            (dataframe["close"].shift(-label_period).rolling(label_period).min() - 
             dataframe["close"]) / dataframe["close"]
        )
        
        # Target 5: Volume prediction (regression)
        # Predict the volume ratio over the next N candles
        dataframe["&-s_volume_ratio"] = (
            dataframe["volume"]
            .shift(-label_period)
            .rolling(label_period)
            .mean()
            / dataframe["volume"].rolling(20).mean()
        )
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators using FreqAI feature engineering.
        This function calls self.freqai.start() which processes all features
        and returns predictions along with quality metrics.
        """
        
        # FreqAI processes all features and returns predictions
        dataframe = self.freqai.start(dataframe, metadata, self)
        
        # Log prediction quality metrics
        if len(dataframe) > 0:
            last_candle = dataframe.iloc[-1]
            
            # Log prediction confidence
            if "do_predict" in last_candle:
                do_predict = last_candle["do_predict"]
                if do_predict != 1:
                    logger.warning(
                        f"Low prediction confidence for {metadata['pair']}: "
                        f"do_predict={do_predict}. "
                        f"This may indicate: "
                        f"{'DI threshold exceeded' if do_predict == 0 else 'SVM outlier detected' if do_predict == -1 else 'DBSCAN outlier detected' if do_predict == -2 else 'Model expired' if do_predict == 2 else 'Unknown issue'}"
                    )
            
            # Log DI values
            if "DI_values" in last_candle:
                di_value = last_candle["DI_values"]
                logger.info(f"DI value for {metadata['pair']}: {di_value:.4f}")
                
                # Warning for high DI values
                if di_value > 2.0:
                    logger.warning(
                        f"High DI value ({di_value:.4f}) for {metadata['pair']} - "
                        f"prediction may be unreliable"
                    )
            
            # Log target statistics
            if "&-s_close_mean" in last_candle and "&-s_close_std" in last_candle:
                mean_val = last_candle["&-s_close_mean"]
                std_val = last_candle["&-s_close_std"]
                logger.info(
                    f"Target stats for {metadata['pair']}: "
                    f"mean={mean_val:.6f}, std={std_val:.6f}"
                )
        
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions based on FreqAI predictions and quality metrics.
        """
        
        # Initialize entry columns
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = ""
        
        # Entry conditions for long positions
        enter_long_conditions = [
            df["do_predict"] == 1,  # Prediction is reliable
            df["&-s_close"] > 0.01,  # Predicted price increase > 1%
            df["&-s_trend"] == "up",  # Trend prediction is up
            df["&-s_volatility"] < 0.05,  # Low volatility prediction
            df["DI_values"] < 1.5,  # DI not too high
        ]
        
        # Calculate z-score for dynamic threshold
        if "&-s_close_mean" in df.columns and "&-s_close_std" in df.columns:
            z_score = (df["&-s_close"] - df["&-s_close_mean"]) / df["&-s_close_std"]
            enter_long_conditions.append(z_score > 1.0)  # Z-score > 1.0
        
        # Apply long entry conditions
        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), 
                ["enter_long", "enter_tag"]
            ] = (1, "freqai_long")
        
        # Entry conditions for short positions
        enter_short_conditions = [
            df["do_predict"] == 1,  # Prediction is reliable
            df["&-s_close"] < -0.01,  # Predicted price decrease > 1%
            df["&-s_trend"] == "down",  # Trend prediction is down
            df["&-s_volatility"] < 0.05,  # Low volatility prediction
            df["DI_values"] < 1.5,  # DI not too high
        ]
        
        # Apply short entry conditions
        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), 
                ["enter_short", "enter_tag"]
            ] = (1, "freqai_short")
        
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit conditions based on FreqAI predictions and quality metrics.
        """
        
        # Initialize exit columns
        df["exit_long"] = 0
        df["exit_short"] = 0
        df["exit_tag"] = ""
        
        # Exit conditions for long positions
        exit_long_conditions = [
            df["do_predict"] == 1,  # Prediction is reliable
            df["&-s_close"] < 0,  # Predicted price decrease
        ]
        
        # Add trend-based exit
        if "&-s_trend" in df.columns:
            exit_long_conditions.append(df["&-s_trend"] == "down")
        
        # Apply long exit conditions
        if exit_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_long_conditions), 
                "exit_long"
            ] = 1
        
        # Exit conditions for short positions
        exit_short_conditions = [
            df["do_predict"] == 1,  # Prediction is reliable
            df["&-s_close"] > 0,  # Predicted price increase
        ]
        
        # Add trend-based exit
        if "&-s_trend" in df.columns:
            exit_short_conditions.append(df["&-s_trend"] == "up")
        
        # Apply short exit conditions
        if exit_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_short_conditions), 
                "exit_short"
            ] = 1
        
        return df

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        """
        Additional confirmation before entering a trade.
        """
        
        # Get current dataframe
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(df) == 0:
            return False
        
        last_candle = df.iloc[-1]
        
        # Check if prediction is still reliable
        if "do_predict" in last_candle and last_candle["do_predict"] != 1:
            logger.warning(
                f"Trade entry rejected for {pair}: "
                f"prediction confidence dropped (do_predict={last_candle['do_predict']})"
            )
            return False
        
        # Check DI values
        if "DI_values" in last_candle and last_candle["DI_values"] > 2.0:
            logger.warning(
                f"Trade entry rejected for {pair}: "
                f"DI value too high ({last_candle['DI_values']:.4f})"
            )
            return False
        
        # Check if model is expired
        if "do_predict" in last_candle and last_candle["do_predict"] == 2:
            logger.warning(f"Trade entry rejected for {pair}: model has expired")
            return False
        
        # Price slippage protection
        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):  # 0.25% slippage
                logger.info(f"Trade entry rejected for {pair}: price slippage too high")
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):  # 0.25% slippage
                logger.info(f"Trade entry rejected for {pair}: price slippage too high")
                return False
        
        logger.info(
            f"Trade entry confirmed for {pair} ({side}): "
            f"rate={rate:.6f}, close={last_candle['close']:.6f}, "
            f"do_predict={last_candle.get('do_predict', 'N/A')}, "
            f"DI={last_candle.get('DI_values', 'N/A'):.4f}"
        )
        
        return True

    def custom_stoploss(
        self, pair: str, trade, current_time, current_rate, current_profit, **kwargs
    ) -> float:
        """
        Custom stoploss based on FreqAI predictions.
        """
        
        # Get current dataframe
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(df) == 0:
            return self.stoploss
        
        last_candle = df.iloc[-1]
        
        # Adjust stoploss based on volatility prediction
        if "&-s_volatility" in last_candle:
            volatility = last_candle["&-s_volatility"]
            # Increase stoploss for high volatility periods
            if volatility > 0.05:  # 5% volatility
                adjusted_stoploss = self.stoploss * 1.5
                logger.info(
                    f"Adjusted stoploss for {pair}: "
                    f"volatility={volatility:.4f}, "
                    f"stoploss={adjusted_stoploss:.4f}"
                )
                return adjusted_stoploss
        
        return self.stoploss
