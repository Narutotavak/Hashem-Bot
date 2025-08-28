"""
Unit tests for FreqAI strategy functionality.
Tests entry/exit conditions, trade confirmation, and custom stoploss.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from freqtrade.templates.FreqaiExampleStrategy import FreqaiExampleStrategy


class TestFreqaiStrategy:
    """Test class for FreqAI strategy functionality."""
    
    @pytest.fixture
    def strategy(self):
        """Create a FreqAI strategy instance for testing."""
        mock_config = {
            'freqai': {
                'enabled': True,
                'identifier': 'test-strategy',
                'feature_parameters': {
                    'label_period_candles': 24,
                    'indicator_periods_candles': [10, 20, 50],
                    'include_timeframes': ["5m", "15m", "1h", "4h"],
                    'include_corr_pairlist': ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                    'include_shifted_candles': 3
                }
            }
        }
        strategy = FreqaiExampleStrategy(mock_config)
        # Add required attributes
        strategy.timeframe = "5m"
        strategy.freqai_info = {
            "feature_parameters": {
                "label_period_candles": 24,
                "indicator_periods_candles": [10, 20, 50],
                "include_timeframes": ["5m", "15m", "1h", "4h"],
                "include_corr_pairlist": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                "include_shifted_candles": 3
            }
        }
        return strategy
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe with FreqAI predictions."""
        dates = pd.date_range('2023-01-01', periods=100, freq='5min')
        df = pd.DataFrame({
            'date': dates,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Add FreqAI prediction columns
        df['do_predict'] = 1  # All predictions are reliable
        df['DI_values'] = 0.5  # Low DI values
        df['&-s_close'] = np.random.randn(100) * 0.02  # Random price predictions
        df['&-s_volatility'] = np.random.rand(100) * 0.1  # Random volatility predictions
        df['&-s_trend'] = np.random.choice(['up', 'down', 'sideways'], 100)
        df['&-s_close_mean'] = 0.001
        df['&-s_close_std'] = 0.002
        
        return df
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return {
            "pair": "BTC/USDT:USDT"
        }
    
    def test_entry_trend_long_conditions(self, strategy, sample_dataframe, sample_metadata):
        """Test long entry conditions."""
        
        # Set up conditions for long entry
        df = sample_dataframe.copy()
        df.loc[50, 'do_predict'] = 1
        df.loc[50, '&-s_close'] = 0.015  # > 0.01
        df.loc[50, '&-s_trend'] = 'up'
        df.loc[50, '&-s_volatility'] = 0.03  # < 0.05
        df.loc[50, 'DI_values'] = 1.2  # < 1.5
        
        # Calculate z-score
        z_score = (df.loc[50, '&-s_close'] - df.loc[50, '&-s_close_mean']) / df.loc[50, '&-s_close_std']
        df.loc[50, 'z_score'] = z_score
        
        result = strategy.populate_entry_trend(df, sample_metadata)
        
        # Check that long entry was triggered
        assert result.loc[50, 'enter_long'] == 1
        assert result.loc[50, 'enter_tag'] == 'freqai_long'
    
    def test_entry_trend_short_conditions(self, strategy, sample_dataframe, sample_metadata):
        """Test short entry conditions."""
        
        # Set up conditions for short entry
        df = sample_dataframe.copy()
        df.loc[60, 'do_predict'] = 1
        df.loc[60, '&-s_close'] = -0.015  # < -0.01
        df.loc[60, '&-s_trend'] = 'down'
        df.loc[60, '&-s_volatility'] = 0.03  # < 0.05
        df.loc[60, 'DI_values'] = 1.2  # < 1.5
        
        result = strategy.populate_entry_trend(df, sample_metadata)
        
        # Check that short entry was triggered
        assert result.loc[60, 'enter_short'] == 1
        assert result.loc[60, 'enter_tag'] == 'freqai_short'
    
    def test_entry_trend_no_entry_low_confidence(self, strategy, sample_dataframe, sample_metadata):
        """Test that no entry occurs when prediction confidence is low."""
        
        df = sample_dataframe.copy()
        df.loc[70, 'do_predict'] = 0  # Low confidence
        
        result = strategy.populate_entry_trend(df, sample_metadata)
        
        # Check that no entry was triggered
        assert result.loc[70, 'enter_long'] == 0
        assert result.loc[70, 'enter_short'] == 0
    
    def test_entry_trend_no_entry_high_di(self, strategy, sample_dataframe, sample_metadata):
        """Test that no entry occurs when DI is too high."""
        
        df = sample_dataframe.copy()
        df.loc[80, 'do_predict'] = 1
        df.loc[80, 'DI_values'] = 2.0  # > 1.5
        
        result = strategy.populate_entry_trend(df, sample_metadata)
        
        # Check that no entry was triggered
        assert result.loc[80, 'enter_long'] == 0
        assert result.loc[80, 'enter_short'] == 0
    
    def test_entry_trend_no_entry_wrong_trend(self, strategy, sample_dataframe, sample_metadata):
        """Test that no entry occurs when trend prediction doesn't match."""
        
        df = sample_dataframe.copy()
        df.loc[90, 'do_predict'] = 1
        df.loc[90, '&-s_close'] = 0.015  # > 0.01
        df.loc[90, '&-s_trend'] = 'down'  # Wrong trend for long
        
        result = strategy.populate_entry_trend(df, sample_metadata)
        
        # Check that no long entry was triggered
        assert result.loc[90, 'enter_long'] == 0
    
    def test_exit_trend_long_exit(self, strategy, sample_dataframe, sample_metadata):
        """Test long exit conditions."""
        
        df = sample_dataframe.copy()
        df.loc[40, 'do_predict'] = 1
        df.loc[40, '&-s_close'] = -0.005  # < 0 (predicted decrease)
        df.loc[40, '&-s_trend'] = 'down'
        
        result = strategy.populate_exit_trend(df, sample_metadata)
        
        # Check that long exit was triggered
        assert result.loc[40, 'exit_long'] == 1
    
    def test_exit_trend_short_exit(self, strategy, sample_dataframe, sample_metadata):
        """Test short exit conditions."""
        
        df = sample_dataframe.copy()
        df.loc[50, 'do_predict'] = 1
        df.loc[50, '&-s_close'] = 0.005  # > 0 (predicted increase)
        df.loc[50, '&-s_trend'] = 'up'
        
        result = strategy.populate_exit_trend(df, sample_metadata)
        
        # Check that short exit was triggered
        assert result.loc[50, 'exit_short'] == 1
    
    def test_exit_trend_no_exit_high_confidence(self, strategy, sample_dataframe, sample_metadata):
        """Test that no exit occurs when prediction confidence is low."""
        
        df = sample_dataframe.copy()
        df.loc[60, 'do_predict'] = 0  # Low confidence
        
        result = strategy.populate_exit_trend(df, sample_metadata)
        
        # Check that no exit was triggered
        assert result.loc[60, 'exit_long'] == 0
        assert result.loc[60, 'exit_short'] == 0
    
    def test_confirm_trade_entry_success(self, strategy, sample_dataframe, sample_metadata):
        """Test successful trade entry confirmation."""
        
        # Set up conditions for successful entry
        df = sample_dataframe.copy()
        df.loc[len(df)-1, 'close'] = 100.0  # Set close price to 100.0
        df.loc[len(df)-1, 'do_predict'] = 1  # High confidence
        df.loc[len(df)-1, 'DI_values'] = 0.5  # Low DI
        
        # Mock the dp attribute directly on the strategy instance
        strategy.dp = Mock()
        strategy.dp.get_analyzed_dataframe.return_value = (df, None)
        
        # Test long entry with rate close to close price
        result = strategy.confirm_trade_entry(
            pair="BTC/USDT:USDT",
            order_type="limit",
            amount=0.1,
            rate=100.1,  # Very close to close price (100.0)
            time_in_force="gtc",
            current_time=pd.Timestamp.now(),
            entry_tag="test",
            side="long"
        )
        
        assert result is True
    
    def test_confirm_trade_entry_low_confidence(self, strategy, sample_dataframe, sample_metadata):
        """Test trade entry rejection due to low prediction confidence."""
        
        # Set low confidence
        df = sample_dataframe.copy()
        df.loc[len(df)-1, 'do_predict'] = 0
        
        # Mock the dp attribute directly on the strategy instance
        strategy.dp = Mock()
        strategy.dp.get_analyzed_dataframe.return_value = (df, None)
        
        result = strategy.confirm_trade_entry(
            pair="BTC/USDT:USDT",
            order_type="limit",
            amount=0.1,
            rate=100.0,
            time_in_force="gtc",
            current_time=pd.Timestamp.now(),
            entry_tag="test",
            side="long"
        )
        
        assert result is False
    
    def test_confirm_trade_entry_high_di(self, strategy, sample_dataframe, sample_metadata):
        """Test trade entry rejection due to high DI values."""
        
        # Set high DI
        df = sample_dataframe.copy()
        df.loc[len(df)-1, 'DI_values'] = 2.5
        
        # Mock the dp attribute directly on the strategy instance
        strategy.dp = Mock()
        strategy.dp.get_analyzed_dataframe.return_value = (df, None)
        
        result = strategy.confirm_trade_entry(
            pair="BTC/USDT:USDT",
            order_type="limit",
            amount=0.1,
            rate=100.0,
            time_in_force="gtc",
            current_time=pd.Timestamp.now(),
            entry_tag="test",
            side="long"
        )
        
        assert result is False
    
    def test_confirm_trade_entry_model_expired(self, strategy, sample_dataframe, sample_metadata):
        """Test trade entry rejection due to expired model."""
        
        # Set expired model
        df = sample_dataframe.copy()
        df.loc[len(df)-1, 'do_predict'] = 2
        
        # Mock the dp attribute directly on the strategy instance
        strategy.dp = Mock()
        strategy.dp.get_analyzed_dataframe.return_value = (df, None)
        
        result = strategy.confirm_trade_entry(
            pair="BTC/USDT:USDT",
            order_type="limit",
            amount=0.1,
            rate=100.0,
            time_in_force="gtc",
            current_time=pd.Timestamp.now(),
            entry_tag="test",
            side="long"
        )
        
        assert result is False
    
    def test_confirm_trade_entry_price_slippage(self, strategy, sample_dataframe, sample_metadata):
        """Test trade entry rejection due to price slippage."""
        
        # Set up conditions for slippage test
        df = sample_dataframe.copy()
        df.loc[len(df)-1, 'close'] = 100.0  # Set close price to 100.0
        df.loc[len(df)-1, 'do_predict'] = 1  # High confidence
        df.loc[len(df)-1, 'DI_values'] = 0.5  # Low DI
        
        # Mock the dp attribute directly on the strategy instance
        strategy.dp = Mock()
        strategy.dp.get_analyzed_dataframe.return_value = (df, None)
        
        # Test long entry with high slippage (0.5% above close)
        result = strategy.confirm_trade_entry(
            pair="BTC/USDT:USDT",
            order_type="limit",
            amount=0.1,
            rate=100.5,  # 0.5% above close (100.0)
            time_in_force="gtc",
            current_time=pd.Timestamp.now(),
            entry_tag="test",
            side="long"
        )
        
        assert result is False
    
    def test_custom_stoploss_normal_volatility(self, strategy, sample_dataframe, sample_metadata):
        """Test custom stoploss with normal volatility."""
        
        # Set low volatility
        df = sample_dataframe.copy()
        df.loc[len(df)-1, '&-s_volatility'] = 0.03  # < 0.05
        
        # Mock the dp attribute directly on the strategy instance
        strategy.dp = Mock()
        strategy.dp.get_analyzed_dataframe.return_value = (df, None)
        
        # Mock trade object
        trade = Mock()
        
        result = strategy.custom_stoploss(
            pair="BTC/USDT:USDT",
            trade=trade,
            current_time=pd.Timestamp.now(),
            current_rate=100.0,
            current_profit=0.0
        )
        
        # Should return default stoploss
        assert result == strategy.stoploss
    
    def test_custom_stoploss_high_volatility(self, strategy, sample_dataframe, sample_metadata):
        """Test custom stoploss with high volatility."""
        
        # Set high volatility
        df = sample_dataframe.copy()
        df.loc[len(df)-1, '&-s_volatility'] = 0.08  # > 0.05
        
        # Mock the dp attribute directly on the strategy instance
        strategy.dp = Mock()
        strategy.dp.get_analyzed_dataframe.return_value = (df, None)
        
        # Mock trade object
        trade = Mock()
        
        result = strategy.custom_stoploss(
            pair="BTC/USDT:USDT",
            trade=trade,
            current_time=pd.Timestamp.now(),
            current_rate=100.0,
            current_profit=0.0
        )
        
        # Should return adjusted stoploss
        expected_stoploss = strategy.stoploss * 1.5
        assert result == expected_stoploss
    
    def test_strategy_parameters(self, strategy):
        """Test that strategy parameters are set correctly."""
        
        assert strategy.minimal_roi == {"0": 0.1, "240": -1}
        assert strategy.stoploss == -0.05
        assert strategy.use_exit_signal is True
        assert strategy.can_short is True
        assert strategy.process_only_new_candles is True
        assert strategy.startup_candle_count == 100
    
    def test_plot_config(self, strategy):
        """Test that plot configuration is set correctly."""
        
        plot_config = strategy.plot_config
        
        # Check main plot
        assert "close" in plot_config["main_plot"]
        assert "&-s_close" in plot_config["main_plot"]
        assert "&-s_close_mean" in plot_config["main_plot"]
        assert "&-s_close_std" in plot_config["main_plot"]
        
        # Check subplots
        assert "Features" in plot_config["subplots"]
        assert "Predictions" in plot_config["subplots"]
        assert "Quality" in plot_config["subplots"]
        
        # Check specific features
        assert "%-rsi-10" in plot_config["subplots"]["Features"]
        assert "do_predict" in plot_config["subplots"]["Predictions"]
        assert "DI_values" in plot_config["subplots"]["Quality"]
    
    def test_feature_engineering_functions_exist(self, strategy):
        """Test that all required feature engineering functions exist."""
        
        assert hasattr(strategy, 'feature_engineering_expand_all')
        assert hasattr(strategy, 'feature_engineering_expand_basic')
        assert hasattr(strategy, 'feature_engineering_standard')
        assert hasattr(strategy, 'set_freqai_targets')
        
        # Check that they are callable
        assert callable(strategy.feature_engineering_expand_all)
        assert callable(strategy.feature_engineering_expand_basic)
        assert callable(strategy.feature_engineering_standard)
        assert callable(strategy.set_freqai_targets)
    
    def test_strategy_methods_exist(self, strategy):
        """Test that all required strategy methods exist."""
        
        assert hasattr(strategy, 'populate_indicators')
        assert hasattr(strategy, 'populate_entry_trend')
        assert hasattr(strategy, 'populate_exit_trend')
        assert hasattr(strategy, 'confirm_trade_entry')
        assert hasattr(strategy, 'custom_stoploss')
        
        # Check that they are callable
        assert callable(strategy.populate_indicators)
        assert callable(strategy.populate_entry_trend)
        assert callable(strategy.populate_exit_trend)
        assert callable(strategy.confirm_trade_entry)
        assert callable(strategy.custom_stoploss)


class TestFreqaiStrategyIntegration:
    """Test class for FreqAI strategy integration scenarios."""
    
    @pytest.fixture
    def strategy(self):
        """Create a FreqAI strategy instance for testing."""
        mock_config = {
            'freqai': {
                'enabled': True,
                'identifier': 'test-strategy',
                'feature_parameters': {
                    'label_period_candles': 24,
                    'indicator_periods_candles': [10, 20, 50],
                    'include_timeframes': ["5m", "15m", "1h", "4h"],
                    'include_corr_pairlist': ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                    'include_shifted_candles': 3
                }
            }
        }
        strategy = FreqaiExampleStrategy(mock_config)
        # Add required attributes
        strategy.timeframe = "5m"
        strategy.freqai_info = {
            "feature_parameters": {
                "label_period_candles": 24,
                "indicator_periods_candles": [10, 20, 50],
                "include_timeframes": ["5m", "15m", "1h", "4h"],
                "include_corr_pairlist": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                "include_shifted_candles": 3
            }
        }
        return strategy
    
    def test_complete_trading_cycle(self, strategy):
        """Test a complete trading cycle with the strategy."""
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=200, freq='5min')
        df = pd.DataFrame({
            'date': dates,
            'open': np.random.randn(200).cumsum() + 100,
            'high': np.random.randn(200).cumsum() + 102,
            'low': np.random.randn(200).cumsum() + 98,
            'close': np.random.randn(200).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 200)
        })
        
        # Add FreqAI predictions
        df['do_predict'] = 1
        df['DI_values'] = 0.5
        df['&-s_close'] = np.random.randn(200) * 0.02
        df['&-s_volatility'] = np.random.rand(200) * 0.1
        df['&-s_trend'] = np.random.choice(['up', 'down', 'sideways'], 200)
        df['&-s_close_mean'] = 0.001
        df['&-s_close_std'] = 0.002
        
        # Set up specific conditions for testing
        # Entry point
        df.loc[100, '&-s_close'] = 0.015  # Strong positive prediction
        df.loc[100, '&-s_trend'] = 'up'
        df.loc[100, '&-s_volatility'] = 0.03
        
        # Exit point
        df.loc[150, '&-s_close'] = -0.005  # Negative prediction
        df.loc[150, '&-s_trend'] = 'down'
        
        metadata = {"pair": "BTC/USDT:USDT"}
        
        # Test entry
        entry_df = strategy.populate_entry_trend(df.copy(), metadata)
        assert entry_df.loc[100, 'enter_long'] == 1
        
        # Test exit
        exit_df = strategy.populate_exit_trend(df.copy(), metadata)
        assert exit_df.loc[150, 'exit_long'] == 1
    
    def test_multiple_targets_handling(self, strategy):
        """Test that multiple targets are handled correctly."""
        
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='5min'),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        metadata = {"pair": "BTC/USDT:USDT"}
        
        # Generate targets
        result = strategy.set_freqai_targets(df.copy(), metadata)
        
        # Check all targets exist
        targets = ["&-s_close", "&-s_volatility", "&-s_trend", "&-s_max_drawdown", "&-s_volume_ratio"]
        for target in targets:
            assert target in result.columns
        
        # Check target data types
        assert result["&-s_close"].dtype in ['float64', 'float32']
        assert result["&-s_volatility"].dtype in ['float64', 'float32']
        assert result["&-s_trend"].dtype == 'object'
        assert result["&-s_max_drawdown"].dtype in ['float64', 'float32']
        assert result["&-s_volume_ratio"].dtype in ['float64', 'float32']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
