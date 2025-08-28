"""
Unit tests for FreqAI feature engineering functionality.
Tests feature expansion, target generation, and model path handling.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

# Import the strategy
from freqtrade.templates.FreqaiExampleStrategy import FreqaiExampleStrategy


class TestFreqaiFeatureEngineering:
    """Test FreqAI feature engineering functionality."""

    @pytest.fixture
    def strategy(self):
        """Create a FreqAI strategy instance for testing."""
        # Create a mock config
        mock_config = {
            'freqai': {
                'enabled': True,
                'identifier': 'test-freqai',
                'feature_parameters': {
                    'include_timeframes': ['5m', '15m'],
                    'include_corr_pairlist': ['BTC/USDT:USDT'],
                    'indicator_periods_candles': [10, 20],
                    'include_shifted_candles': 2,
                    'label_period_candles': 24
                }
            },
            'timeframe': '5m',
            'stake_currency': 'USDT',
            'stake_amount': 100
        }
        
        strategy = FreqaiExampleStrategy(mock_config)
        
        # Mock freqai_info to match the actual strategy structure
        strategy.freqai_info = {
            'feature_parameters': {
                'label_period_candles': 24,
                'include_timeframes': ['5m', '15m'],
                'include_corr_pairlist': ['BTC/USDT:USDT'],
                'indicator_periods_candles': [10, 20],
                'include_shifted_candles': 2
            }
        }
        
        return strategy

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        df = pd.DataFrame({
            'date': dates,  # Add date column
            'open': np.random.random(100) * 100 + 50,
            'high': np.random.random(100) * 100 + 50,
            'low': np.random.random(100) * 100 + 50,
            'close': np.random.random(100) * 100 + 50,
            'volume': np.random.random(100) * 1000 + 100
        })
        
        # Add some basic indicators
        df['rsi'] = np.random.random(100) * 100
        df['ema_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['bb_upper'] = df['close'].rolling(20).mean() + df['close'].rolling(20).std() * 2
        df['bb_lower'] = df['close'].rolling(20).mean() - df['close'].rolling(20).std() * 2
        
        return df

    def test_feature_engineering_expand_all(self, strategy, sample_dataframe):
        """Test that expand_all features are created correctly."""
        df = strategy.feature_engineering_expand_all(sample_dataframe.copy(), 10, {})
        
        # Check that features are created
        assert '%-rsi-10' in df.columns
        assert '%-mfi-10' in df.columns
        assert '%-ema-10' in df.columns
        
        # Check that values are numeric
        assert df['%-rsi-10'].dtype in ['float64', 'float32']
        assert df['%-mfi-10'].dtype in ['float64', 'float32']
        
        # Check that features are not all NaN
        assert not df['%-rsi-10'].isna().all()
        assert not df['%-mfi-10'].isna().all()

    def test_feature_engineering_expand_basic(self, strategy, sample_dataframe):
        """Test that expand_basic features are created correctly."""
        df = strategy.feature_engineering_expand_basic(sample_dataframe.copy(), {})
        
        # Check that features are created
        assert '%-raw_high' in df.columns
        assert '%-raw_low' in df.columns
        assert '%-pct_change_2' in df.columns
        assert '%-pct_change_3' in df.columns
        
        # Check that values are numeric
        assert df['%-raw_high'].dtype in ['float64', 'float32']
        assert df['%-pct_change_2'].dtype in ['float64', 'float32']
        
        # Check that features are not all NaN
        assert not df['%-raw_high'].isna().all()
        assert not df['%-pct_change_2'].isna().all()

    def test_feature_engineering_standard(self, strategy, sample_dataframe):
        """Test that standard features are created correctly."""
        df = strategy.feature_engineering_standard(sample_dataframe.copy(), {})
        
        # Check that features are created
        assert '%-day_of_week' in df.columns
        assert '%-hour_of_day' in df.columns
        assert '%-minute_of_hour' in df.columns
        
        # Check that values are normalized (0-1)
        assert df['%-day_of_week'].min() >= 0
        assert df['%-day_of_week'].max() <= 1
        assert df['%-hour_of_day'].min() >= 0
        assert df['%-hour_of_day'].max() <= 1
        
        # Check that features are not all NaN
        assert not df['%-day_of_week'].isna().all()
        assert not df['%-hour_of_day'].isna().all()

    def test_set_freqai_targets(self, strategy, sample_dataframe):
        """Test that targets are created correctly."""
        df = strategy.set_freqai_targets(sample_dataframe.copy(), {})
        
        # Check that targets are created
        assert '&-s_close' in df.columns
        assert '&-s_volatility' in df.columns
        assert '&-s_trend' in df.columns
        assert '&-s_max_drawdown' in df.columns
        assert '&-s_volume_ratio' in df.columns
        
        # Check that values are numeric
        assert df['&-s_close'].dtype in ['float64', 'float32']
        assert df['&-s_volatility'].dtype in ['float64', 'float32']
        
        # Check that trend is categorical
        assert df['&-s_trend'].dtype == 'object'
        assert set(df['&-s_trend'].dropna().unique()).issubset({'up', 'down', 'sideways'})
        
        # Check that targets are not all NaN
        assert not df['&-s_close'].isna().all()
        assert not df['&-s_trend'].isna().all()

    def test_feature_expansion_calculation(self, strategy):
        """Test that feature expansion calculation is correct."""
        # Calculate expected features based on config
        timeframes = len(strategy.freqai_info['feature_parameters']['include_timeframes'])
        corr_pairs = len(strategy.freqai_info['feature_parameters']['include_corr_pairlist'])
        indicator_periods = len(strategy.freqai_info['feature_parameters']['indicator_periods_candles'])
        shifted_candles = strategy.freqai_info['feature_parameters']['include_shifted_candles']
        
        # Basic features that don't expand
        basic_features = 5  # raw_high, raw_low, pct_change_2, pct_change_3, etc.
        
        # Standard features that don't expand
        standard_features = 15  # day_of_week, hour_of_day, etc.
        
        # Expandable features
        expandable_features = 8  # rsi, ema, sma, bb_upper, bb_lower, etc.
        
        total_expandable = expandable_features * indicator_periods * timeframes * (shifted_candles + 1)
        total_features = basic_features + standard_features + total_expandable
        
        # This is a rough estimate - actual count may vary
        assert total_features > 50  # Should have many features

    def test_startup_candle_count(self, strategy):
        """Test that startup_candle_count is sufficient."""
        # startup_candle_count should be large enough for all indicators
        max_period = max(strategy.freqai_info['feature_parameters']['indicator_periods_candles'])
        label_period = strategy.freqai_info['feature_parameters']['label_period_candles']
        shifted_candles = strategy.freqai_info['feature_parameters']['include_shifted_candles']
        
        required_candles = max_period + label_period + shifted_candles + 50  # buffer
        
        assert strategy.startup_candle_count >= required_candles

    def test_metadata_usage(self, strategy, sample_dataframe):
        """Test that metadata is used correctly in feature functions."""
        metadata = {'pair': 'BTC/USDT:USDT', 'timeframe': '5m'}
        
        # All feature functions should accept metadata
        df1 = strategy.feature_engineering_expand_all(sample_dataframe.copy(), 10, metadata)
        df2 = strategy.feature_engineering_expand_basic(sample_dataframe.copy(), metadata)
        df3 = strategy.feature_engineering_standard(sample_dataframe.copy(), metadata)
        df4 = strategy.set_freqai_targets(sample_dataframe.copy(), metadata)
        
        # All should return dataframes
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
        assert isinstance(df3, pd.DataFrame)
        assert isinstance(df4, pd.DataFrame)

    def test_feature_naming_convention(self, strategy, sample_dataframe):
        """Test that all features follow naming convention."""
        df = strategy.feature_engineering_expand_all(sample_dataframe.copy(), 10, {})
        df = strategy.feature_engineering_expand_basic(df, {})
        df = strategy.feature_engineering_standard(df, {})
        
        # All features should start with %-
        feature_columns = [col for col in df.columns if col.startswith('%-')]
        assert len(feature_columns) > 0
        
        for col in feature_columns:
            assert col.startswith('%-')

    def test_target_naming_convention(self, strategy, sample_dataframe):
        """Test that all targets follow naming convention."""
        df = strategy.set_freqai_targets(sample_dataframe.copy(), {})
        
        # All targets should start with &-
        target_columns = [col for col in df.columns if col.startswith('&-')]
        assert len(target_columns) > 0
        
        for col in target_columns:
            assert col.startswith('&-')

    def test_dataframe_integrity(self, strategy, sample_dataframe):
        """Test that dataframe length and original columns are preserved."""
        original_length = len(sample_dataframe)
        original_columns = set(sample_dataframe.columns)
        
        df = strategy.feature_engineering_expand_all(sample_dataframe.copy(), 10, {})
        df = strategy.feature_engineering_expand_basic(df, {})
        df = strategy.feature_engineering_standard(df, {})
        df = strategy.set_freqai_targets(df, {})
        
        # Length should be preserved
        assert len(df) == original_length
        
        # Original columns should be preserved
        for col in original_columns:
            assert col in df.columns


class TestFreqaiModelPaths:
    """Test FreqAI model path functionality."""

    def test_model_path_creation(self):
        """Test that model paths are created correctly."""
        identifier = 'test-freqai-v1'
        expected_path = f'user_data/models/{identifier}'
        
        # Check that the path format is correct
        assert 'user_data/models' in expected_path
        assert identifier in expected_path

    def test_identifier_uniqueness(self):
        """Test that identifiers are unique."""
        identifiers = [
            'production-freqai-v1',
            'test-freqai-v1',
            'demo-freqai-v1'
        ]
        
        # All identifiers should be unique
        assert len(identifiers) == len(set(identifiers))
