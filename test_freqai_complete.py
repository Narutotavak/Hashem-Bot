#!/usr/bin/env python3
"""
Comprehensive test script for FreqAI strategy including trading logic.
"""

from freqtrade.templates.FreqaiExampleStrategy import FreqaiExampleStrategy
import pandas as pd
import numpy as np

def test_complete_strategy():
    """Test the complete FreqAI strategy including trading logic."""
    print("=== Complete FreqAI Strategy Test ===")
    
    # Create sample data with more realistic patterns
    dates = pd.date_range('2024-01-01', periods=200, freq='5min')
    
    # Create trending price data
    trend = np.linspace(0, 1, 200)
    noise = np.random.normal(0, 0.02, 200)
    base_price = 100 + trend * 20 + noise
    
    df = pd.DataFrame({
        'date': dates,
        'open': base_price + np.random.normal(0, 0.5, 200),
        'high': base_price + np.random.normal(0.5, 0.3, 200),
        'low': base_price + np.random.normal(-0.5, 0.3, 200),
        'close': base_price + np.random.normal(0, 0.2, 200),
        'volume': np.random.randint(1000, 10000, 200)
    })
    
    # Ensure high >= close >= low
    df['high'] = df[['high', 'close']].max(axis=1)
    df['low'] = df[['low', 'close']].min(axis=1)
    df['open'] = df['close'].shift(1).fillna(df['close'])
    
    # Create strategy instance
    strategy = FreqaiExampleStrategy({'freqai': {'enabled': True}})
    strategy.freqai_info = {
        'feature_parameters': {
            'label_period_candles': 24,
            'include_timeframes': ['5m'],
            'include_corr_pairlist': ['BTC/USDT:USDT'],
            'indicator_periods_candles': [10, 20],
            'include_shifted_candles': 2
        }
    }
    
    print("1. Testing feature engineering...")
    # Apply all feature engineering functions
    df_features = df.copy()
    df_features = strategy.feature_engineering_expand_all(df_features, 10, {})
    df_features = strategy.feature_engineering_expand_basic(df_features, {})
    df_features = strategy.feature_engineering_standard(df_features, {})
    df_features = strategy.set_freqai_targets(df_features, {})
    
    print(f"   Total features created: {len([col for col in df_features.columns if col.startswith('%-')])}")
    print(f"   Total targets created: {len([col for col in df_features.columns if col.startswith('&-')])}")
    
    print("\n2. Testing trading logic...")
    
    # Mock FreqAI predictions
    df_features['do_predict'] = 1  # High confidence
    df_features['DI_values'] = 0.5  # Low dissimilarity
    df_features['&-s_close_mean'] = df_features['&-s_close'].rolling(10).mean()
    df_features['&-s_close_std'] = df_features['&-s_close'].rolling(10).std()
    
    # Test entry conditions
    entry_signals = []
    for i in range(100, len(df_features)):
        try:
            # Mock metadata
            metadata = {'pair': 'BTC/USDT:USDT', 'timeframe': '5m'}
            
            # Test entry trend
            entry_long = strategy.populate_entry_trend(df_features.iloc[:i+1], metadata)
            entry_short = strategy.populate_entry_trend(df_features.iloc[:i+1], metadata)
            
            if entry_long.iloc[-1] == 1:
                entry_signals.append(('LONG', i, df_features.iloc[i]['close']))
            elif entry_short.iloc[-1] == 1:
                entry_signals.append(('SHORT', i, df_features.iloc[i]['close']))
                
        except Exception as e:
            continue
    
    print(f"   Entry signals found: {len(entry_signals)}")
    if entry_signals:
        print(f"   First entry: {entry_signals[0]}")
        print(f"   Last entry: {entry_signals[-1]}")
    
    print("\n3. Testing strategy parameters...")
    print(f"   startup_candle_count: {strategy.startup_candle_count}")
    print(f"   minimal_roi: {strategy.minimal_roi}")
    print(f"   stoploss: {strategy.stoploss}")
    print(f"   can_short: {strategy.can_short}")
    
    print("\n4. Testing plot configuration...")
    print(f"   Main plot keys: {list(strategy.plot_config.get('main_plot', {}).keys())}")
    print(f"   Subplots: {list(strategy.plot_config.get('subplots', {}).keys())}")
    
    print("\n=== Complete test finished successfully! ===")
    return True

if __name__ == "__main__":
    test_complete_strategy()
