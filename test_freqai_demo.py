#!/usr/bin/env python3
"""
Simple test script to demonstrate FreqAI strategy functionality.
"""

from freqtrade.templates.FreqaiExampleStrategy import FreqaiExampleStrategy
import pandas as pd
import numpy as np

def test_freqai_strategy():
    """Test the FreqAI strategy with sample data."""
    print("=== FreqAI Strategy Feature Engineering Test ===")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    df = pd.DataFrame({
        'date': dates,
        'open': np.random.random(100) * 100 + 50,
        'high': np.random.random(100) * 100 + 50,
        'low': np.random.random(100) * 100 + 50,
        'close': np.random.random(100) * 100 + 50,
        'volume': np.random.random(100) * 1000 + 100
    })
    
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
    
    print("1. Testing expand_all...")
    df1 = strategy.feature_engineering_expand_all(df.copy(), 10, {})
    expand_all_features = len([col for col in df1.columns if col.startswith('%-')])
    print(f"   Features created: {expand_all_features}")
    
    print("2. Testing expand_basic...")
    df2 = strategy.feature_engineering_expand_basic(df1.copy(), {})
    basic_features = len([col for col in df2.columns if col.startswith('%-') and not any(x in col for x in ['rsi', 'mfi', 'adx'])])
    print(f"   Basic features: {basic_features}")
    
    print("3. Testing standard...")
    df3 = strategy.feature_engineering_standard(df2.copy(), {})
    standard_features = len([col for col in df3.columns if col.startswith('%-') and any(x in col for x in ['day_of_week', 'hour_of_day'])])
    print(f"   Standard features: {standard_features}")
    
    print("4. Testing targets...")
    df4 = strategy.set_freqai_targets(df3.copy(), {})
    targets = len([col for col in df4.columns if col.startswith('&-')])
    print(f"   Targets created: {targets}")
    
    print(f"\nTotal columns: {len(df4.columns)}")
    print(f"Original columns: {len(df.columns)}")
    print(f"New features: {len(df4.columns) - len(df.columns)}")
    
    # Show some sample features
    print(f"\nSample expand_all features: {[col for col in df1.columns if col.startswith('%-')][:5]}")
    print(f"Sample targets: {[col for col in df4.columns if col.startswith('&-')]}")
    
    print("\n=== Test completed successfully! ===")
    return True

if __name__ == "__main__":
    test_freqai_strategy()
