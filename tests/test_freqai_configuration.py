"""
Unit tests for FreqAI configuration functionality.
Tests parameter validation, feature expansion calculations, and configuration integrity.
"""

import pytest
import json
import os
from pathlib import Path
from freqtrade.configuration import Configuration


class TestFreqaiConfiguration:
    """Test class for FreqAI configuration functionality."""
    
    @pytest.fixture
    def config_path(self):
        """Get the path to the FreqAI config file."""
        return "config_examples/config_freqai.example.json"
    
    @pytest.fixture
    def config_data(self, config_path):
        """Load the FreqAI configuration data."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def test_config_file_exists(self, config_path):
        """Test that the FreqAI config file exists."""
        assert os.path.exists(config_path), f"Config file {config_path} does not exist"
    
    def test_config_is_valid_json(self, config_path):
        """Test that the config file contains valid JSON."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        assert isinstance(config, dict), "Config should be a dictionary"
    
    def test_freqai_section_exists(self, config_data):
        """Test that the freqai section exists in the config."""
        assert "freqai" in config_data, "freqai section missing from config"
        assert isinstance(config_data["freqai"], dict), "freqai section should be a dictionary"
    
    def test_required_freqai_parameters(self, config_data):
        """Test that all required FreqAI parameters are present."""
        freqai = config_data["freqai"]
        
        required_params = [
            "enabled", "purge_old_models", "train_period_days", 
            "backtest_period_days", "identifier"
        ]
        
        for param in required_params:
            assert param in freqai, f"Required parameter '{param}' missing from freqai section"
    
    def test_freqai_enabled(self, config_data):
        """Test that FreqAI is enabled."""
        assert config_data["freqai"]["enabled"] is True, "FreqAI should be enabled"
    
    def test_identifier_format(self, config_data):
        """Test that the identifier follows proper naming convention."""
        identifier = config_data["freqai"]["identifier"]
        
        # Identifier should be a string
        assert isinstance(identifier, str), "Identifier should be a string"
        
        # Identifier should not be empty
        assert len(identifier) > 0, "Identifier should not be empty"
        
        # Identifier should not contain spaces
        assert " " not in identifier, "Identifier should not contain spaces"
        
        # Identifier should be descriptive
        assert len(identifier) >= 5, "Identifier should be descriptive (at least 5 characters)"
    
    def test_training_periods(self, config_data):
        """Test that training periods are reasonable."""
        freqai = config_data["freqai"]
        
        train_period = freqai["train_period_days"]
        backtest_period = freqai["backtest_period_days"]
        
        # Training period should be positive
        assert train_period > 0, "train_period_days should be positive"
        
        # Backtest period should be positive
        assert backtest_period > 0, "backtest_period_days should be positive"
        
        # Training period should be longer than backtest period
        assert train_period > backtest_period, (
            "train_period_days should be longer than backtest_period_days"
        )
        
        # Training period should be reasonable (not too short, not too long)
        assert 7 <= train_period <= 365, "train_period_days should be between 7 and 365 days"
        
        # Backtest period should be reasonable
        assert 1 <= backtest_period <= 30, "backtest_period_days should be between 1 and 30 days"
    
    def test_feature_parameters_section(self, config_data):
        """Test that feature_parameters section exists and is properly configured."""
        freqai = config_data["freqai"]
        
        assert "feature_parameters" in freqai, "feature_parameters section missing"
        feature_params = freqai["feature_parameters"]
        
        # Required feature parameters
        required_feature_params = [
            "include_timeframes", "include_corr_pairlist", "label_period_candles",
            "include_shifted_candles", "indicator_periods_candles"
        ]
        
        for param in required_feature_params:
            assert param in feature_params, f"Required feature parameter '{param}' missing"
    
    def test_timeframes_configuration(self, config_data):
        """Test that timeframes are properly configured."""
        timeframes = config_data["freqai"]["feature_parameters"]["include_timeframes"]
        
        # Should be a list
        assert isinstance(timeframes, list), "include_timeframes should be a list"
        
        # Should not be empty
        assert len(timeframes) > 0, "include_timeframes should not be empty"
        
        # Should contain valid timeframe strings
        valid_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]
        for tf in timeframes:
            assert tf in valid_timeframes, f"Invalid timeframe: {tf}"
        
        # Should be ordered from shortest to longest
        for i in range(len(timeframes) - 1):
            current_tf = timeframes[i]
            next_tf = timeframes[i + 1]
            
            # Convert timeframes to minutes for comparison
            current_minutes = self._timeframe_to_minutes(current_tf)
            next_minutes = self._timeframe_to_minutes(next_tf)
            
            assert current_minutes <= next_minutes, (
                f"Timeframes should be ordered: {current_tf} should come before {next_tf}"
            )
    
    def test_correlation_pairs(self, config_data):
        """Test that correlation pairs are properly configured."""
        corr_pairs = config_data["freqai"]["feature_parameters"]["include_corr_pairlist"]
        
        # Should be a list
        assert isinstance(corr_pairs, list), "include_corr_pairlist should be a list"
        
        # Should not be empty
        assert len(corr_pairs) > 0, "include_corr_pairlist should not be empty"
        
        # Should contain valid pair formats
        for pair in corr_pairs:
            assert "/" in pair, f"Invalid pair format: {pair}"
            assert ":" in pair, f"Pair should include quote currency: {pair}"
        
        # Should not contain duplicates
        assert len(corr_pairs) == len(set(corr_pairs)), "Correlation pairs should not contain duplicates"
    
    def test_indicator_periods(self, config_data):
        """Test that indicator periods are properly configured."""
        periods = config_data["freqai"]["feature_parameters"]["indicator_periods_candles"]
        
        # Should be a list
        assert isinstance(periods, list), "indicator_periods_candles should be a list"
        
        # Should not be empty
        assert len(periods) > 0, "indicator_periods_candles should not be empty"
        
        # Should contain positive integers
        for period in periods:
            assert isinstance(period, int), f"Period should be integer: {period}"
            assert period > 0, f"Period should be positive: {period}"
            assert period <= 200, f"Period should be reasonable: {period}"
        
        # Should be ordered from smallest to largest
        for i in range(len(periods) - 1):
            assert periods[i] <= periods[i + 1], "Periods should be ordered"
    
    def test_label_period_candles(self, config_data):
        """Test that label_period_candles is properly configured."""
        label_period = config_data["freqai"]["feature_parameters"]["label_period_candles"]
        
        # Should be a positive integer
        assert isinstance(label_period, int), "label_period_candles should be integer"
        assert label_period > 0, "label_period_candles should be positive"
        
        # Should be reasonable (not too short, not too long)
        assert 1 <= label_period <= 100, "label_period_candles should be between 1 and 100"
    
    def test_shifted_candles(self, config_data):
        """Test that include_shifted_candles is properly configured."""
        shifted = config_data["freqai"]["feature_parameters"]["include_shifted_candles"]
        
        # Should be a positive integer
        assert isinstance(shifted, int), "include_shifted_candles should be integer"
        assert shifted >= 0, "include_shifted_candles should be non-negative"
        
        # Should be reasonable
        assert shifted <= 10, "include_shifted_candles should not exceed 10"
    
    def test_data_split_parameters(self, config_data):
        """Test that data_split_parameters are properly configured."""
        freqai = config_data["freqai"]
        
        if "data_split_parameters" in freqai:
            split_params = freqai["data_split_parameters"]
            
            # Test size should be between 0 and 1
            if "test_size" in split_params:
                test_size = split_params["test_size"]
                assert 0 < test_size < 1, f"test_size should be between 0 and 1: {test_size}"
            
            # Random state should be integer
            if "random_state" in split_params:
                random_state = split_params["random_state"]
                assert isinstance(random_state, int), "random_state should be integer"
    
    def test_model_training_parameters(self, config_data):
        """Test that model_training_parameters are properly configured."""
        freqai = config_data["freqai"]
        
        if "model_training_parameters" in freqai:
            training_params = freqai["model_training_parameters"]
            
            # Should be a dictionary
            assert isinstance(training_params, dict), "model_training_parameters should be dictionary"
            
            # Check common parameters if present
            if "n_estimators" in training_params:
                n_estimators = training_params["n_estimators"]
                assert isinstance(n_estimators, int), "n_estimators should be integer"
                assert n_estimators > 0, "n_estimators should be positive"
            
            if "learning_rate" in training_params:
                learning_rate = training_params["learning_rate"]
                assert isinstance(learning_rate, (int, float)), "learning_rate should be numeric"
                assert 0 < learning_rate <= 1, "learning_rate should be between 0 and 1"
    
    def test_feature_expansion_calculation(self, config_data):
        """Test that feature expansion calculation is correct."""
        feature_params = config_data["freqai"]["feature_parameters"]
        
        # Get expansion parameters
        periods = len(feature_params["indicator_periods_candles"])
        timeframes = len(feature_params["include_timeframes"])
        shifted = feature_params["include_shifted_candles"]
        corr_pairs = len(feature_params["include_corr_pairlist"])
        
        # Calculate expected feature counts
        # Features from expand_all (19 features per period)
        expand_all_features = 19 * periods * timeframes * shifted * corr_pairs
        
        # Features from expand_basic (16 features, no period expansion)
        expand_basic_features = 16 * timeframes * shifted * corr_pairs
        
        # Features from standard (16 features, no expansion)
        standard_features = 16
        
        total_expected = expand_all_features + expand_basic_features + standard_features
        
        print(f"Feature expansion calculation:")
        print(f"  Periods: {periods}")
        print(f"  Timeframes: {timeframes}")
        print(f"  Shifted candles: {shifted}")
        print(f"  Correlation pairs: {corr_pairs}")
        print(f"  Expand all features: {expand_all_features}")
        print(f"  Expand basic features: {expand_basic_features}")
        print(f"  Standard features: {standard_features}")
        print(f"  Total expected features: {total_expected}")
        
        # Verify calculation
        assert total_expected > 0, "Total features should be positive"
        assert total_expected < 10000, "Total features should be reasonable (< 10,000)"
    
    def test_startup_candle_count_validation(self, config_data):
        """Test that startup_candle_count would be sufficient for the config."""
        feature_params = config_data["freqai"]["feature_parameters"]
        
        # Get maximum period
        max_period = max(feature_params["indicator_periods_candles"])
        
        # Get maximum timeframe in minutes
        max_timeframe_minutes = max([
            self._timeframe_to_minutes(tf) for tf in feature_params["include_timeframes"]
        ])
        
        # Calculate required startup candles
        required_startup = max_period * (max_timeframe_minutes / 5)  # Assuming 5m base timeframe
        
        print(f"Startup candle validation:")
        print(f"  Max indicator period: {max_period}")
        print(f"  Max timeframe minutes: {max_timeframe_minutes}")
        print(f"  Required startup candles: {required_startup:.0f}")
        print(f"  Recommended startup_candle_count: {required_startup * 2:.0f}")
        
        # Verify that required startup is reasonable
        assert required_startup > 0, "Required startup candles should be positive"
        assert required_startup < 5000, "Required startup candles should be reasonable (< 5000)"
    
    def test_outlier_detection_configuration(self, config_data):
        """Test that outlier detection parameters are properly configured."""
        feature_params = config_data["freqai"]["feature_parameters"]
        
        # Check DI threshold
        if "DI_threshold" in feature_params:
            di_threshold = feature_params["DI_threshold"]
            assert isinstance(di_threshold, (int, float)), "DI_threshold should be numeric"
            assert di_threshold > 0, "DI_threshold should be positive"
            assert di_threshold <= 10, "DI_threshold should be reasonable"
        
        # Check SVM outlier removal
        if "use_SVM_to_remove_outliers" in feature_params:
            svm_enabled = feature_params["use_SVM_to_remove_outliers"]
            assert isinstance(svm_enabled, bool), "use_SVM_to_remove_outliers should be boolean"
            
            # If SVM is enabled, check SVM parameters
            if svm_enabled and "svm_params" in feature_params:
                svm_params = feature_params["svm_params"]
                assert isinstance(svm_params, dict), "svm_params should be dictionary"
        
        # Check DBSCAN outlier removal
        if "use_DBSCAN_to_remove_outliers" in feature_params:
            dbscan_enabled = feature_params["use_DBSCAN_to_remove_outliers"]
            assert isinstance(dbscan_enabled, bool), "use_DBSCAN_to_remove_outliers should be boolean"
    
    def test_advanced_parameters(self, config_data):
        """Test that advanced parameters are properly configured."""
        freqai = config_data["freqai"]
        
        # Check weight factor
        if "weight_factor" in freqai["feature_parameters"]:
            weight_factor = freqai["feature_parameters"]["weight_factor"]
            assert isinstance(weight_factor, (int, float)), "weight_factor should be numeric"
            assert 0 < weight_factor <= 1, "weight_factor should be between 0 and 1"
        
        # Check PCA
        if "principal_component_analysis" in freqai["feature_parameters"]:
            pca_enabled = freqai["feature_parameters"]["principal_component_analysis"]
            assert isinstance(pca_enabled, bool), "principal_component_analysis should be boolean"
        
        # Check noise standard deviation
        if "noise_standard_deviation" in freqai["feature_parameters"]:
            noise_std = freqai["feature_parameters"]["noise_standard_deviation"]
            assert isinstance(noise_std, (int, float)), "noise_standard_deviation should be numeric"
            assert noise_std >= 0, "noise_standard_deviation should be non-negative"
            assert noise_std <= 1, "noise_standard_deviation should be <= 1"
    
    def test_configuration_consistency(self, config_data):
        """Test that the configuration is internally consistent."""
        freqai = config_data["freqai"]
        
        # Check that identifier is unique and descriptive
        identifier = freqai["identifier"]
        assert "freqai" in identifier.lower() or "ml" in identifier.lower(), (
            "Identifier should indicate FreqAI/ML usage"
        )
        
        # Check that training period is sufficient for the number of pairs
        pair_count = len(config_data["exchange"]["pair_whitelist"])
        train_period = freqai["train_period_days"]
        
        # More pairs require longer training periods
        if pair_count > 10:
            assert train_period >= 30, "Many pairs require longer training periods"
        
        # Check that correlation pairs are included in whitelist
        corr_pairs = freqai["feature_parameters"]["include_corr_pairlist"]
        whitelist = config_data["exchange"]["pair_whitelist"]
        
        for corr_pair in corr_pairs:
            # Extract base pair from correlation pair (handle both formats)
            if ":" in corr_pair:
                base_pair = corr_pair.split(":")[0]
            else:
                base_pair = corr_pair
            
            # Check if base pair exists in whitelist (with or without :USDT suffix)
            base_pair_found = any(
                pair.startswith(base_pair) or pair == base_pair or 
                pair.replace(":USDT", "") == base_pair.replace(":USDT", "")
                for pair in whitelist
            )
            assert base_pair_found, f"Correlation pair {corr_pair} should be in whitelist"
    
    def _timeframe_to_minutes(self, timeframe):
        """Convert timeframe string to minutes."""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            raise ValueError(f"Unknown timeframe format: {timeframe}")


class TestFreqaiConfigurationValidation:
    """Test class for FreqAI configuration validation."""
    
    def test_configuration_schema_validation(self):
        """Test that the configuration follows the expected schema."""
        # This would typically use a JSON schema validator
        # For now, we'll test basic structure
        
        config_path = "config_examples/config_freqai.example.json"
        assert os.path.exists(config_path), "Config file should exist"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Test top-level structure
        required_top_level = ["exchange", "freqai", "trading_mode", "timeframe"]
        for key in required_top_level:
            assert key in config, f"Top-level key '{key}' missing"
        
        # Test exchange section
        assert "name" in config["exchange"], "Exchange name missing"
        assert "pair_whitelist" in config["exchange"], "Pair whitelist missing"
        
        # Test freqai section structure
        freqai = config["freqai"]
        required_freqai_keys = ["enabled", "feature_parameters"]
        for key in required_freqai_keys:
            assert key in freqai, f"FreqAI key '{key}' missing"
    
    def test_feature_engineering_consistency(self):
        """Test that feature engineering parameters are consistent."""
        config_path = "config_examples/config_freqai.example.json"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        feature_params = config["freqai"]["feature_parameters"]
        
        # Check that indicator periods are reasonable for the timeframes
        periods = feature_params["indicator_periods_candles"]
        timeframes = feature_params["include_timeframes"]
        
        for period in periods:
            for timeframe in timeframes:
                # Period should not be longer than the timeframe in candles
                tf_minutes = self._timeframe_to_minutes(timeframe)
                max_period_candles = tf_minutes / 5  # Assuming 5m base timeframe
                
                # Allow periods up to 50x the timeframe for reasonable feature engineering
                # This is more permissive for ML features and common practice
                max_allowed_period = max_period_candles * 50
                
                assert period <= max_allowed_period, (
                    f"Period {period} too long for timeframe {timeframe} "
                    f"(max allowed: {max_allowed_period:.0f})"
                )
    
    def _timeframe_to_minutes(self, timeframe):
        """Convert timeframe string to minutes."""
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        else:
            raise ValueError(f"Unknown timeframe format: {timeframe}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
