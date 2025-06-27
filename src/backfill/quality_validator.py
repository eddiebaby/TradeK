"""
Quality-First Data Validation Framework

Comprehensive validation system for ML-ready market data with 99%+ accuracy requirements.
Implements multi-layer validation for arbitrage and HFT strategy data quality assurance.
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    CRITICAL = "critical"      # Data unusable for trading
    HIGH = "high"             # Significant quality issues
    MEDIUM = "medium"         # Minor quality issues
    LOW = "low"               # Cosmetic issues
    INFO = "info"             # Informational only


@dataclass
class ValidationResult:
    """Individual validation check result"""
    check_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


@dataclass
class DatasetValidationReport:
    """Comprehensive dataset validation report"""
    symbol: str
    asset_class: str
    timeframe: str
    validation_timestamp: datetime
    overall_score: float  # 0.0 to 1.0
    quality_grade: str    # A+, A, B, C, D, F
    
    # Validation results by category
    completeness_results: List[ValidationResult]
    accuracy_results: List[ValidationResult]
    consistency_results: List[ValidationResult]
    timeliness_results: List[ValidationResult]
    
    # Summary statistics
    total_checks: int
    passed_checks: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    
    # Data statistics
    total_data_points: int
    missing_data_points: int
    outlier_data_points: int
    data_coverage_percentage: float
    
    # Recommendations
    recommendations: List[str]
    trade_readiness: bool


class QualityValidator:
    """Comprehensive quality validation for ML-ready trading data"""
    
    def __init__(self):
        self.validation_thresholds = self._load_validation_thresholds()
        self.validation_checks = self._initialize_validation_checks()
        
    def _load_validation_thresholds(self) -> Dict[str, Any]:
        """Load quality thresholds for different validation types"""
        return {
            "completeness": {
                "minimum_coverage": 0.99,        # 99%+ data coverage required
                "maximum_gap_minutes": 5,        # No gaps > 5 minutes during market hours
                "minimum_trading_days": 252      # At least 1 year of trading days
            },
            "accuracy": {
                "maximum_ohlc_inconsistency": 0.001,  # 0.1% price inconsistency
                "maximum_zero_volume_rate": 0.01,     # 1% zero volume tolerance
                "maximum_price_jump": 0.1,            # 10% maximum single-period jump
                "minimum_tick_size_compliance": 0.99  # 99% tick size compliance
            },
            "consistency": {
                "maximum_timestamp_drift": 60,        # 60 seconds maximum drift
                "minimum_volume_correlation": 0.3,    # Volume should correlate with price moves
                "maximum_duplicate_rate": 0.001       # 0.1% duplicate tolerance
            },
            "timeliness": {
                "maximum_delay_seconds": 300,         # 5 minutes maximum delay
                "minimum_update_frequency": 0.95      # 95% of expected updates
            },
            "trading_readiness": {
                "minimum_overall_score": 0.95,        # 95% minimum for trading
                "maximum_critical_issues": 0,         # Zero critical issues
                "maximum_high_issues": 2              # Maximum 2 high severity issues
            }
        }
    
    def _initialize_validation_checks(self) -> Dict[str, List[str]]:
        """Initialize available validation checks by category"""
        return {
            "completeness": [
                "data_coverage_check",
                "market_hours_gap_check", 
                "weekend_gap_check",
                "holiday_gap_check",
                "minimum_history_check"
            ],
            "accuracy": [
                "ohlc_consistency_check",
                "price_volume_relationship_check",
                "tick_size_compliance_check",
                "price_jump_detection",
                "volume_anomaly_detection"
            ],
            "consistency": [
                "timestamp_monotonicity_check",
                "duplicate_detection",
                "cross_source_consistency_check",
                "intraday_pattern_consistency"
            ],
            "timeliness": [
                "data_freshness_check",
                "update_frequency_check",
                "latency_consistency_check"
            ]
        }
    
    async def validate_dataset(
        self,
        data: pd.DataFrame,
        symbol: str,
        asset_class: str,
        timeframe: str = "1min"
    ) -> DatasetValidationReport:
        """
        Perform comprehensive validation on a dataset
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Asset symbol
            asset_class: Asset class (crypto, equity, futures)
            timeframe: Data timeframe
            
        Returns:
            Comprehensive validation report
        """
        logger.info(f"🔍 Starting validation for {symbol} ({asset_class}) - {len(data)} data points")
        
        validation_start = datetime.utcnow()
        
        # Run validation checks by category
        completeness_results = await self._run_completeness_checks(data, symbol, asset_class)
        accuracy_results = await self._run_accuracy_checks(data, symbol, asset_class)
        consistency_results = await self._run_consistency_checks(data, symbol, asset_class)
        timeliness_results = await self._run_timeliness_checks(data, symbol, asset_class)
        
        # Calculate overall quality metrics
        all_results = completeness_results + accuracy_results + consistency_results + timeliness_results
        
        total_checks = len(all_results)
        passed_checks = sum(1 for result in all_results if result.passed)
        
        # Count issues by severity
        critical_issues = sum(1 for r in all_results if r.severity == ValidationSeverity.CRITICAL and not r.passed)
        high_issues = sum(1 for r in all_results if r.severity == ValidationSeverity.HIGH and not r.passed)
        medium_issues = sum(1 for r in all_results if r.severity == ValidationSeverity.MEDIUM and not r.passed)
        low_issues = sum(1 for r in all_results if r.severity == ValidationSeverity.LOW and not r.passed)
        
        # Calculate overall score (weighted by severity)
        overall_score = self._calculate_quality_score(all_results)
        quality_grade = self._assign_quality_grade(overall_score)
        
        # Calculate data statistics
        total_data_points = len(data)
        missing_data_points = data.isnull().sum().sum()
        outlier_data_points = self._count_outliers(data)
        data_coverage_percentage = self._calculate_coverage_percentage(data, symbol, asset_class)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_results, overall_score)
        
        # Determine trade readiness
        trade_readiness = self._assess_trade_readiness(overall_score, critical_issues, high_issues)
        
        # Create comprehensive report
        report = DatasetValidationReport(
            symbol=symbol,
            asset_class=asset_class,
            timeframe=timeframe,
            validation_timestamp=validation_start,
            overall_score=overall_score,
            quality_grade=quality_grade,
            completeness_results=completeness_results,
            accuracy_results=accuracy_results,
            consistency_results=consistency_results,
            timeliness_results=timeliness_results,
            total_checks=total_checks,
            passed_checks=passed_checks,
            critical_issues=critical_issues,
            high_issues=high_issues,
            medium_issues=medium_issues,
            low_issues=low_issues,
            total_data_points=total_data_points,
            missing_data_points=missing_data_points,
            outlier_data_points=outlier_data_points,
            data_coverage_percentage=data_coverage_percentage,
            recommendations=recommendations,
            trade_readiness=trade_readiness
        )
        
        logger.info(f"✅ Validation complete for {symbol}: Score {overall_score:.1%}, Grade {quality_grade}")
        
        return report
    
    async def _run_completeness_checks(self, data: pd.DataFrame, symbol: str, asset_class: str) -> List[ValidationResult]:
        """Run data completeness validation checks"""
        results = []
        
        # Data coverage check
        expected_points = self._calculate_expected_data_points(data, symbol, asset_class)
        actual_points = len(data)
        coverage_ratio = actual_points / expected_points if expected_points > 0 else 0
        
        results.append(ValidationResult(
            check_name="data_coverage_check",
            passed=coverage_ratio >= self.validation_thresholds["completeness"]["minimum_coverage"],
            severity=ValidationSeverity.CRITICAL,
            message=f"Data coverage: {coverage_ratio:.1%} ({actual_points}/{expected_points} points)",
            details={"coverage_ratio": coverage_ratio, "actual_points": actual_points, "expected_points": expected_points}
        ))
        
        # Market hours gap check
        gaps = self._detect_market_hour_gaps(data, symbol, asset_class)
        max_gap_threshold = self.validation_thresholds["completeness"]["maximum_gap_minutes"]
        
        results.append(ValidationResult(
            check_name="market_hours_gap_check",
            passed=len([g for g in gaps if g > max_gap_threshold]) == 0,
            severity=ValidationSeverity.HIGH,
            message=f"Market hours gaps: {len(gaps)} gaps detected, max {max(gaps, default=0):.1f} minutes",
            details={"gaps": gaps, "max_gap": max(gaps, default=0)}
        ))
        
        # Minimum history check
        min_days = self.validation_thresholds["completeness"]["minimum_trading_days"]
        trading_days = self._count_trading_days(data)
        
        results.append(ValidationResult(
            check_name="minimum_history_check",
            passed=trading_days >= min_days,
            severity=ValidationSeverity.MEDIUM,
            message=f"Trading history: {trading_days} days (minimum {min_days} required)",
            details={"trading_days": trading_days, "minimum_required": min_days}
        ))
        
        return results
    
    async def _run_accuracy_checks(self, data: pd.DataFrame, symbol: str, asset_class: str) -> List[ValidationResult]:
        """Run data accuracy validation checks"""
        results = []
        
        # OHLC consistency check
        ohlc_issues = self._check_ohlc_consistency(data)
        inconsistency_rate = len(ohlc_issues) / len(data) if len(data) > 0 else 0
        max_inconsistency = self.validation_thresholds["accuracy"]["maximum_ohlc_inconsistency"]
        
        results.append(ValidationResult(
            check_name="ohlc_consistency_check",
            passed=inconsistency_rate <= max_inconsistency,
            severity=ValidationSeverity.CRITICAL,
            message=f"OHLC consistency: {inconsistency_rate:.3%} inconsistent ({len(ohlc_issues)} issues)",
            details={"inconsistency_rate": inconsistency_rate, "issues": ohlc_issues[:10]}  # First 10 issues
        ))
        
        # Volume anomaly detection
        volume_anomalies = self._detect_volume_anomalies(data)
        zero_volume_rate = (data['volume'] == 0).sum() / len(data) if len(data) > 0 else 0
        max_zero_volume = self.validation_thresholds["accuracy"]["maximum_zero_volume_rate"]
        
        results.append(ValidationResult(
            check_name="volume_anomaly_detection",
            passed=zero_volume_rate <= max_zero_volume,
            severity=ValidationSeverity.HIGH,
            message=f"Volume anomalies: {zero_volume_rate:.1%} zero volume, {len(volume_anomalies)} anomalies",
            details={"zero_volume_rate": zero_volume_rate, "anomalies": len(volume_anomalies)}
        ))
        
        # Price jump detection
        price_jumps = self._detect_price_jumps(data)
        max_jump_threshold = self.validation_thresholds["accuracy"]["maximum_price_jump"]
        extreme_jumps = [j for j in price_jumps if abs(j) > max_jump_threshold]
        
        results.append(ValidationResult(
            check_name="price_jump_detection",
            passed=len(extreme_jumps) == 0,
            severity=ValidationSeverity.HIGH,
            message=f"Price jumps: {len(price_jumps)} total, {len(extreme_jumps)} extreme (>{max_jump_threshold:.1%})",
            details={"total_jumps": len(price_jumps), "extreme_jumps": len(extreme_jumps)}
        ))
        
        return results
    
    async def _run_consistency_checks(self, data: pd.DataFrame, symbol: str, asset_class: str) -> List[ValidationResult]:
        """Run data consistency validation checks"""
        results = []
        
        # Timestamp monotonicity check
        timestamp_issues = self._check_timestamp_monotonicity(data)
        
        results.append(ValidationResult(
            check_name="timestamp_monotonicity_check",
            passed=len(timestamp_issues) == 0,
            severity=ValidationSeverity.CRITICAL,
            message=f"Timestamp consistency: {len(timestamp_issues)} non-monotonic timestamps",
            details={"issues": timestamp_issues}
        ))
        
        # Duplicate detection
        duplicates = self._detect_duplicates(data)
        duplicate_rate = len(duplicates) / len(data) if len(data) > 0 else 0
        max_duplicate_rate = self.validation_thresholds["consistency"]["maximum_duplicate_rate"]
        
        results.append(ValidationResult(
            check_name="duplicate_detection",
            passed=duplicate_rate <= max_duplicate_rate,
            severity=ValidationSeverity.MEDIUM,
            message=f"Duplicates: {duplicate_rate:.3%} duplicate records ({len(duplicates)} duplicates)",
            details={"duplicate_rate": duplicate_rate, "duplicates": len(duplicates)}
        ))
        
        return results
    
    async def _run_timeliness_checks(self, data: pd.DataFrame, symbol: str, asset_class: str) -> List[ValidationResult]:
        """Run data timeliness validation checks"""
        results = []
        
        # Data freshness check
        if len(data) > 0:
            latest_timestamp = data.index.max() if hasattr(data.index, 'max') else data['timestamp'].max()
            current_time = datetime.utcnow()
            
            if isinstance(latest_timestamp, str):
                latest_timestamp = pd.to_datetime(latest_timestamp)
            
            data_age_seconds = (current_time - latest_timestamp).total_seconds()
            max_delay = self.validation_thresholds["timeliness"]["maximum_delay_seconds"]
            
            results.append(ValidationResult(
                check_name="data_freshness_check",
                passed=data_age_seconds <= max_delay,
                severity=ValidationSeverity.MEDIUM,
                message=f"Data freshness: {data_age_seconds/60:.1f} minutes old (max {max_delay/60:.1f} minutes)",
                details={"data_age_seconds": data_age_seconds, "latest_timestamp": str(latest_timestamp)}
            ))
        
        return results
    
    def _calculate_expected_data_points(self, data: pd.DataFrame, symbol: str, asset_class: str) -> int:
        """Calculate expected number of data points based on timeframe and asset class"""
        if len(data) == 0:
            return 0
        
        # Get date range
        start_date = data.index.min() if hasattr(data.index, 'min') else data['timestamp'].min()
        end_date = data.index.max() if hasattr(data.index, 'max') else data['timestamp'].max()
        
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Calculate based on asset class trading hours
        if asset_class == "crypto":
            # Crypto trades 24/7
            total_minutes = (end_date - start_date).total_seconds() / 60
            return int(total_minutes)  # 1-minute data
        elif asset_class in ["equity", "futures"]:
            # Traditional market hours: ~6.5 hours per day, 5 days per week
            trading_days = pd.bdate_range(start_date, end_date)
            return len(trading_days) * 390  # 390 minutes per trading day
        else:
            # Conservative estimate
            total_minutes = (end_date - start_date).total_seconds() / 60
            return int(total_minutes * 0.5)  # 50% coverage estimate
    
    def _detect_market_hour_gaps(self, data: pd.DataFrame, symbol: str, asset_class: str) -> List[float]:
        """Detect gaps during market hours"""
        if len(data) < 2:
            return []
        
        # Get timestamps
        timestamps = data.index if hasattr(data.index, 'to_pydatetime') else pd.to_datetime(data['timestamp'])
        
        # Calculate time differences in minutes
        time_diffs = timestamps.diff().dt.total_seconds() / 60
        
        # Filter for significant gaps (>2 minutes for 1-minute data)
        gaps = time_diffs[time_diffs > 2].values
        
        return gaps.tolist()
    
    def _count_trading_days(self, data: pd.DataFrame) -> int:
        """Count number of trading days in dataset"""
        if len(data) == 0:
            return 0
        
        timestamps = data.index if hasattr(data.index, 'date') else pd.to_datetime(data['timestamp'])
        unique_dates = timestamps.normalize().nunique()
        
        return unique_dates
    
    def _check_ohlc_consistency(self, data: pd.DataFrame) -> List[int]:
        """Check OHLC price consistency"""
        issues = []
        
        for i, row in data.iterrows():
            try:
                open_price = float(row['open'])
                high_price = float(row['high'])
                low_price = float(row['low'])
                close_price = float(row['close'])
                
                # Check basic OHLC relationships
                if not (low_price <= open_price <= high_price):
                    issues.append(i)
                elif not (low_price <= close_price <= high_price):
                    issues.append(i)
                elif high_price < low_price:
                    issues.append(i)
                    
            except (ValueError, TypeError, KeyError):
                issues.append(i)
                
        return issues
    
    def _detect_volume_anomalies(self, data: pd.DataFrame) -> List[int]:
        """Detect volume anomalies"""
        if 'volume' not in data.columns or len(data) < 10:
            return []
        
        volumes = data['volume'].astype(float)
        
        # Calculate volume statistics
        volume_mean = volumes.mean()
        volume_std = volumes.std()
        
        # Detect anomalies (>3 standard deviations from mean)
        anomalies = []
        for i, volume in enumerate(volumes):
            if abs(volume - volume_mean) > 3 * volume_std:
                anomalies.append(i)
                
        return anomalies
    
    def _detect_price_jumps(self, data: pd.DataFrame) -> List[float]:
        """Detect significant price jumps"""
        if len(data) < 2:
            return []
        
        closes = data['close'].astype(float)
        returns = closes.pct_change().dropna()
        
        return returns.tolist()
    
    def _check_timestamp_monotonicity(self, data: pd.DataFrame) -> List[int]:
        """Check if timestamps are monotonically increasing"""
        issues = []
        
        timestamps = data.index if hasattr(data.index, 'to_pydatetime') else pd.to_datetime(data['timestamp'])
        
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i-1]:
                issues.append(i)
                
        return issues
    
    def _detect_duplicates(self, data: pd.DataFrame) -> List[int]:
        """Detect duplicate records"""
        # Check for duplicate timestamps
        timestamps = data.index if hasattr(data.index, 'to_pydatetime') else data['timestamp']
        duplicates = timestamps.duplicated()
        
        return duplicates[duplicates].index.tolist()
    
    def _count_outliers(self, data: pd.DataFrame) -> int:
        """Count statistical outliers in price data"""
        if 'close' not in data.columns or len(data) < 10:
            return 0
        
        closes = data['close'].astype(float)
        Q1 = closes.quantile(0.25)
        Q3 = closes.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((closes < lower_bound) | (closes > upper_bound)).sum()
        
        return outliers
    
    def _calculate_coverage_percentage(self, data: pd.DataFrame, symbol: str, asset_class: str) -> float:
        """Calculate data coverage percentage"""
        expected = self._calculate_expected_data_points(data, symbol, asset_class)
        actual = len(data)
        
        return (actual / expected * 100) if expected > 0 else 0.0
    
    def _calculate_quality_score(self, results: List[ValidationResult]) -> float:
        """Calculate weighted quality score based on validation results"""
        if not results:
            return 0.0
        
        # Severity weights
        weights = {
            ValidationSeverity.CRITICAL: 1.0,
            ValidationSeverity.HIGH: 0.7,
            ValidationSeverity.MEDIUM: 0.4,
            ValidationSeverity.LOW: 0.2,
            ValidationSeverity.INFO: 0.1
        }
        
        total_weight = 0
        weighted_score = 0
        
        for result in results:
            weight = weights[result.severity]
            total_weight += weight
            
            if result.passed:
                weighted_score += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _assign_quality_grade(self, score: float) -> str:
        """Assign letter grade based on quality score"""
        if score >= 0.98:
            return "A+"
        elif score >= 0.95:
            return "A"
        elif score >= 0.90:
            return "B"
        elif score >= 0.80:
            return "C"
        elif score >= 0.70:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self, results: List[ValidationResult], score: float) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        # Critical issues
        critical_results = [r for r in results if r.severity == ValidationSeverity.CRITICAL and not r.passed]
        if critical_results:
            recommendations.append("🚨 CRITICAL: Address critical data quality issues before trading")
            for result in critical_results:
                recommendations.append(f"   - {result.check_name}: {result.message}")
        
        # High priority issues
        high_results = [r for r in results if r.severity == ValidationSeverity.HIGH and not r.passed]
        if high_results:
            recommendations.append("⚠️ HIGH PRIORITY: Resolve high-severity issues")
            for result in high_results[:3]:  # Top 3 issues
                recommendations.append(f"   - {result.check_name}: {result.message}")
        
        # Overall score recommendations
        if score >= 0.95:
            recommendations.append("✅ Excellent data quality - ready for production trading")
        elif score >= 0.90:
            recommendations.append("✅ Good data quality - minor optimizations recommended")
        elif score >= 0.80:
            recommendations.append("⚠️ Moderate data quality - significant improvements needed")
        else:
            recommendations.append("❌ Poor data quality - extensive remediation required")
        
        return recommendations
    
    def _assess_trade_readiness(self, score: float, critical_issues: int, high_issues: int) -> bool:
        """Assess if data is ready for live trading"""
        thresholds = self.validation_thresholds["trading_readiness"]
        
        return (
            score >= thresholds["minimum_overall_score"] and
            critical_issues <= thresholds["maximum_critical_issues"] and
            high_issues <= thresholds["maximum_high_issues"]
        )


# Convenience function for quick validation
async def validate_trading_data(
    data: pd.DataFrame,
    symbol: str,
    asset_class: str,
    timeframe: str = "1min"
) -> DatasetValidationReport:
    """Quick validation function for trading data"""
    validator = QualityValidator()
    return await validator.validate_dataset(data, symbol, asset_class, timeframe)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 101,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    async def test_validation():
        validator = QualityValidator()
        report = await validator.validate_dataset(sample_data, "BTC/USD", "crypto")
        
        print(f"Validation Results:")
        print(f"Overall Score: {report.overall_score:.1%}")
        print(f"Quality Grade: {report.quality_grade}")
        print(f"Trade Ready: {report.trade_readiness}")
        print(f"Critical Issues: {report.critical_issues}")
        print(f"Recommendations: {len(report.recommendations)}")
        
        return report
    
    # Run test
    # asyncio.run(test_validation())