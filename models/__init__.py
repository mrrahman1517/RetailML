"""
Retail ML Models Package
"""

from .demand_forecasting import DemandForecastingModel
from .markdown_optimization import MarkdownOptimizationModel
from .clv_retention import CLVModel
from .recommendations import RecommendationSystem
from .returns_prediction import ReturnsPredictionModel
from .fraud_detection import FraudDetectionModel
from .omnichannel import OmnichannelFulfillmentModel
from .merchandising import MerchandisingForecastModel

__all__ = [
    'DemandForecastingModel',
    'MarkdownOptimizationModel',
    'CLVModel',
    'RecommendationSystem',
    'ReturnsPredictionModel',
    'FraudDetectionModel',
    'OmnichannelFulfillmentModel',
    'MerchandisingForecastModel'
]

