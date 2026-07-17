"""
inference package — anomaly scoring pipeline for trained autoencoders.

Pipeline
--------
FrameScorer          : protocol — per-frame anomaly score interface
ReconstructionFrameScorer : MSE reconstruction error scorer (DCASE baseline)
MahalanobisFrameScorer    : Mahalanobis distance scorer in latent space
aggregation          : frame errors → single file score  (mean / max / percentile)
FileScorer           : orchestrates MelFrameDataset construction, FrameScorer, aggregation
AnomalyDetector      : file score → AnomalyResult (threshold comparison)

Public API
----------
FrameScorer              : structural protocol for frame scorers
ReconstructionFrameScorer: MSE-based concrete scorer
MahalanobisFrameScorer   : Mahalanobis-based concrete scorer
FileScorer               : file-level scoring (frames → aggregate score)
make_mel_dataset_factory : factory helper for MelFrameDataset
AnomalyDetector          : threshold-based anomaly decision
AnomalyResult            : dataclass holding path, score, and is_anomaly flag
mean_score               : built-in mean aggregation function
max_score                : built-in max aggregation function
percentile_score         : factory for percentile-based aggregation
AggregationFn            : type alias for aggregation callables
"""

from inference.aggregation import AggregationFn, max_score, mean_score, percentile_score
from inference.detector import AnomalyDetector, AnomalyResult
from inference.file_scorer import FileScorer, make_mel_dataset_factory
from inference.frame_scorer import FrameScorer, MahalanobisFrameScorer, ReconstructionFrameScorer

__all__ = [
    "FrameScorer",
    "ReconstructionFrameScorer",
    "MahalanobisFrameScorer",
    "FileScorer",
    "make_mel_dataset_factory",
    "AnomalyDetector",
    "AnomalyResult",
    "mean_score",
    "max_score",
    "percentile_score",
    "AggregationFn",
]
