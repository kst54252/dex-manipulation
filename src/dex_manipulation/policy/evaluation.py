"""Trajectory-level tracking metrics, independent of the simulator and RSI."""
import numpy as np


def full_horizon_metrics(keypoint_error, failure, height, target_height, *, mean_threshold_m=.005, peak_threshold_m=.025):
    """Failures remain in the denominator; short RSI episodes cannot pass."""
    error,failure,height,target=map(np.asarray,(keypoint_error,failure,height,target_height))
    if error.ndim!=2 or not error.size or any(a.shape!=error.shape for a in (failure,height,target)):
        raise ValueError('Expected nonempty [time, environment] evaluation arrays')
    if not all(np.isfinite(a).all() for a in (error,height,target)) or (error<0).any():
        raise ValueError('Nonfinite or negative evaluation errors')
    complete=~failure.any(axis=0)
    means,peaks=error.mean(axis=0),error.max(axis=0)
    good=complete&(means<=mean_threshold_m)&(peaks<=peak_threshold_m)
    return dict(frames=error.shape[0],episodes=error.shape[1],includes_failed_frames=True,
        mean_error_mm=float(error.mean()*1000),max_error_mm=float(error.max()*1000),
        per_episode_mean_error_mm=(means*1000).tolist(),per_episode_max_error_mm=(peaks*1000).tolist(),
        completed_count=int(complete.sum()),within_mean_and_peak_tolerances_count=int(good.sum()),
        mean_threshold_mm=mean_threshold_m*1000,peak_threshold_mm=peak_threshold_m*1000,
        mean_final_height_error_mm=float(np.abs(height[-1]-target[-1]).mean()*1000),
        mean_final_lift_mm=float((height[-1]-height[0]).mean()*1000),
        failed_environment_indices=np.flatnonzero(~complete).tolist(),
        interpretation='Dynamic object tracking only; does not certify joint-speed limits, nonpenetration or real-world grasp robustness.')
