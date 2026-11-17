"""
Track Conditions Model.

Treats Green Flag, Virtual Safety Car (VSC), and Safety Car (SC) as three distinct physical states:
1. Green Flag:
   - Pace multiplier: 1.0x (full racing pace)
   - Gap dynamics: Natural divergence based on driver pace and tyre degradation
   - Bunching: 0.0 (none)

2. Virtual Safety Car (VSC):
   - Track status codes 6 / 7
   - Pace multiplier: 1.35x (cars slowed to maintain delta time)
   - Gap dynamics: Gaps are FROZEN (relative time intervals between cars preserved)
   - Bunching: 0.0 (no compression behind a physical pace car)

3. Safety Car (SC):
   - Track status code 4
   - Pace multiplier: 1.45x (field led by physical safety car)
   - Gap dynamics: Field BUNCHES (gaps compress towards target queue spacing of ~0.8s)
   - Bunching rate: 70% compression of excess gap per SC lap until queued
   - Restart transition: racing resumes at the line with bunched field
"""

from enum import Enum


class TrackStatus(str, Enum):
    GREEN = "GREEN"
    VSC = "VSC"
    SC = "SC"


class TrackConditionModel:
    def __init__(self):
        self.slowdown_factors = {
            TrackStatus.GREEN: 1.0,
            TrackStatus.VSC: 1.35,
            TrackStatus.SC: 1.45,
        }
        self.bunching_target_gap = 0.8  # seconds between cars in queue
        self.bunching_compression_rate = 0.70  # fraction of excess gap compressed per SC lap
        self.assumptions = {
            "green_flag_slowdown_factor": 1.0,
            "vsc_slowdown_factor": 1.35,
            "sc_slowdown_factor": 1.45,
            "sc_bunching_target_gap_sec": self.bunching_target_gap,
            "sc_bunching_compression_rate": self.bunching_compression_rate,
            "vsc_bunching_behavior": "Gaps strictly frozen (relative time deltas preserved, 0% compression)",
        }

    def get_slowdown_multiplier(self, condition: TrackStatus) -> float:
        return self.slowdown_factors.get(condition, 1.0)

    def apply_sc_bunching(self, cumulative_times: list) -> list:
        """
        Compresses gaps between consecutive cars towards target queue spacing (0.8s).
        cumulative_times: list of tuples (driver_number, cum_time), sorted by cum_time ascending.
        Returns updated cumulative times with bunching compression applied.
        """
        if not cumulative_times or len(cumulative_times) <= 1:
            return cumulative_times

        bunched = []
        leader_dn, leader_time = cumulative_times[0]
        bunched.append((leader_dn, leader_time))

        prev_time = leader_time
        for dn, t in cumulative_times[1:]:
            current_gap = t - prev_time
            if current_gap > self.bunching_target_gap:
                excess = current_gap - self.bunching_target_gap
                compressed_gap = self.bunching_target_gap + (excess * (1.0 - self.bunching_compression_rate))
                new_time = prev_time + compressed_gap
            else:
                new_time = t
            bunched.append((dn, round(new_time, 3)))
            prev_time = new_time

        return bunched

    def get_assumptions(self) -> dict:
        return self.assumptions
