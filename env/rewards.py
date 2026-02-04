"""
Configurable reward functions for orbital maneuvers.

All reward functions take observation/state tensors and action tensors,
returning a scalar float64 tensor reward. Designed for use with
reinforcement learning environments.

Reward types:
    - station_keeping: Minimize relative orbital element (ROE) magnitude
    - orbit_raising: Reward positive semi-major axis changes
    - deorbit: Reward altitude decrease toward a target altitude
"""

import torch
from typing import Optional


def station_keeping_reward(
    roe: torch.Tensor,
    action: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    fuel_penalty: float = 0.1,
) -> torch.Tensor:
    """
    Reward for maintaining a reference orbit by minimizing ROE magnitude.

    reward = -weighted_norm(roe) - fuel_penalty * |action|

    Args:
        roe: (6,) tensor of quasi-nonsingular relative orbital elements
             [da, dl, dex, dey, dix, diy].
        action: (3,) tensor of RTN thrust commands.
        weights: (6,) tensor of weights for each ROE component.
                 Defaults to [1, 1, 1, 1, 1, 1].
        fuel_penalty: Scalar penalty coefficient for fuel usage.

    Returns:
        Scalar float64 tensor reward.
    """
    roe = roe.to(torch.float64)
    action = action.to(torch.float64)

    if weights is None:
        weights = torch.ones(6, dtype=torch.float64, device=roe.device)
    else:
        weights = weights.to(dtype=torch.float64, device=roe.device)

    weighted_roe = weights * roe
    roe_norm = torch.norm(weighted_roe)
    action_norm = torch.norm(action)

    reward = -roe_norm - fuel_penalty * action_norm
    return reward


def orbit_raising_reward(
    delta_a: torch.Tensor,
    action: torch.Tensor,
    scale: float = 1.0,
    fuel_penalty: float = 0.1,
) -> torch.Tensor:
    """
    Reward for increasing the semi-major axis.

    reward = scale * delta_a - fuel_penalty * |action|

    Args:
        delta_a: Scalar tensor, change in semi-major axis [m] over the step.
        action: (3,) tensor of RTN thrust commands.
        scale: Scaling factor for the semi-major axis change reward.
        fuel_penalty: Scalar penalty coefficient for fuel usage.

    Returns:
        Scalar float64 tensor reward.
    """
    delta_a = delta_a.to(torch.float64)
    action = action.to(torch.float64)

    action_norm = torch.norm(action)
    reward = scale * delta_a - fuel_penalty * action_norm
    return reward


def deorbit_reward(
    altitude: torch.Tensor,
    target_altitude: torch.Tensor,
    action: torch.Tensor,
    fuel_penalty: float = 0.1,
) -> torch.Tensor:
    """
    Reward for decreasing altitude toward a target.

    reward = -|altitude - target_altitude| - fuel_penalty * |action|

    Args:
        altitude: Scalar tensor, current altitude [m].
        target_altitude: Scalar tensor, target altitude [m].
        action: (3,) tensor of RTN thrust commands.
        fuel_penalty: Scalar penalty coefficient for fuel usage.

    Returns:
        Scalar float64 tensor reward.
    """
    altitude = altitude.to(torch.float64)
    target_altitude = target_altitude.to(torch.float64)
    action = action.to(torch.float64)

    alt_error = torch.abs(altitude - target_altitude)
    action_norm = torch.norm(action)

    reward = -alt_error - fuel_penalty * action_norm
    return reward


class RewardFunction:
    """
    Configurable reward function wrapper that dispatches to specific
    reward implementations based on a configuration dictionary.

    Config keys:
        type: str
            One of 'station_keeping', 'orbit_raising', 'deorbit'.
        weights: list[float] or None
            ROE weights for station_keeping. Length 6. Default: [1,1,1,1,1,1].
        fuel_penalty: float
            Penalty coefficient for thrust magnitude. Default: 0.1.
        scale: float
            Scale factor for orbit_raising reward. Default: 1.0.
        target_altitude: float
            Target altitude in meters for deorbit. Default: 150000.0 (150 km).

    Example:
        config = {
            'type': 'station_keeping',
            'weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'fuel_penalty': 0.1,
        }
        reward_fn = RewardFunction(config)
        reward = reward_fn(obs_dict, action)
    """

    def __init__(self, config: dict):
        self.config = config
        self.reward_type = config.get("type", "station_keeping")
        self.fuel_penalty = config.get("fuel_penalty", 0.1)

        # Station-keeping weights
        w = config.get("weights", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.weights = torch.tensor(w, dtype=torch.float64)

        # Orbit-raising scale
        self.scale = config.get("scale", 1.0)

        # Deorbit target altitude [m]
        self.target_altitude = torch.tensor(
            config.get("target_altitude", 150000.0), dtype=torch.float64
        )

        # Validate type
        valid_types = {"station_keeping", "orbit_raising", "deorbit"}
        if self.reward_type not in valid_types:
            raise ValueError(
                f"Unknown reward type '{self.reward_type}'. "
                f"Must be one of {valid_types}."
            )

    def __call__(self, obs_dict: dict, action: torch.Tensor) -> torch.Tensor:
        """
        Compute reward by dispatching to the appropriate reward function.

        Args:
            obs_dict: Dictionary containing observation data. Expected keys
                      depend on the reward type:
                - station_keeping: 'roe' -> (6,) tensor
                - orbit_raising: 'delta_a' -> scalar tensor
                - deorbit: 'altitude' -> scalar tensor (meters)
            action: (3,) tensor of RTN thrust commands.

        Returns:
            Scalar float64 tensor reward.
        """
        if self.reward_type == "station_keeping":
            roe = obs_dict["roe"]
            return station_keeping_reward(
                roe=roe,
                action=action,
                weights=self.weights.to(device=roe.device),
                fuel_penalty=self.fuel_penalty,
            )

        elif self.reward_type == "orbit_raising":
            delta_a = obs_dict["delta_a"]
            return orbit_raising_reward(
                delta_a=delta_a,
                action=action,
                scale=self.scale,
                fuel_penalty=self.fuel_penalty,
            )

        elif self.reward_type == "deorbit":
            altitude = obs_dict["altitude"]
            return deorbit_reward(
                altitude=altitude,
                target_altitude=self.target_altitude.to(device=altitude.device),
                action=action,
                fuel_penalty=self.fuel_penalty,
            )
