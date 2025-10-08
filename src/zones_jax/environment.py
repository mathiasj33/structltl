"""A simple differentiable reinforcement learning environment built with JAX.

The environment simulates a point-mass agent moving in a 2D plane. The agent
applies a forward force aligned with its current heading and can control its
angular velocity. Observations expose the current acceleration, velocity, and
angular velocity, making the dynamics suitable for model-based control or
reinforcement learning experiments.

The environment is written as pure JAX functions so that transformations such
as ``jax.vmap`` and ``jax.jit`` can be applied directly to ``reset`` and
``step`` without additional wrappers.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

_EPS = 1e-8


class EnvParams(NamedTuple):
    """Container holding static environment parameters."""

    # Physics
    dt: float
    drag: float
    max_speed: float
    max_force: float
    max_angular_velocity: float
    # World
    agent_radius: float
    world_size: float
    spawn_size: float
    max_steps: int = 1000
    # Zones
    zone_radius: float = 0.4
    colors: tuple[str, str, str, str] = ("blue", "green", "magenta", "yellow")
    zones_per_color: int = 2
    keepout_radius: float = 0.55
    # Lidar
    num_lidar_bins: int = 16
    lidar_max_range: float | None = None  # default to world_size if None


class EnvState(NamedTuple):
    """Dynamical state of the environment."""

    # Physics
    position: jnp.ndarray  # shape: (2,)
    velocity: jnp.ndarray  # shape: (2,)
    angle: jnp.ndarray  # shape: ()
    angular_velocity: jnp.ndarray  # shape: ()
    acceleration: jnp.ndarray  # shape: (2,)
    # Zones (static for an episode)
    zone_centers: jnp.ndarray  # shape: (N, 2)
    zone_colors: jnp.ndarray  # shape: (N,) int in [0, C)
    # Episode length
    num_steps: jnp.ndarray = jnp.zeros((), dtype=jnp.int32)  # shape: ()


class EnvObservation(NamedTuple):
    """Observation emitted after reset or step."""

    acceleration: jnp.ndarray  # shape: (2,)
    velocity: jnp.ndarray  # shape: (2,)
    angular_velocity: jnp.ndarray  # shape: ()
    lidar: jnp.ndarray | None = None  # Lidar per color: (C, num_lidar_bins)


class EnvTransition(NamedTuple):
    """Bundle returned by a call to :func:`step`."""

    state: EnvState
    observation: EnvObservation
    reward: jnp.ndarray  # shape: ()
    done: jnp.ndarray  # shape: () boolean


def default_params() -> EnvParams:
    """Return a default set of environment parameters."""

    return EnvParams(
        dt=0.05,
        world_size=6.6,
        spawn_size=5.0,
        agent_radius=0.1,
        drag=0.08,
        max_speed=3.0,
        max_force=2.0,
        max_angular_velocity=3.0,
        zone_radius=0.4,
        zones_per_color=2,
        num_lidar_bins=16,
        lidar_max_range=None,
    )


_DEFAULT_PARAMS = default_params()
_MAX_ZONE_PLACEMENT_ITERS = 10000


def _wrap_angle(angle: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles to the ``(-pi, pi]`` interval."""

    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


@partial(jax.jit, static_argnames="params")
def reset(
    key: jax.Array, params: EnvParams | None = None
) -> tuple[EnvState, EnvObservation]:
    """Reset the environment state using ``key``.

    Args:
        key: PRNG key used for sampling the initial agent configuration.
        params: Optional override for the environment parameters.

    Returns:
        A tuple ``(state, observation)`` with batched ``jnp.ndarray`` leaves,
        enabling ``jax.vmap`` over the result without additional wrappers.
    """

    params = _DEFAULT_PARAMS if params is None else params
    key_zones, key_pos, key_angle = jax.random.split(key, 3)

    centers, colors = _sample_zones(key_zones, params)
    position = _sample_agent_position(key_pos, params, centers)
    velocity = jnp.zeros(2, dtype=jnp.float32)
    acceleration = jnp.zeros(2, dtype=jnp.float32)
    angle = jax.random.uniform(
        key_angle,
        shape=(),
        minval=-jnp.pi,
        maxval=jnp.pi,
    )
    angular_velocity = jnp.zeros((), dtype=jnp.float32)

    state = EnvState(
        position=position,
        velocity=velocity,
        angle=angle,
        angular_velocity=angular_velocity,
        acceleration=acceleration,
        zone_centers=centers,
        zone_colors=colors,
    )
    # Compute lidar for initial observation
    lidar = _compute_lidar(state, params)
    observation = EnvObservation(
        acceleration=acceleration,
        velocity=velocity,
        angular_velocity=angular_velocity,
        lidar=lidar,
    )
    return state, observation


def _sample_zones(key: jax.Array, params: EnvParams) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample non-overlapping zone centers and assign colors.

    Returns (centers:(Z,2), colors:(Z,)), with two zones per color.
    """
    num_colors = len(params.colors)
    total_zones = num_colors * params.zones_per_color
    minval = -params.spawn_size / 2 + params.keepout_radius
    maxval = params.spawn_size / 2 - params.keepout_radius
    keepout = params.keepout_radius

    centers0 = jnp.zeros((total_zones, 2), dtype=jnp.float32)
    colors = jnp.repeat(jnp.arange(num_colors, dtype=jnp.int32), params.zones_per_color)

    def cond_fun(carry):
        key, centers, count, it = carry
        return jnp.logical_and(count < total_zones, it < _MAX_ZONE_PLACEMENT_ITERS)

    def body_fun(carry):
        key, centers, count, it = carry
        key, sub = jax.random.split(key)
        proposal = jax.random.uniform(sub, (2,), minval=minval, maxval=maxval)
        idxs = jnp.arange(total_zones)
        mask = idxs < count
        dists = jnp.linalg.norm(centers - proposal, axis=1)
        cond_ok = dists >= 2.0 * keepout
        all_ok = jnp.all(jnp.logical_or(~mask, cond_ok))
        centers = lax.cond(
            all_ok,
            lambda c: c.at[count].set(proposal),
            lambda c: c,
            centers,
        )
        count = count + jnp.where(all_ok, 1, 0)
        return key, centers, count, it + 1

    key, centers, count, _ = lax.while_loop(
        cond_fun, body_fun, (key, centers0, jnp.int32(0), jnp.int32(0))
    )

    # Fallback: evenly space on a circle near border to avoid overlaps
    def fallback_centers(_centers):
        r = params.spawn_size / 2
        k = total_zones
        thetas = jnp.arange(k, dtype=jnp.float32) * (2.0 * jnp.pi / k)
        x = r * jnp.cos(thetas)
        y = r * jnp.sin(thetas)
        return jnp.stack([x, y], axis=1)

    centers = lax.cond(count < total_zones, fallback_centers, lambda x: x, centers)
    return centers, colors


def _sample_agent_position(
    key: jax.Array, params: EnvParams, centers: jnp.ndarray
) -> jnp.ndarray:
    minval = -params.spawn_size / 2 + params.keepout_radius
    maxval = params.spawn_size / 2 - params.keepout_radius

    def agent_cond(carry):
        key, pos = carry
        dists = jnp.linalg.norm(centers - pos, axis=1)
        return jnp.any(dists < params.keepout_radius * 2)

    def agent_body(carry):
        key, _pos = carry
        key, sub = jax.random.split(key)
        pos = jax.random.uniform(sub, (2,), minval=minval, maxval=maxval)
        return key, pos

    key_init, key = jax.random.split(key)
    init_pos = jax.random.uniform(key_init, (2,), minval=minval, maxval=maxval)
    key, pos = lax.while_loop(agent_cond, agent_body, (key, init_pos))
    return pos


def _compute_lidar(state: EnvState, params: EnvParams) -> jnp.ndarray:
    """Compute per-color lidar distances with evenly spaced bins around the agent.

    Returns an array of shape (C, num_bins) with distances in world units.
    """
    num_bins = params.num_lidar_bins
    max_range = (
        params.world_size if params.lidar_max_range is None else params.lidar_max_range
    )
    pos = state.position  # (2,)
    angles = jnp.arange(num_bins, dtype=jnp.float32) * (2.0 * jnp.pi / num_bins)
    dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (B,2)

    centers = state.zone_centers  # (N,2)
    colors = state.zone_colors  # (N,)
    radius_sq = params.zone_radius * params.zone_radius

    def ray_zone_intersect(direction: jnp.ndarray, center: jnp.ndarray) -> jnp.ndarray:
        """Compute intersection distance for a single ray and zone.

        Args:
            direction: Ray direction vector (2,)
            center: Zone center (2,)

        Returns:
            Distance to intersection, or inf if no intersection.
        """
        # Uses ray-sphere intersection, i.e. ||p + t*d - c||^2 = r^2
        # where p=pos, d=direction, c=center, r=radius

        m = pos - center  # (2,)
        b = jnp.dot(direction, m)
        c = jnp.dot(m, m) - radius_sq
        disc = b * b - c

        has_intersection = disc >= 0.0
        sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
        t_min = -b - sqrt_disc
        t_max = -b + sqrt_disc

        in_sphere = jnp.logical_and(t_min <= 0.0, t_max >= 0.0)
        sphere_in_front = t_min >= 0.0
        return jnp.where(
            has_intersection,
            jnp.where(in_sphere, 0.0, jnp.where(sphere_in_front, t_min, jnp.inf)),
            jnp.inf,
        )

    # Vectorize over rays (axis 0) and zones (axis 1)
    ray_zone_distances = jax.vmap(
        jax.vmap(ray_zone_intersect, in_axes=(None, 0)),  # vmap over zones
        in_axes=(0, None),  # vmap over rays
    )(dirs, centers)  # (num_bins, num_zones)

    def compute_color_lidar(color_id: jnp.ndarray) -> jnp.ndarray:
        """Compute lidar for a single color.

        Args:
            color_id: Scalar color ID

        Returns:
            Distances for each bin (num_bins,)
        """
        # Mask distances for this color
        mask_color = colors == color_id  # (num_zones,)
        t_color = jnp.where(mask_color, ray_zone_distances, jnp.inf)
        # Find minimum distance per bin
        dmin = jnp.min(t_color, axis=1)  # (num_bins,)
        # Clamp to max_range
        dmin = jnp.where(jnp.isfinite(dmin), dmin, max_range)
        dmin = jnp.clip(dmin, 0.0, max_range)
        return dmin.astype(jnp.float32)

    color_ids = jnp.arange(len(params.colors), dtype=jnp.int32)
    lidar = jax.vmap(compute_color_lidar)(color_ids)  # (C, num_bins)
    return lidar


@partial(jax.jit, static_argnames="params")
def step(
    state: EnvState,
    action: jnp.ndarray,
    params: EnvParams | None = None,
) -> EnvTransition:
    """Advance the environment by one time-step.

    Args:
        state: The current :class:`EnvState`.
        action: Array ``[force, angular_velocity]`` controlling the agent.
        params: Optional override for the environment parameters.

    Returns:
        An :class:`EnvTransition` consisting of the next state, observation,
        scalar reward, termination flag, and diagnostic metrics.
    """

    params = params or _DEFAULT_PARAMS

    force = jnp.clip(action[0], -params.max_force, params.max_force)
    target_angular_velocity = jnp.clip(
        action[1], -params.max_angular_velocity, params.max_angular_velocity
    )

    heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])
    acceleration = heading * force

    velocity = state.velocity + acceleration * params.dt
    velocity *= 1.0 - params.drag

    speed = jnp.linalg.norm(velocity)
    speed_scale = jnp.minimum(1.0, params.max_speed / (speed + _EPS))
    velocity = velocity * speed_scale

    position = state.position + velocity * params.dt

    # If the agent is out of bounds, reflect its velocity and clamp its position
    # to the edge of the world.
    half_size = params.world_size / 2.0
    velocity = jnp.where(jnp.abs(position) > half_size, -velocity, velocity)
    position = jnp.clip(position, -half_size, half_size)

    angle = _wrap_angle(state.angle + target_angular_velocity * params.dt)
    angular_velocity = target_angular_velocity

    done = state.num_steps >= params.max_steps - 1

    reward = jnp.zeros((), dtype=jnp.float32)

    next_state = EnvState(
        position=position,
        velocity=velocity,
        angle=angle,
        angular_velocity=angular_velocity,
        acceleration=acceleration,
        num_steps=state.num_steps + 1,
        zone_centers=state.zone_centers,
        zone_colors=state.zone_colors,
    )
    lidar = _compute_lidar(next_state, params)
    observation = EnvObservation(
        acceleration=acceleration,
        velocity=velocity,
        angular_velocity=angular_velocity,
        lidar=lidar,
    )
    return EnvTransition(
        state=next_state,
        observation=observation,
        reward=reward,
        done=done,
    )
