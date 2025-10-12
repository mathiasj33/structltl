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
_MAX_ZONE_PLACEMENT_ITERS = 1000


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
    colors: tuple[str, str, str, str] = ("red", "green", "purple", "yellow")
    zones_per_color: int = 2
    keepout_radius: float = 0.55
    # Lidar
    num_lidar_bins: int = 16


class EnvState(NamedTuple):
    key: jax.Array  # PRNG key for any stochasticity
    state: InternalState
    num_steps: jnp.ndarray = jnp.zeros((), dtype=jnp.int32)  # shape: ()
    info: dict = {}


class InternalState(NamedTuple):
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


class EnvObservation(NamedTuple):
    """Observation emitted after reset or step."""

    acceleration: jnp.ndarray  # shape: (2,)
    velocity: jnp.ndarray  # shape: (2,)
    angular_velocity: jnp.ndarray  # shape: ()
    lidar: jnp.ndarray  # Lidar per color: (C, num_lidar_bins)
    propositions: jnp.ndarray  # shape: (C,) boolean


class EnvTransition(NamedTuple):
    """Bundle returned by a call to :func:`step`."""

    state: EnvState
    observation: EnvObservation
    reward: jnp.ndarray  # shape: ()
    terminated: jnp.ndarray  # shape: () boolean
    truncated: jnp.ndarray  # shape: ()
    terminal_observation: EnvObservation | None = None  # used if done


def default_params(**overrides) -> EnvParams:
    """Return a default set of environment parameters."""
    default = {
        "dt": 0.05,
        "world_size": 6.6,
        "spawn_size": 5.0,
        "agent_radius": 0.1,
        "drag": 0.08,
        "max_speed": 3.0,
        "max_force": 2.0,
        "max_angular_velocity": 3.0,
        "zone_radius": 0.4,
        "zones_per_color": 2,
        "num_lidar_bins": 16,
    }
    return EnvParams(**(default | overrides))


_DEFAULT_PARAMS = default_params()


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
    key, key_zones, key_pos, key_angle = jax.random.split(key, 4)

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

    internal_state = InternalState(
        position=position,
        velocity=velocity,
        angle=angle,
        angular_velocity=angular_velocity,
        acceleration=acceleration,
        zone_centers=centers,
        zone_colors=colors,
    )
    observation = compute_observation(internal_state, params)
    state = EnvState(
        key=key,
        state=internal_state,
        num_steps=jnp.zeros((), dtype=jnp.int32),
        info={"initial_observation": observation, "initial_state": internal_state},
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

    key, centers, count, it = lax.while_loop(
        cond_fun, body_fun, (key, centers0, jnp.int32(0), jnp.int32(0))
    )

    fallback_centers = jnp.array(  # random but fixed fallback
        [
            [-1.60, -0.54],
            [0.82, 0.10],
            [-0.18, -1.12],
            [-1.91, 1.68],
            [1.58, 1.23],
            [1.82, -1.71],
            [-0.70, 0.58],
            [-0.49, 1.83],
        ]
    )

    centers = jnp.where(count < total_zones, fallback_centers, centers)
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


def _compute_lidar(state: InternalState, params: EnvParams) -> jnp.ndarray:
    """Compute per-color lidar distances with evenly spaced bins around the agent.

    Returns an array of shape (C, num_bins) with distances in world units.
    """
    max_range = params.world_size
    pos = state.position  # (2,)
    bin_size = 2.0 * jnp.pi / params.num_lidar_bins
    heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])  # (2,)

    centers = state.zone_centers  # (N,2)
    colors = state.zone_colors  # (N,)

    def zone_sensor_binned(zone_pos: jnp.ndarray) -> jnp.ndarray:
        """Compute the sensor of a single zone.

        Returns: (num_bins,)"""

        direction = zone_pos - pos  # (2,)
        dist: float = jnp.linalg.norm(direction)  # ()
        sensor = jnp.clip(1.0 - dist / max_range, 0.0, 1.0)  # ()
        direction = direction / (dist + _EPS)  # (2,)
        dotp = jnp.dot(heading, direction)
        cross = jnp.cross(heading, direction)
        angle = jnp.arctan2(cross, dotp) % (2.0 * jnp.pi)
        bin_idx = jnp.floor(angle / bin_size).astype(jnp.int32)
        bin_angle = bin_size * bin_idx
        bins = jnp.zeros((params.num_lidar_bins,), dtype=jnp.float32)
        alias = (angle - bin_angle) / bin_size
        bins = bins.at[bin_idx].set(sensor)
        bins = bins.at[bin_idx + 1].set(sensor * alias)
        bins = bins.at[bin_idx - 1].set(sensor * (1.0 - alias))
        return bins

    sensors = jax.vmap(zone_sensor_binned, in_axes=0)(centers)  # (N, num_bins)

    def compute_color_lidar(color_id: jnp.ndarray) -> jnp.ndarray:
        """Compute lidar for a single color."""
        mask_color = colors == color_id  # (num_zones,)
        sensors_color = jnp.where(mask_color[:, None], sensors, 0.0)  # (N, num_bins)
        sensors_color = jnp.max(sensors_color, axis=0)  # (num_bins,)
        return sensors_color

    color_ids = jnp.arange(len(params.colors), dtype=jnp.int32)
    lidar = jax.vmap(compute_color_lidar)(color_ids)  # (C, num_bins)
    return lidar


def _compute_propositions(state: InternalState, params: EnvParams) -> jnp.ndarray:
    """Compute which zones the agent is currently inside.

    Returns a boolean array of shape (C,) indicating for each color whether
    the agent is inside any zone of that color.
    """
    pos = state.position  # (2,)
    centers = state.zone_centers  # (N,2)
    colors = state.zone_colors  # (N,)

    dists = jnp.linalg.norm(centers - pos, axis=1)  # (N,)
    inside = dists < params.zone_radius + params.agent_radius  # (N,)

    def compute_color_prop(color_id: jnp.ndarray) -> jnp.ndarray:
        mask_color = colors == color_id  # (N,)
        inside_color = jnp.logical_and(mask_color, inside)  # (N,)
        return jnp.any(inside_color)

    color_ids = jnp.arange(len(params.colors), dtype=jnp.int32)
    propositions = jax.vmap(compute_color_prop)(color_ids)  # (C,)
    return propositions


def sample_action(state: EnvState) -> tuple[EnvState, jnp.ndarray]:
    """Sample a random action for the current state."""
    key, force_key, vel_key = jax.random.split(state.key, 3)
    force = jax.random.uniform(force_key, (), minval=-1.0, maxval=1.0)
    angular_velocity = jax.random.uniform(vel_key, (), minval=-1.0, maxval=1.0)
    return state._replace(key=key), jnp.array(
        [force, angular_velocity], dtype=jnp.float32
    )


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

    internal_state = state.state

    heading = jnp.array([jnp.cos(internal_state.angle), jnp.sin(internal_state.angle)])
    acceleration = heading * force

    velocity = internal_state.velocity + acceleration * params.dt
    velocity *= 1.0 - params.drag

    speed = jnp.linalg.norm(velocity)
    speed_scale = jnp.minimum(1.0, params.max_speed / (speed + _EPS))
    velocity = velocity * speed_scale

    position = internal_state.position + velocity * params.dt

    # If the agent is out of bounds, reflect its velocity and clamp its position
    # to the edge of the world.
    half_size = params.world_size / 2.0 - params.agent_radius / 2.0
    velocity = jnp.where(jnp.abs(position) > half_size, -velocity, velocity)
    position = jnp.clip(position, -half_size, half_size)

    angle = _wrap_angle(internal_state.angle + target_angular_velocity * params.dt)
    angular_velocity = target_angular_velocity

    truncated = state.num_steps + 1 >= params.max_steps

    reward = jnp.zeros((), dtype=jnp.float32)

    next_internal_state = InternalState(
        position=position,
        velocity=velocity,
        angle=angle,
        angular_velocity=angular_velocity,
        acceleration=acceleration,
        zone_centers=internal_state.zone_centers,
        zone_colors=internal_state.zone_colors,
    )
    next_observation = compute_observation(next_internal_state, params)

    next_state = EnvState(
        key=state.key,
        state=next_internal_state,
        info=state.info,
        num_steps=state.num_steps + 1,
    )

    initial_state = EnvState(
        key=state.key,
        state=state.info["initial_state"],
        info=state.info,
        num_steps=jnp.zeros((), dtype=jnp.int32),
    )

    return EnvTransition(
        state=lax.cond(truncated, lambda: initial_state, lambda: next_state),
        observation=lax.cond(
            truncated,
            lambda: state.info["initial_observation"],
            lambda: next_observation,
        ),
        reward=reward,
        truncated=truncated,
        terminated=jnp.zeros((), dtype=jnp.bool),
        terminal_observation=None,
    )


def compute_observation(state: InternalState, params: EnvParams) -> EnvObservation:
    """Compute the observation for a given state."""
    lidar = _compute_lidar(state, params)
    propositions = _compute_propositions(state, params)
    observation = EnvObservation(
        acceleration=state.acceleration,
        velocity=state.velocity,
        angular_velocity=state.angular_velocity,
        lidar=lidar,
        propositions=propositions,
    )
    return observation


# TODO: save initial observation in state for vectorised reset. Implement resetting envs every x steps. Check performance.
# TODO: update sample agent position to have a max iters and fallback to a fixed position if exceeded.
