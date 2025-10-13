"""An implementation of the zone environment introduced by LTL2Action (Vaezipoor et al., 2021).

The environment simulates a point-mass agent moving in a 2D plane. The agent
applies a forward force aligned with its current heading and can control its
angular velocity. The world contains colored zones that the agent can enter.
The agent is equipped with a lidar sensor that detects the distance to the
nearest zone of each color in a set of evenly spaced angular bins.
"""

from typing import Any, NamedTuple, override

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from jaxltl.environments import environment, spaces

_EPS = 1e-8
_MAX_ZONE_PLACEMENT_ITERS = 1000


class EnvParams(environment.EnvParams):
    # World
    agent_radius: jax.Array  # float
    world_size: jax.Array  # float
    spawn_size: jax.Array  # float
    # Zones
    zone_radius: jax.Array  # float
    zones_per_color: int
    keepout_radius: jax.Array  # float
    # Lidar
    num_lidar_bins: int
    # Physics
    dt: jax.Array  # float
    drag: jax.Array  # float
    max_speed: jax.Array  # float
    max_force: jax.Array  # float
    max_angular_velocity: jax.Array  # float


class EnvState(eqx.Module):
    # Physics
    position: jax.Array  # shape: (2,)
    velocity: jax.Array  # shape: (2,)
    angle: jax.Array  # shape: ()
    angular_velocity: jax.Array  # float
    acceleration: jax.Array  # shape: (2,)
    # Zones (static for an episode)
    zone_centers: jax.Array  # shape: (N, 2)
    zone_colors: jax.Array  # shape: (N,) int in [0, C)


class ObsFeatures(NamedTuple):
    acceleration: jax.Array  # shape: (2,)
    velocity: jax.Array  # shape: (2,)
    angular_velocity: jax.Array  # shape: (1,)
    lidar: jax.Array  # shape: (C, num_bins)


class ZoneEnv(environment.Environment[EnvState, EnvParams, ObsFeatures]):
    default_params = EnvParams(
        max_steps_in_episode=jnp.int32(1000),
        agent_radius=jnp.float32(0.1),
        world_size=jnp.float32(6.6),
        spawn_size=jnp.float32(5.0),
        zone_radius=jnp.float32(0.4),
        zones_per_color=2,
        keepout_radius=jnp.float32(0.55),
        num_lidar_bins=16,
        dt=jnp.float32(0.05),
        drag=jnp.float32(0.08),
        max_speed=jnp.float32(3.0),
        max_force=jnp.float32(2.0),
        max_angular_velocity=jnp.float32(3.0),
    )
    propositions = ("red", "green", "purple", "yellow")
    reset_to_initial_state = True  # the reset function is expensive, so we avoid it

    def __init__(self):
        super().__init__(
            default_params=self.default_params,
            propositions=self.propositions,
            reset_to_initial_state=self.reset_to_initial_state,
        )

    @override
    def _observation_space(self, params: EnvParams) -> spaces.Space:
        return spaces.Box(
            low=0.0,
            high=jnp.inf,
            shape=(5 + len(self.propositions) * int(params.num_lidar_bins),),
            dtype=jnp.float32,
        )

    @override
    def _action_space(self, params: EnvParams) -> spaces.Space:
        return spaces.Box(
            low=jnp.array(
                [-params.max_force, -params.max_angular_velocity], dtype=jnp.float32
            ),
            high=jnp.array(
                [params.max_force, params.max_angular_velocity], dtype=jnp.float32
            ),
            shape=(2,),
            dtype=jnp.float32,
        )

    @override
    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[EnvState, NamedTuple]:
        key_zones, key_pos, key_angle = jax.random.split(key, 3)
        centers, colors = self._sample_zones(key_zones, params)
        position = self._sample_agent_position(key_pos, params, centers)
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
        obs = self._compute_obs(state, params)
        return state, obs

    def _sample_zones(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, jax.Array]:
        """Sample non-overlapping zone centers and assign colors.

        Returns (centers:(Z,2), colors:(Z,))
        """
        num_colors = len(self.propositions)
        total_zones = num_colors * params.zones_per_color
        minval = -params.spawn_size / 2 + params.keepout_radius
        maxval = params.spawn_size / 2 - params.keepout_radius
        keepout = params.keepout_radius

        centers0 = jnp.zeros((total_zones, 2), dtype=jnp.float32)
        colors = jnp.repeat(
            jnp.arange(num_colors, dtype=jnp.int32), params.zones_per_color
        )

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
        self, key: jax.Array, params: EnvParams, centers: jax.Array
    ) -> jax.Array:
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

    def _compute_obs(self, state: EnvState, params: EnvParams) -> ObsFeatures:
        """Compute the observation for a given state."""
        lidar = self._compute_lidar(state, params)
        return ObsFeatures(
            acceleration=state.acceleration,
            velocity=state.velocity,
            angular_velocity=state.angular_velocity.reshape(1),
            lidar=lidar,
        )

    def _compute_lidar(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Compute per-color lidar distances with evenly spaced bins around the agent.

        Returns an array of shape (C, num_bins) with distances in world units.
        """
        max_range = params.world_size
        pos = state.position  # (2,)
        bin_size = 2.0 * jnp.pi / params.num_lidar_bins
        heading = jnp.array([jnp.cos(state.angle), jnp.sin(state.angle)])  # (2,)

        centers = state.zone_centers  # (N,2)
        colors = state.zone_colors  # (N,)

        def zone_sensor_binned(zone_pos: jax.Array) -> jax.Array:
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

        def compute_color_lidar(color_id: jax.Array) -> jax.Array:
            """Compute lidar for a single color."""
            mask_color = colors == color_id  # (num_zones,)
            sensors_color = jnp.where(
                mask_color[:, None], sensors, 0.0
            )  # (N, num_bins)
            sensors_color = jnp.max(sensors_color, axis=0)  # (num_bins,)
            return sensors_color

        color_ids = jnp.arange(len(self.propositions), dtype=jnp.int32)
        lidar = jax.vmap(compute_color_lidar)(color_ids)  # (C, num_bins)
        return lidar

    @override
    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[EnvState, NamedTuple, jax.Array, jax.Array, dict[Any, Any]]:
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
        half_size = params.world_size / 2.0 - params.agent_radius / 2.0
        velocity = jnp.where(jnp.abs(position) > half_size, -velocity, velocity)
        position = jnp.clip(position, -half_size, half_size)

        angle = self._wrap_angle(state.angle + target_angular_velocity * params.dt)
        angular_velocity = target_angular_velocity

        reward = jnp.zeros((), dtype=jnp.float32)
        terminated = jnp.zeros((), dtype=jnp.bool)
        next_state = EnvState(
            position=position,
            velocity=velocity,
            angle=angle,
            angular_velocity=angular_velocity,
            acceleration=acceleration,
            zone_centers=state.zone_centers,
            zone_colors=state.zone_colors,
        )
        next_obs = self._compute_obs(next_state, params)
        return next_state, next_obs, reward, terminated, {}

    @staticmethod
    def _wrap_angle(angle: jax.Array) -> jax.Array:
        """Wrap angles to the (-pi, pi] interval."""
        return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    @override
    def compute_propositions(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Compute which zones the agent is currently inside.

        Returns a boolean array of shape (C,) indicating for each color whether
        the agent is inside any zone of that color.
        """
        pos = state.position  # (2,)
        centers = state.zone_centers  # (N,2)
        colors = state.zone_colors  # (N,)

        dists = jnp.linalg.norm(centers - pos, axis=1)  # (N,)
        inside = dists < params.zone_radius + params.agent_radius  # (N,)

        def compute_color_prop(color_id: jax.Array) -> jax.Array:
            mask_color = colors == color_id  # (N,)
            inside_color = jnp.logical_and(mask_color, inside)  # (N,)
            return jnp.any(inside_color)

        color_ids = jnp.arange(len(self.propositions), dtype=jnp.int32)
        propositions = jax.vmap(compute_color_prop)(color_ids)  # (C,)
        return propositions

    @override
    def unflatten_obs(self, obs: jax.Array) -> ObsFeatures:
        num_props = len(self.propositions)
        acceleration = obs[0:2]
        velocity = obs[2:4]
        angular_velocity = obs[4:5]
        lidar = obs[5:].reshape((num_props, -1))
        return ObsFeatures(
            acceleration=acceleration,
            velocity=velocity,
            angular_velocity=angular_velocity,
            lidar=lidar,
        )
