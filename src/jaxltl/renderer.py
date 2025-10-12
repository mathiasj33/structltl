"""A 2D renderer for the zones-jax environment based on pygame."""

from __future__ import annotations

import math
import sys
from functools import partial

import jax
import jax.numpy as jnp
import pygame
from pygame import gfxdraw

from jaxltl import environment


class Renderer:
    def __init__(
        self,
        params: environment.EnvParams,
        screen_size: int = 800,
        grid_size: int = 50,
        show_lidar: bool = False,
    ):
        pygame.init()
        pygame.display.set_caption("zones-jax")

        self._params = params
        self._screen_size = screen_size
        self.draw_lidar = show_lidar

        self._screen = pygame.display.set_mode((screen_size, screen_size))
        self._background = pygame.Surface(self._screen.get_size())
        self._lidar_surface = pygame.Surface(self._screen.get_size(), pygame.SRCALPHA)

        self._world_to_screen_scale = screen_size / params.world_size
        self._agent_radius_px = int(params.agent_radius * self._world_to_screen_scale)
        self._zone_radius_px = int(params.zone_radius * self._world_to_screen_scale)

        # Checkerboard background
        self.grid_size = grid_size
        self._grid_color_1 = (248, 250, 252)
        self._grid_color_2 = (241, 245, 249)
        self._render_background()

        # Agent color
        self._agent_color = (59, 130, 246)  # blue-500
        self._agent_heading_color = (59, 130, 246, 180)  # blue-500 with alpha

        # Color mapping for zones
        self._zone_colors = {
            0: (239, 68, 68, 180),  # red-500
            1: (34, 197, 94, 180),  # green-500
            2: (168, 85, 247, 180),  # purple-500
            3: (234, 179, 8, 180),  # yellow-500
        }

    def _render_background(self):
        """Draw checkerboard background."""
        self._background.fill(self._grid_color_1)
        for y in range(0, self._screen_size, self.grid_size):
            for x in range(0, self._screen_size, self.grid_size):
                if (y // self.grid_size + x // self.grid_size) % 2 == 1:
                    rect = pygame.Rect(x, y, self.grid_size, self.grid_size)
                    self._background.fill(self._grid_color_2, rect)

    def _render_zones(self, state: environment.EnvState):
        centers = self._world_to_screen(state.zone_centers).tolist()
        for i, center in enumerate(centers):
            color_id = int(state.zone_colors[i])
            col = self._zone_colors.get(color_id, (0, 0, 0))
            self._draw_circle(self._screen, col, center, self._zone_radius_px)

    def _draw_circle(self, surface, color, position, radius):
        """Draw an anti-aliased filled circle."""
        gfxdraw.aacircle(surface, position[0], position[1], radius, color)
        gfxdraw.filled_circle(surface, position[0], position[1], radius, color)

    @partial(jax.jit, static_argnames=("self",))
    def _world_to_screen(self, pos: jnp.ndarray) -> jnp.ndarray:
        """Convert world coordinates to screen coordinates."""
        pos = (pos + self._params.world_size / 2) * self._world_to_screen_scale
        pos = pos.at[:, 1].set(self._screen_size - pos[:, 1])
        return pos.astype(jnp.int32)

    def render(
        self,
        state: environment.EnvState,
        previous_state: environment.EnvState,
        obs: environment.EnvObservation,
        alpha: float,
    ):
        """Render the environment state."""
        self._screen.blit(self._background, (0, 0))
        self._render_zones(state)

        # Interpolation for smooth rendering
        interpolated_position = (
            previous_state.position * (1.0 - alpha) + state.position * alpha
        )
        angle_diff = (state.angle - previous_state.angle + jnp.pi) % (
            2 * jnp.pi
        ) - jnp.pi
        interpolated_angle = previous_state.angle + alpha * angle_diff

        self._draw_agent(interpolated_position, interpolated_angle)
        if self.draw_lidar:
            self._draw_lidar(interpolated_position, obs, state)

        pygame.display.flip()

    def _draw_agent(self, position: jnp.ndarray, angle: jnp.ndarray):
        # Draw agent heading as a rectangle
        cos_angle = jnp.cos(angle)
        sin_angle = jnp.sin(angle)
        rect_w = 0.02
        rect_l = 0.2
        corners = jnp.array(
            [
                [0, -rect_w / 2],
                [rect_l, -rect_w / 2],
                [rect_l, rect_w / 2],
                [0, rect_w / 2],
            ]
        )

        # Rotate and translate corners
        rotation_matrix = jnp.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        rotated_corners = jnp.dot(corners, rotation_matrix.T)
        translated_corners = rotated_corners + position

        agent_and_corners = jnp.vstack([position, translated_corners])
        screen_positions = self._world_to_screen(agent_and_corners).tolist()
        agent_pos = screen_positions[0]
        self._draw_circle(
            self._screen, self._agent_color, agent_pos, self._agent_radius_px
        )
        corners = screen_positions[1:]
        gfxdraw.filled_polygon(self._screen, corners, self._agent_heading_color)
        gfxdraw.aapolygon(self._screen, corners, self._agent_heading_color)

    def _draw_lidar(
        self,
        position: jnp.ndarray,
        obs: environment.EnvObservation,
        state: environment.EnvState,
    ):
        self._lidar_surface.fill((0, 0, 0, 0))
        points = self._compute_lidar_points(position, obs, state)  # (C, num_bins, 3)
        screen_pos = self._world_to_screen(points.reshape(-1, 3)[:, :2]).reshape(
            points.shape[0], points.shape[1], 2
        )
        points = points.at[:, :, :2].set(screen_pos).tolist()
        for color, lidar_points in zip(self._zone_colors.values(), points, strict=True):
            for point in lidar_points:
                pos = point[:2]
                strength = point[2]
                if strength > 0.0:
                    exp_strength = math.exp(-2 * (1 - strength))
                    faded_color = (
                        color[0],
                        color[1],
                        color[2],
                        int(exp_strength * 255),
                    )
                    pygame.draw.circle(self._lidar_surface, faded_color, pos, 2)
        self._screen.blit(self._lidar_surface, (0, 0))

    @staticmethod
    @jax.jit
    def _compute_lidar_points(
        position: jnp.ndarray,
        obs: environment.EnvObservation,
        state: environment.EnvState,
    ) -> jnp.ndarray:
        # obs.lidar shape (C, num_bins)
        num_bins = obs.lidar.shape[1]
        bin_size = 2 * jnp.pi / num_bins
        bin_idx = state.angle // bin_size
        normalized_angle = bin_idx * bin_size
        points = jnp.zeros((*obs.lidar.shape, 3))
        angles = (jnp.arange(num_bins) / num_bins) * 2 * jnp.pi + normalized_angle
        angles = angles % (2 * jnp.pi)
        directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

        for i, signals in enumerate(obs.lidar):
            positions = position + directions * (0.2 + i * 0.1)
            points = points.at[i].set(
                jnp.concatenate([positions, signals[:, None]], axis=-1)
            )
        return points

    def get_action(self) -> jnp.ndarray:
        """Get action from keyboard input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            pygame.quit()
            sys.exit()

        force = 0.0
        angular_velocity = 0.0

        if keys[pygame.K_w]:
            force = self._params.max_force
        if keys[pygame.K_s]:
            force = -self._params.max_force
        if keys[pygame.K_a]:
            angular_velocity = self._params.max_angular_velocity
        if keys[pygame.K_d]:
            angular_velocity = -self._params.max_angular_velocity

        return jnp.array([force, angular_velocity])

    def show_fps(self, clock):
        """Display the current FPS on the window title."""
        fps = clock.get_fps()
        pygame.display.set_caption(f"zones-jax - FPS: {fps:.2f}")

    def close(self):
        """Close the renderer."""
        pygame.quit()


def run_manual_control():
    """Run the environment with manual control."""
    params = environment.default_params()
    renderer = Renderer(params, show_lidar=False)

    key = jax.random.PRNGKey(0)
    state, obs = environment.reset(key, params)
    previous_state = state

    clock = pygame.time.Clock()
    time_accumulator = 0.0
    time_scale = 1.0  # Speed up the simulation

    while True:
        # Get elapsed time in seconds and add to accumulator
        time_accumulator += (clock.tick(120) / 1000.0) * time_scale
        renderer.show_fps(clock)

        # Get user action once per frame
        action = renderer.get_action()

        # Run physics steps to catch up with accumulated time
        while time_accumulator >= params.dt:
            previous_state = state
            transition = environment.step(state, action, params)
            state = transition.state
            obs = transition.observation

            if transition.truncated or transition.terminated:
                key, reset_key = jax.random.split(key)
                print(state.num_steps)
                state, obs = environment.reset(reset_key, params)
                previous_state = state
                # If we reset, we can break the inner loop to render the new state
                break

            time_accumulator -= params.dt

        # Calculate interpolation factor
        alpha = time_accumulator / params.dt
        renderer.render(state.state, previous_state.state, obs, alpha)
        props = {params.colors[i] for i, p in enumerate(obs.propositions.tolist()) if p}
        print(props)


if __name__ == "__main__":
    run_manual_control()
