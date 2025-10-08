"""A 2D renderer for the zones-jax environment based on pygame."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import pygame
from pygame import gfxdraw

from zones_jax import environment


class Renderer:
    """A 2D renderer for the zones-jax environment."""

    def __init__(
        self,
        params: environment.EnvParams,
        screen_size: int = 800,
    ):
        """Initialize the renderer.

        Args:
            params: Environment parameters.
            screen_size: The size of the square screen in pixels.
        """
        pygame.init()
        pygame.display.set_caption("zones-jax")

        self._params = params
        self._screen_size = screen_size
        self._screen = pygame.display.set_mode((screen_size, screen_size))
        self._font = pygame.font.Font(None, 36)

        self._world_to_screen_scale = screen_size / params.world_size
        self._agent_radius_px = int(params.agent_radius * self._world_to_screen_scale)
        self._zone_radius_px = int(params.zone_radius * self._world_to_screen_scale)

        # --- Visual enhancements ---
        # Checkerboard background
        self._grid_size_px = 50
        self._grid_color_1 = (248, 250, 252)  # slate-50
        self._grid_color_2 = (241, 245, 249)  # slate-100

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

    def _world_to_screen(self, pos: jnp.ndarray) -> tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        pos = (pos + self._params.world_size / 2) * self._world_to_screen_scale
        return int(pos[0]), self._screen_size - int(pos[1])

    def render(
        self,
        state: environment.EnvState,
        previous_state: environment.EnvState,
        alpha: float,
    ):
        """Render the environment state."""
        # Draw checkerboard background
        self._screen.fill(self._grid_color_1)
        for y in range(0, self._screen_size, self._grid_size_px):
            for x in range(0, self._screen_size, self._grid_size_px):
                if (y // self._grid_size_px + x // self._grid_size_px) % 2 == 1:
                    rect = pygame.Rect(x, y, self._grid_size_px, self._grid_size_px)
                    self._screen.fill(self._grid_color_2, rect)

        # Interpolate position for smooth rendering
        interpolated_position = (
            previous_state.position * (1.0 - alpha) + state.position * alpha
        )

        # Interpolate angle for smooth rendering, handling wrapping
        angle_diff = (state.angle - previous_state.angle + jnp.pi) % (
            2 * jnp.pi
        ) - jnp.pi
        interpolated_angle = previous_state.angle + alpha * angle_diff

        # Draw zones
        if state.zone_centers is not None and state.zone_colors is not None:
            for i in range(state.zone_centers.shape[0]):
                center = state.zone_centers[i]
                color_id = int(state.zone_colors[i])
                col = self._zone_colors.get(color_id, (0, 0, 0))
                zpos = self._world_to_screen(center)
                gfxdraw.aacircle(self._screen, *zpos, self._zone_radius_px, col)
                gfxdraw.filled_circle(self._screen, *zpos, self._zone_radius_px, col)

        # Draw agent
        agent_pos_screen = self._world_to_screen(interpolated_position)
        gfxdraw.aacircle(
            self._screen, *agent_pos_screen, self._agent_radius_px, self._agent_color
        )
        gfxdraw.filled_circle(
            self._screen, *agent_pos_screen, self._agent_radius_px, self._agent_color
        )

        # Draw agent heading as a rectangle
        angle = interpolated_angle
        cos_angle = jnp.cos(angle)
        sin_angle = jnp.sin(angle)
        rect_w = 0.02
        rect_l = 0.2

        # Define corners relative to agent center
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
        translated_corners = rotated_corners + interpolated_position

        # Convert to screen coordinates
        screen_corners = [self._world_to_screen(c) for c in translated_corners]

        gfxdraw.filled_polygon(self._screen, screen_corners, self._agent_heading_color)
        gfxdraw.aapolygon(self._screen, screen_corners, self._agent_heading_color)

        pygame.display.flip()

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

    def close(self):
        """Close the renderer."""
        pygame.quit()


def run_manual_control():
    """Run the environment with manual control."""
    params = environment.default_params()
    renderer = Renderer(params)

    key = jax.random.PRNGKey(0)
    state, _ = environment.reset(key, params)
    previous_state = state

    clock = pygame.time.Clock()
    time_accumulator = 0.0
    time_scale = 1.0  # Speed up the simulation

    while True:
        # Get elapsed time in seconds and add to accumulator
        time_accumulator += (clock.tick(120) / 1000.0) * time_scale
        print(f"FPS: {clock.get_fps():.2f}")

        # Get user action once per frame
        action = renderer.get_action()

        # Run physics steps to catch up with accumulated time
        while time_accumulator >= params.dt:
            previous_state = state
            transition = environment.step(state, action, params)
            state = transition.state

            if transition.done:
                key, reset_key = jax.random.split(key)
                print(state.num_steps)
                state, _ = environment.reset(reset_key, params)
                previous_state = state
                # If we reset, we can break the inner loop to render the new state
                break

            time_accumulator -= params.dt

        # Calculate interpolation factor
        alpha = time_accumulator / params.dt
        renderer.render(state, previous_state, alpha)


if __name__ == "__main__":
    run_manual_control()
