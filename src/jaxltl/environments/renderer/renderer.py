import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import pygame

from jaxltl.environments.environment import Environment, EnvObservation, EnvParams
from jaxltl.environments.wrappers.wrapper import EnvWrapper


class BaseRenderer[TEnvState: eqx.Module, TObsFeatures: NamedTuple](ABC):
    """Base class for renderers."""

    def __init__(
        self,
        title: str,
        screen_size: int = 800,
    ):
        self.screen_size = screen_size
        self.title = title

        pygame.init()
        pygame.display.set_caption(title)
        self._screen = pygame.display.set_mode((screen_size, screen_size))

    @abstractmethod
    def render(
        self,
        state: TEnvState,
        previous_state: TEnvState,
        obs: TObsFeatures,
        alpha: float,
    ):
        """Renders the environment state. Use alpha for interpolation between frames."""
        pass

    def get_pressed_keys(self) -> pygame.key.ScancodeWrapper:
        """Gets the currently pressed keys. Exits if QUIT event is detected, or user pressed Q."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            pygame.quit()
            sys.exit()
        return keys

    @abstractmethod
    def get_action(self, keys: pygame.key.ScancodeWrapper) -> jax.Array:
        """Gets an action from user input."""
        pass

    def show_fps(self, clock):
        """Display the current FPS on the window title."""
        fps = clock.get_fps()
        pygame.display.set_caption(f"{self.title} - FPS: {fps:.2f}")

    def close(self):
        """Close the renderer."""
        pygame.quit()

    def run_render_loop(
        self,
        env: Environment | EnvWrapper,
        params: EnvParams,
        time_scale: float = 1.0,
        policy: Callable[[EnvObservation[TObsFeatures], jax.Array], jax.Array]
        | None = None,
        key: jax.Array | None = None,
    ):
        """Run the environment with manual control.

        Params:
            env: The environment to render.
            params: Environment parameters.
            time_scale: Speed multiplier for the simulation.
            policy: Optional policy function to generate actions from observations. If
                    None, user input is used to generate actions.
            key: JAX random key. If None, a default key is used.
        """

        if key is None:
            key = jax.random.key(0)
        key, key_reset = jax.random.split(key)
        state, obs = env.reset(key_reset, None, params)
        action = policy(obs, key) if policy else env.action_space(params).sample(key)  # type: ignore
        # Warm-up step, make sure everything is jitted
        env.step(key, state, action, params)  # type: ignore
        previous_state = state

        clock = pygame.time.Clock()
        time_accumulator = 0.0

        while True:
            # Get elapsed time in seconds and add to accumulator
            time_accumulator += (clock.tick(180) / 1000.0) * time_scale
            self.show_fps(clock)

            # Get user action once per frame
            pressed_keys = self.get_pressed_keys()
            key, key_action = jax.random.split(key)  # type: ignore
            action = (
                policy(obs, key_action) if policy else self.get_action(pressed_keys)
            )

            # Run physics steps to catch up with accumulated time
            while time_accumulator >= params.dt:
                previous_state = state
                key, key_step = jax.random.split(key)  # type: ignore
                transition = env.step(key_step, state, action, params)
                state = transition.state
                obs = transition.observation

                if transition.reward > 0:
                    print(f"Reward received: {transition.reward}")

                if transition.truncated or transition.terminated:
                    previous_state = state
                    # If we reset, we can break the inner loop to render the new state
                    break

                time_accumulator -= params.dt

            # Calculate interpolation factor
            alpha = float(time_accumulator / params.dt)
            self.render(
                env.unwrapped(state), env.unwrapped(previous_state), obs.features, alpha
            )
