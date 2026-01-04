from math import prod
from typing import NamedTuple, cast, override

import distrax
import hydra
import jax
import jax.numpy as jnp
import jraph
from equinox import nn
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxltl.deep_ltl.reach_avoid.jax_graph_reach_avoid_sequence import (
    JaxGraphReachAvoidSequence,
    NodeData,
)
from jaxltl.environments import spaces
from jaxltl.environments.spaces import Space
from jaxltl.networks.gcn import GCN, NodeFeatures
from jaxltl.networks.gru_cell import GRUCell
from jaxltl.networks.mlp import MLP
from jaxltl.rl.actor.continuous_actor import ContinuousActor
from jaxltl.rl.actor_critic import ActorCritic

# TODO: refactor this to be a separate alg


class LTLGNNModel(ActorCritic):
    env_net: MLP
    prop_embedding: nn.Embedding
    type_embedding: nn.Embedding
    gcn: GCN
    gru: GRUCell
    actor: ContinuousActor
    critic: MLP

    _flatten_features: bool

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        act_space: Space,
        num_propositions: int,
        key: jax.Array,
        **kwargs,
    ):
        config = DictConfig(kwargs)
        key, env_key = jax.random.split(key)
        is_conv = "ConvNet" in config.env_net._target_
        params = {"obs_shape": obs_shape} if is_conv else {"in_size": prod(obs_shape)}
        self.env_net = hydra.utils.instantiate(config.env_net, **params, key=env_key)
        self._flatten_features = not is_conv

        key, prop_emb_key, type_emb_key = jax.random.split(key, 3)
        embedding_dim = config.sequence.embedding_dim
        self.prop_embedding = nn.Embedding(
            num_embeddings=num_propositions,
            embedding_size=embedding_dim,
            key=prop_emb_key,
        )
        # AND, OR, NOT, EMPTY, EPSILON
        num_node_types = 5
        self.type_embedding = nn.Embedding(
            num_embeddings=num_node_types,
            embedding_size=embedding_dim,
            key=type_emb_key,
        )

        key, gcn_key = jax.random.split(key)
        self.gcn = hydra.utils.instantiate(
            config.sequence.gcn,
            in_size=embedding_dim,
            out_size=embedding_dim,
            key=gcn_key,
        )

        key, gru_key = jax.random.split(key)
        # GRU takes concatenated reach and avoid root node embeddings
        self.gru = GRUCell(
            input_size=2 * embedding_dim,
            hidden_size=2 * embedding_dim,
            key=gru_key,
        )

        actor_key, critic_key = jax.random.split(key)
        joint_dim = config.env_net.out_size + 2 * embedding_dim
        params = self._get_actor_params_from_space(act_space)
        self.actor = hydra.utils.instantiate(
            config.actor, in_size=joint_dim, **params, key=actor_key
        )
        self.critic = hydra.utils.instantiate(
            config.critic,
            in_size=joint_dim,
            out_size=1,
            final_layer_activation=False,
            key=critic_key,
        )

    @staticmethod
    def _get_actor_params_from_space(act_space: Space) -> dict:
        """Returns a dict of parameters required to instantiate the actor based on the
        action space."""
        if isinstance(act_space, spaces.Discrete):
            return {"num_actions": act_space.n}
        elif isinstance(act_space, spaces.Box):
            return {"action_dim": act_space.shape[0]}
        elif isinstance(act_space, spaces.Composite):
            return {
                "continuous_action_dim": act_space.continuous.shape[0],
                "num_discrete_actions": act_space.discrete.n,
            }
        else:
            raise NotImplementedError(
                f"Actor parameters extraction not implemented for space type "
                f"{type(act_space)}"
            )

    @override
    def _get_action(
        self, features: jax.Array, epsilon_mask: jax.Array
    ) -> distrax.Distribution:
        return self.actor(features, epsilon_mask)

    @override
    def _get_value(self, features: jax.Array) -> jax.Array:
        value = jax.vmap(self.critic)(features)
        return value.squeeze(-1)

    @override
    def _compute_common_features(self, obs: PyTree) -> jax.Array:
        x = (
            self.flatten_features(obs.features)
            if self._flatten_features
            else obs.features.features
        )
        x = jax.vmap(self.env_net)(x)
        emb = jax.vmap(self._compute_sequence_embedding)(obs.seq)
        return jnp.concatenate([x, emb], axis=-1)

    def _compute_sequence_embedding(self, seq: JaxGraphReachAvoidSequence) -> jax.Array:
        # Sequence of graphs is treated as one big graph, no vmap required
        reach_root_features = self._get_root_features(seq.reach_graphs)
        avoid_root_features = self._get_root_features(seq.avoid_graphs)

        reach_avoid = jnp.concatenate(
            [reach_root_features, avoid_root_features], axis=-1
        )
        h0 = jnp.zeros((self.gru.hidden_size,))  # initial hidden state

        def gru_step(
            carry: tuple[jax.Array, int], inputs: jax.Array
        ) -> tuple[tuple[jax.Array, int], None]:
            hidden, step = carry
            hidden = jax.lax.cond(
                step <= seq.depth, lambda: self.gru(inputs, hidden), lambda: hidden
            )
            return (hidden, step - 1), None

        max_seq_length = reach_avoid.shape[0]
        (final_hidden, _), _ = jax.lax.scan(
            gru_step, (h0, max_seq_length), reach_avoid, reverse=True, unroll=8
        )
        return final_hidden

    def _get_root_features(self, graph: jraph.GraphsTuple) -> jax.Array:
        """Embeds graph nodes, runs GCN, and extracts root node features for a sequence."""
        # 1. Embed nodes
        nodes = cast(NodeData, graph.nodes)

        prop_idx = nodes["prop_idx"]
        type_idx = nodes["type_idx"]
        node_mask = nodes["mask"]

        # Embed propositions (for proposition nodes)
        is_prop = prop_idx != -1
        prop_emb = jax.vmap(self.prop_embedding)(prop_idx * is_prop) * is_prop[:, None]

        # Embed node types (for non-proposition nodes)
        is_type = type_idx != -1
        type_emb = jax.vmap(self.type_embedding)(type_idx * is_type) * is_type[:, None]

        # Combine embeddings and apply final padding mask
        node_features = (prop_emb + type_emb) * node_mask[:, None]

        # 2. Run GCN
        graph_with_features = graph._replace(
            nodes={"features": node_features, "mask": node_mask}
        )
        processed_graph = self.gcn(graph_with_features)

        processed_nodes = cast(NodeFeatures, processed_graph.nodes)

        output_node_features = processed_nodes["features"]

        # 3. Extract root node features
        # For ragged graphs, root indices are the cumulative sum of nodes in preceding graphs.
        root_indices = jnp.concatenate([jnp.array([0]), jnp.cumsum(graph.n_node[:-1])])
        root_features = output_node_features[root_indices]

        return root_features

    @staticmethod
    def flatten_features(features: NamedTuple) -> jax.Array:
        return jnp.concatenate(
            [v.reshape(v.shape[0], -1) for v in jax.tree.leaves(features)], axis=-1
        )
