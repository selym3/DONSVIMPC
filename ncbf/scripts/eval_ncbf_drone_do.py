import pathlib

import einops as ei
import ipdb
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.colors import CenteredNorm
from og.ckpt_utils import load_ckpt_ez
from og.cmap import get_BuRd
from og.jax_utils import jax2np, jax_vmap
from og.path_utils import mkdir
from og.schedules import Constant

from ncbf.drone_do_task import get_h_vector_drone, state_to_obs_drone
from ncbf.offline.train_offline_alg_drone_do import (
    TrainOfflineCfg,
    TrainOfflineDroneAlg,
)
import configparser as ConfigParser
from robot_planning.factory.factories import robot_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
import os


H_LABELS = ["obs", "boundary", "drone_angle", "px_left"]


def load_config_for_obstacles():
    config_path = (
        "../../robot_planning/scripts/configs/run_quadrotor2d_dynamic_obstacles.cfg"
    )
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)

    # Agent name should not matter, all use the same collision checker
    agent_name = "baseline"

    agent1 = factory_from_config(robot_factory_base, config_data, agent_name + "_agent")

    collision_checker = agent1.controller.cost_evaluator.collision_checker
    obstacles = collision_checker.obstacles
    obstacles_radius = collision_checker.obstacles_radius
    obstacles_velocity = collision_checker.obstacles_velocity
    obstacle_paths = collision_checker.obstacle_paths

    path_deltas = obstacle_paths[:, 1] - obstacle_paths[:, 0]
    path_norms = np.linalg.norm(path_deltas, axis=1, keepdims=True)
    assert np.all(path_norms > 1e-5), "path norm"
    path_norm_vecs = path_deltas / path_norms
    twod_obstacle_velocities = obstacles_velocity[:, None] * path_norm_vecs
    assert twod_obstacle_velocities.shape == (len(obstacles), 2)

    obstacle_info = np.concatenate(
        [obstacles, twod_obstacle_velocities, obstacles_radius.reshape(-1, 1)], axis=1
    )

    return obstacle_info


def compute_Vh_grid(alg: TrainOfflineDroneAlg, eval_obstacles: jnp.ndarray):
    # [ px py theta vx vy omega ]
    x0 = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])

    n_px = 560
    n_py = 120

    b_x = np.linspace(-4.0, 10.0, num=n_px)
    b_y = np.linspace(0.0, 1.25, num=n_py)
    bb_x, bb_y = jnp.meshgrid(b_x, b_y)

    bb_state = ei.repeat(x0, "nx -> b1 b2 nx", b1=n_py, b2=n_px)
    bb_state = jnp.array(bb_state)
    bb_state = bb_state.at[..., 0].set(bb_x)
    bb_state = bb_state.at[..., 1].set(bb_y)

    bb_pos = bb_state[..., :2]

    bb_obs = jax_vmap(state_to_obs_drone, in_axes=(0, None), rep=2)(
        bb_state, eval_obstacles
    )
    bb_obs_norm = (bb_obs - alg.obs_mean) / alg.obs_std
    bbh_Vh = alg.value_net.apply_with(bb_obs_norm, params=alg.ema)
    return bb_pos, bbh_Vh


def plot_Vh(bb_pos, bbh_Vh, eval_obstacles, fig_path: pathlib.Path):
    nh = bbh_Vh.shape[2]
    figsize = np.array([8.0, nh * 3.0])
    fig, axes = plt.subplots(nh, dpi=300, figsize=figsize)
    [ax.set_aspect("equal") for ax in axes]

    cmap = get_BuRd()

    for ii, ax in enumerate(axes):
        cm = ax.contourf(
            bb_pos[:, :, 0],
            bb_pos[:, :, 1],
            bbh_Vh[:, :, ii],
            levels=32,
            cmap=cmap,
            norm=CenteredNorm(),
        )
        fig.colorbar(cm, ax=ax)
        ax.set_title(H_LABELS[ii] if ii < len(H_LABELS) else f"h{ii}")

        print(eval_obstacles)
        for px, py, vx, vy, r in eval_obstacles:
            print(px, py)
            ax.add_patch(
                plt.Circle((float(px), float(py)), float(r), color="C3", alpha=0.5)
            )

    plt.show()


def main(ckpt_path: pathlib.Path):
    # Build a default cfg, then overwrite from ckpt.
    hids = [96, 96]
    lr = Constant(3e-4)
    wd = Constant(5e-2)
    cfg = TrainOfflineCfg("relu", "identity", hids, lr, wd, 1, 0.85, 0.95, 1e-3)
    ckpt_dict = load_ckpt_ez(ckpt_path, {"cfg": cfg})
    cfg = TrainOfflineCfg.fromdict(ckpt_dict["cfg"])

    nh = len(H_LABELS)
    dummy = np.zeros(1)
    alg = TrainOfflineDroneAlg.create(jr.PRNGKey(0), dummy, dummy, nh, cfg)

    ckpt_dict = load_ckpt_ez(ckpt_path, {"alg": alg})
    alg: TrainOfflineDroneAlg = ckpt_dict["alg"]
    logger.info("Loaded ckpt from {}! update_idx={}".format(ckpt_path, alg.update_idx))

    eval_obstacles = jnp.array(load_config_for_obstacles())

    bb_pos, bbh_Vh = jax2np(compute_Vh_grid(alg, eval_obstacles))

    fig_path = ckpt_path.parent.parent.parent / "Vh_{}.jpg".format(
        ckpt_path.parent.name
    )
    plot_Vh(bb_pos, bbh_Vh, eval_obstacles, fig_path)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
