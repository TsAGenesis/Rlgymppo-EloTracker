import argparse
import numpy as np
import rlgym_sim
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import (
    VelocityPlayerToBallReward,
    VelocityBallToGoalReward,
    EventReward,
)
from lookup_act import LookupAction
from advancedobs import AdvancedObsPadder   
from state_setters import DefaultState      


OBS_BUILDER   = AdvancedObsPadder(team_size=3, expanding=False)
ACTION_PARSER = LookupAction()

def parse_layers(layer_str: str):
    return tuple(int(x) for x in layer_str.split(",") if x.strip())


def build_rocketsim_env(tick_skip: int = 8, team_size: int = 1):
    game_tick_rate  = 120
    timeout_seconds = 10
    timeout_ticks   = int(round(timeout_seconds * game_tick_rate / tick_skip))
    terminal_conditions = [
        NoTouchTimeoutCondition(timeout_ticks),
        GoalScoredCondition(),
    ]


    reward_fn = CombinedReward(
        reward_functions=[
            VelocityPlayerToBallReward(),
            VelocityBallToGoalReward(),
            EventReward(team_goal=1, concede=-1, demo=0.1),
        ],
        reward_weights=[0.01, 0.1, 10.0],
    )


    obs_builder   = OBS_BUILDER
    action_parser = ACTION_PARSER
    state_setter  = DefaultState()

    return rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=team_size,
        spawn_opponents=True,
        terminal_conditions=terminal_conditions,
        reward_fn=reward_fn,
        obs_builder=obs_builder,
        action_parser=action_parser,
        state_setter=state_setter,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO mit AdvancedObsPadder, DefaultState, anpassbaren Layer-Sizes & Game-Speed"
    )
    parser.add_argument(
        "--policy-layers", type=str, default="256,256",
        help="Comma-separated hidden layer sizes für die Policy"
    )
    parser.add_argument(
        "--critic-layers", type=str, default="256,256",
        help="Comma-separated hidden layer sizes für den Critic"
    )
    parser.add_argument(
        "--tick-skip", type=int, default=8,
        help="tick_skip fürs Env (höher = schnellere Spiele)"
    )
    parser.add_argument(
        "--team-size", type=int, default=1,
        help="Autos pro Team (z.B. 1, 2 oder 3)"
    )
    parser.add_argument(
        "--n-proc", type=int, default=32,
        help="Parallel-Prozesse für PPO"
    )
    args = parser.parse_args()

    policy_layers = parse_layers(args.policy_layers)
    critic_layers = parse_layers(args.critic_layers)

    from rlgym_ppo import Learner

    Learner(
        lambda: build_rocketsim_env(args.tick_skip, args.team_size),
        n_proc=args.n_proc,
        policy_layer_sizes=policy_layers,
        critic_layer_sizes=critic_layers,
        render=True,
        render_delay=1e-5,
    ).learn()