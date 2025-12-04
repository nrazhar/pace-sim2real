# © 2025 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0

from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from isaaclab.assets import ArticulationCfg
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch

# Define Actuator Configs matching G1_29DOF_CFG
G1_LEGS_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[
        ".*_hip_yaw_joint",
        ".*_hip_roll_joint",
        ".*_hip_pitch_joint",
        ".*_knee_joint",
    ],
    effort_limit={
        ".*_hip_yaw_joint": 88.0,
        ".*_hip_roll_joint": 88.0,
        ".*_hip_pitch_joint": 88.0,
        ".*_knee_joint": 139.0,
    },
    velocity_limit={
        ".*_hip_yaw_joint": 32.0,
        ".*_hip_roll_joint": 32.0,
        ".*_hip_pitch_joint": 32.0,
        ".*_knee_joint": 20.0,
    },
    stiffness={
        ".*_hip_yaw_joint": 100.0,
        ".*_hip_roll_joint": 100.0,
        ".*_hip_pitch_joint": 100.0,
        ".*_knee_joint": 200.0,
    },
    damping={
        ".*_hip_yaw_joint": 2.5,
        ".*_hip_roll_joint": 2.5,
        ".*_hip_pitch_joint": 2.5,
        ".*_knee_joint": 5.0,
    },
    armature={
        ".*_hip_.*": 0.03,
        ".*_knee_joint": 0.03,
    },
    saturation_effort=180.0,
    encoder_bias=[0.0] * 8,  # 4 joints * 2 legs
    max_delay=10,
)

G1_FEET_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
    effort_limit={
        ".*_ankle_pitch_joint": 50.0,
        ".*_ankle_roll_joint": 50.0,
    },
    velocity_limit={
        ".*_ankle_pitch_joint": 37.0,
        ".*_ankle_roll_joint": 37.0,
    },
    stiffness={
        ".*_ankle_pitch_joint": 20.0,
        ".*_ankle_roll_joint": 20.0,
    },
    damping={
        ".*_ankle_pitch_joint": 0.2,
        ".*_ankle_roll_joint": 0.1,
    },
    armature=0.03,
    saturation_effort=80.0,
    encoder_bias=[0.0] * 4,  # 2 joints * 2 feet
    max_delay=10,
)


@configclass
class G1PaceCfg(PaceCfg):
    """Pace configuration for G1 robot."""
    robot_name: str = "g1_sim"
    data_dir: str = "g1_sim/chirp_data.pt"
    bounds_params: torch.Tensor = torch.zeros((49, 2))  # 12 + 12 + 12 + 12 + 1 = 49 parameters to optimize
    joint_order: list[str] = [
        "left_hip_yaw_joint",
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]

    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[:12, 0] = 1e-5
        self.bounds_params[:12, 1] = 1.0  # armature between 1e-5 - 1.0 [kgm2]
        self.bounds_params[12:24, 1] = 7.0  # dof_damping between 0.0 - 7.0 [Nm s/rad]
        self.bounds_params[24:36, 1] = 0.5  # friction between 0.0 - 0.5
        self.bounds_params[36:48, 0] = -0.1
        self.bounds_params[36:48, 1] = 0.1  # bias between -0.1 - 0.1 [rad]
        self.bounds_params[48, 1] = 10.0  # delay between 0.0 - 10.0 [sim steps]


@configclass
class G1PaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for G1 robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(0.7071, 0, 0, 0.7071)
        ),
        actuators={
            "legs": G1_LEGS_PACE_ACTUATOR_CFG,
            "feet": G1_FEET_PACE_ACTUATOR_CFG,
        }
    )


@configclass
class G1PaceEnvCfg(PaceSim2realEnvCfg):

    scene: G1PaceSceneCfg = G1PaceSceneCfg()
    sim2real: PaceCfg = G1PaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.0025  # 400Hz simulation
        self.decimation = 1  # 400Hz control
