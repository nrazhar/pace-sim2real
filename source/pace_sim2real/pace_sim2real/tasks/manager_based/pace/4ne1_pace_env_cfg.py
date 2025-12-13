# © 2025 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch

# -------------------------------------------------------------------------
# Base Articulation Configuration
# -------------------------------------------------------------------------
NEURA_4NE1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/azhar/ws/neura_4ne1_g3_2/4ne1.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            ".*_knee": 0.1,
            ".*_hip_y": -0.1,
            "torso_.*": 0.0,
            ".*_elbow": 0.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_x",
                ".*_hip_y",
                ".*_hip_z",
                ".*_knee",
            ],
            stiffness={
                ".*_hip_x": 200.0,
                ".*_hip_y": 241.805,
                ".*_hip_z": 51.618,
                ".*_knee": 113.5,
            },
            damping={
                ".*_hip_x": 14.451,
                ".*_hip_y": 30.788,
                ".*_hip_z": 6.572,
                ".*_knee": 14.451,
            },
            effort_limit={
                ".*_hip_.*": 200.0,
                ".*_knee": 200.0,
            },
            velocity_limit={
                ".*_hip_.*": 20.0,
                ".*_knee": 20.0,
            },
            armature=0.01,
            saturation_effort=250.0,
        ),
        "feet": DCMotorCfg(
            # FIXED: Removed inner/outer, used only existing x/y joints
            joint_names_expr=[
                ".*_ankle_x",
                ".*_ankle_y",
            ],
            # Applied G1-like defaults because provided gains for x/y were 0
            stiffness={
                ".*_ankle_x": 20.0,
                ".*_ankle_y": 20.0,
            },
            damping={
                ".*_ankle_x": 1.0,
                ".*_ankle_y": 1.0,
            },
            effort_limit=50.0,
            velocity_limit=20.0,
            armature=0.01,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=[
                "torso_x",
                "torso_y",
                "torso_z",
            ],
            stiffness={
                "torso_.*": 100.0,
            },
            damping={
                "torso_.*": 6.572,
            },
            effort_limit=200.0,
            velocity_limit=20.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_x",
                ".*_shoulder_y",
                ".*_shoulder_z",
                ".*_elbow",
                ".*_wrist_x",
                ".*_wrist_y",
                ".*_wrist_z",
            ],
            stiffness={
                ".*_shoulder_.*": 40.0,
                ".*_elbow": 40.0,
                ".*_wrist_.*": 10.0,
            },
            damping={
                ".*_shoulder_.*": 1.0,
                ".*_elbow": 1.0,
                ".*_wrist_.*": 1.0,
            },
            effort_limit=80.0,
            velocity_limit=10.0,
            armature=0.005,
        ),
    },
    prim_path="/World/envs/env_.*/Robot",
)

# -------------------------------------------------------------------------
# Pace Actuator Configurations (Sim2Real wrappers)
# -------------------------------------------------------------------------
NEURA_LEGS_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[
        ".*_hip_x",
        ".*_hip_y",
        ".*_hip_z",
        ".*_knee",
    ],
    stiffness={
        ".*_hip_x": 200.0,
        ".*_hip_y": 241.805,
        ".*_hip_z": 51.618,
        ".*_knee": 113.5,
    },
    damping={
        ".*_hip_x": 14.451,
        ".*_hip_y": 30.788,
        ".*_hip_z": 6.572,
        ".*_knee": 14.451,
    },
    effort_limit={
        ".*_hip_.*": 200.0,
        ".*_knee": 200.0,
    },
    velocity_limit={
        ".*_hip_.*": 20.0,
        ".*_knee": 20.0,
    },
    armature=0.01,
    saturation_effort=250.0,
    encoder_bias=[0.0] * 8,  # 4 joints * 2 legs
    max_delay=10,
)

NEURA_FEET_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    # FIXED: Removed inner/outer here as well
    joint_names_expr=[
        ".*_ankle_x",
        ".*_ankle_y",
    ],
    stiffness={
        ".*_ankle_x": 20.0,
        ".*_ankle_y": 20.0,
    },
    damping={
        ".*_ankle_x": 1.0,
        ".*_ankle_y": 1.0,
    },
    effort_limit=50.0,
    velocity_limit=20.0,
    armature=0.01,
    saturation_effort=80.0,
    encoder_bias=[0.0] * 4, # 2 joints * 2 feet = 4 total
    max_delay=10,
)

# -------------------------------------------------------------------------
# Pace Environment Configurations
# -------------------------------------------------------------------------
@configclass
class Neura4NE1PaceCfg(PaceCfg):
    """Pace configuration for Neura 4NE1 robot."""
    robot_name: str = "neura_4ne1_sim"
    data_dir: str = "neura_4ne1_sim/chirp_data.pt"
    
    # 12 joints total (Hips(3) + Knee(1) + Ankle(2)) * 2 legs = 12
    # 49 params = 12 stiffness + 12 damping + 12 friction + 12 bias + 1 delay
    bounds_params: torch.Tensor = torch.zeros((49, 2)) 
    
    joint_order: list[str] = [
        "left_hip_x",
        "left_hip_y",
        "left_hip_z",
        "left_knee",
        "left_ankle_x",
        "left_ankle_y",
        "right_hip_x",
        "right_hip_y",
        "right_hip_z",
        "right_knee",
        "right_ankle_x",
        "right_ankle_y",
    ]

    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[:12, 0] = 1e-5
        self.bounds_params[:12, 1] = 1.0   # armature
        self.bounds_params[12:24, 1] = 15.0 # Higher damping limit
        self.bounds_params[24:36, 1] = 0.5  # friction
        self.bounds_params[36:48, 0] = -0.1
        self.bounds_params[36:48, 1] = 0.1  # bias
        self.bounds_params[48, 1] = 10.0    # delay


@configclass
class Neura4NE1PaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for Neura 4NE1 robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = NEURA_4NE1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                ".*_knee": 0.1,
                ".*_hip_y": -0.1,
                "torso_.*": 0.0,
                ".*_elbow": 0.5,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": NEURA_LEGS_PACE_ACTUATOR_CFG,
            "feet": NEURA_FEET_PACE_ACTUATOR_CFG,
        }
    )


@configclass
class Neura4NE1PaceEnvCfg(PaceSim2realEnvCfg):
    scene: Neura4NE1PaceSceneCfg = Neura4NE1PaceSceneCfg()
    sim2real: PaceCfg = Neura4NE1PaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.0025  # 400Hz simulation
        self.decimation = 1   # 400Hz control