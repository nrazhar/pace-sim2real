# © 2025 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg

# -------------------------------------------------------------------------
# Base Robot Configuration (HOKU - Flat, Left Ankle Only)
# -------------------------------------------------------------------------
# Actuator to freeze joints in place

# LOCKED_JOINT_ACTUATOR_CFG = PaceDCMotorCfg(
#     joint_names_expr=[],  # Will be assigned in SceneCfg
#     velocity_limit=20.0,   # 0 velocity limit = locked in place
#     stiffness=1000.0,    # Very high stiffness
#     damping=1.0,       # High damping
#     # armature=0.0032,         # No armature effect needed for static joints
#     saturation_effort=50000.0,
#     encoder_bias=[0.0] * 12,     # Initialize with 0 bias
#     max_delay=0,          # No delay for static joints
#     effort_limit=50000.0,   # High effort to allow holding position against gravity
# )

LOCKED_JOINT_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[], # Assigned in SceneCfg
    stiffness=10000.0,   # Rigid
    damping=1000.0,      # No oscillation
)

HOKU_FLAT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/azhar/ws/n_robotics_robot_description/mujoco_robot/mujoco_robot.usd",
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
        pos=(0.0, 0.0, 0.8),
        rot=(0.7071, 0.0, -0.7071, 0.0),  # -90° around Y-axis - flat horizontal face up
        joint_pos={
            "hip_pitch_.*": -0.10,
            "knee_pitch_.*": 0.30,
            "ankle_pitch_.*": -0.20,
            "torso_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={},  # Will be populated by SceneCfg
)

# -------------------------------------------------------------------------
# Pace Actuator Configuration - Left Ankle Only
# -------------------------------------------------------------------------

HOKU_LEFT_ANKLE_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[
        "ankle_pitch_left",
        "ankle_roll_left",
    ],
    effort_limit=45.0,
    velocity_limit=16.0,  # ~916 deg/s
    stiffness=3.0,
    damping=0.5,
    armature=0.0032,
    saturation_effort=45.0,
    encoder_bias=[0.0] * 2,  # 2 joints for left ankle
    max_delay=10,
)


# -------------------------------------------------------------------------
# Pace Environment Configs
# -------------------------------------------------------------------------


@configclass
class HokuFlatLeftAnklePaceCfg(PaceCfg):
    """Pace configuration for Hoku robot - flat with left ankle only."""

    robot_name: str = "hoku_flat_left_ankle_sim"
    data_dir: str = "hoku_flat_left_ankle_sim/chirp_data.pt"
    # 2 joints * 4 params + 1 delay = 9 parameters
    bounds_params: torch.Tensor = torch.zeros((9, 2))

    joint_order: list[str] = [
        "ankle_pitch_left",
        "ankle_roll_left",
    ]

    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[:2, 0] = 1e-5
        self.bounds_params[:2, 1] = 1.0  # armature [kgm2]
        self.bounds_params[2:4, 1] = 7.0  # dof_damping [Nm s/rad]
        self.bounds_params[4:6, 1] = 0.5  # friction
        self.bounds_params[6:8, 0] = -0.1
        self.bounds_params[6:8, 1] = 0.1  # bias [rad]
        self.bounds_params[8, 1] = 10.0  # delay [sim steps]


@configclass
class HokuFlatLeftAnklePaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for Hoku robot flat with left ankle only."""

    robot: ArticulationCfg = HOKU_FLAT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(0.7071, 0.0, -0.7071, 0.0),
            joint_pos={
                "hip_pitch_.*": 0.0,
                "knee_pitch_.*": 0.0,
                "ankle_pitch_.*": 0.0,
                "ankle_roll_.*": 0.0,
                "torso_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            # 1. The joint you want to move (Ankle)
            "left_ankle": HOKU_LEFT_ANKLE_PACE_ACTUATOR_CFG,
            # 2. The joints you want to FIX (Hip, Knee, Torso)
            # This applies the high-stiffness actuator to them
            "locked_joints": LOCKED_JOINT_ACTUATOR_CFG.replace(
                joint_names_expr=[
                    "hip_.*",         # Catches pitch, roll, AND yaw
                    "knee_.*",        # Catches knees
                    "torso_.*",       # Catches torso
                    "ankle_.*_right"  # Catches the entire right ankle
                ]
            ),
        },
    )


@configclass
class HokuFlatLeftAnklePaceEnvCfg(PaceSim2realEnvCfg):

    scene: HokuFlatLeftAnklePaceSceneCfg = HokuFlatLeftAnklePaceSceneCfg()
    sim2real: PaceCfg = HokuFlatLeftAnklePaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.actions.joint_pos.joint_names = [
            "ankle_pitch_left",
            "ankle_roll_left",
        ]

        # robot sim and control settings
        self.sim.dt = 0.0025  # 400Hz simulation
        self.decimation = 1  # 400Hz control
