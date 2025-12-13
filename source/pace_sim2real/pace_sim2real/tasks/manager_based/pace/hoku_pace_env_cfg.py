# © 2025 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DCMotorCfg
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
import torch

# -------------------------------------------------------------------------
# Base Robot Configuration (HOKU)
# -------------------------------------------------------------------------

HOKU_CFG = ArticulationCfg(
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
# Pace Actuator Configurations
# -------------------------------------------------------------------------

HOKU_LEGS_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[
        "hip_yaw_.*",
        "hip_roll_.*",
        "hip_pitch_.*",
        "knee_pitch_.*",
    ],
    effort_limit={
        "hip_.*": 120.0,
        "knee_.*": 120.0,
    },
    velocity_limit={
        "hip_.*": 12.0,  # ~687 deg/s
        "knee_.*": 12.0,
    },
    stiffness={
        "hip_yaw_.*": 100.0,
        "hip_roll_.*": 280.0,
        "hip_pitch_.*": 280.0,
        "knee_pitch_.*": 280.0,
    },
    damping={
        "hip_yaw_.*": 5.0,
        "hip_roll_.*": 5.0,
        "hip_pitch_.*": 5.0,
        "knee_pitch_.*": 4.0,
    },
    armature={
        "hip_yaw_.*": 0.0166,
        "hip_roll_.*": 0.0272,
        "hip_pitch_.*": 0.0166,
        "knee_pitch_.*": 0.0272,
    },
    saturation_effort=120.0,
    encoder_bias=[0.0] * 8,  # 4 joints * 2 legs
    max_delay=10,
)

HOKU_FEET_PACE_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[
        "ankle_pitch_.*", 
        "ankle_roll_.*"
    ],
    effort_limit=45.0,
    velocity_limit=16.0,  # ~916 deg/s
    stiffness=45.0,
    damping=2.0,
    armature=0.0032,
    saturation_effort=45.0,
    encoder_bias=[0.0] * 4,  # 2 joints * 2 feet
    max_delay=10,
)

# -------------------------------------------------------------------------
# Pace Environment Configs
# -------------------------------------------------------------------------

@configclass
class HokuPaceCfg(PaceCfg):
    """Pace configuration for Hoku robot."""
    robot_name: str = "hoku_sim"
    data_dir: str = "hoku_sim/chirp_data.pt"
    # 12 joints * 4 params + 1 delay = 49 parameters
    bounds_params: torch.Tensor = torch.zeros((49, 2))  
    
    joint_order: list[str] = [
        "hip_yaw_left",
        "hip_roll_left",
        "hip_pitch_left",
        "knee_pitch_left",
        "ankle_pitch_left",
        "ankle_roll_left",
        "hip_yaw_right",
        "hip_roll_right",
        "hip_pitch_right",
        "knee_pitch_right",
        "ankle_pitch_right",
        "ankle_roll_right",
    ]

    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[:12, 0] = 1e-5
        self.bounds_params[:12, 1] = 1.0   # armature [kgm2]
        self.bounds_params[12:24, 1] = 7.0 # dof_damping [Nm s/rad]
        self.bounds_params[24:36, 1] = 0.5 # friction
        self.bounds_params[36:48, 0] = -0.1
        self.bounds_params[36:48, 1] = 0.1 # bias [rad]
        self.bounds_params[48, 1] = 10.0   # delay [sim steps]


@configclass
class HokuPaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for Hoku robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = HOKU_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(0.7071, 0.0, -0.7071, 0.0),  # -90° around Y-axis - flat horizontal face up
            joint_pos={
                "hip_pitch_.*": -0.10,
                "knee_pitch_.*": 0.30,
                "ankle_pitch_.*": -0.20,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": HOKU_LEGS_PACE_ACTUATOR_CFG,
            "feet": HOKU_FEET_PACE_ACTUATOR_CFG,
            # Note: Torso actuators can be added here as standard DCMotorCfg 
            # if they are not part of the Pace optimization loop
        }
    )


@configclass
class HokuPaceEnvCfg(PaceSim2realEnvCfg):
    
    scene: HokuPaceSceneCfg = HokuPaceSceneCfg()
    sim2real: PaceCfg = HokuPaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.0025  # 400Hz simulation
        self.decimation = 1   # 400Hz control