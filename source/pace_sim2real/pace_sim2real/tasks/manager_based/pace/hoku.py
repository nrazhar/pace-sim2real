import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import DCMotorCfg
from omni.isaac.lab.assets import ArticulationCfg

HOKU_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Updated path as requested
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
        # Adjusted height estimate for a biped
        pos=(0.0, 0.0, 0.80),
        rot=(1.0, 0.0, 0.0, 0.0),  # Assuming standard orientation (w, x, y, z)
        joint_pos={
            # Initial crouch/stand pose (in radians)
            "hip_pitch_.*": -0.10,
            "knee_pitch_.*": 0.30,
            "ankle_pitch_.*": -0.20,
            "torso_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Group 1: High Torque Joints (Hips & Knees) - Max Force 120 Nm
        "legs": DCMotorCfg(
            joint_names_expr=[
                "hip_pitch_.*",
                "hip_roll_.*",
                "hip_yaw_.*",
                "knee_pitch_.*",
            ],
            effort_limit={
                "hip_.*": 120.0,
                "knee_.*": 120.0,
            },
            # Converted ~687 deg/s to ~12.0 rad/s
            velocity_limit={
                "hip_.*": 12.0,
                "knee_.*": 12.0,
            },
            # Tuned initial gains (Higher torque joints usually need higher stiffness)
            stiffness={
                "hip_.*": 150.0,
                "knee_.*": 200.0,
            },
            damping={
                "hip_.*": 5.0,
                "knee_.*": 5.0,
            },
            armature=0.03,  # Estimated armature
            saturation_effort=120.0,
        ),
        # Group 2: Lower Torque Joints (Ankles) - Max Force 45 Nm
        "feet": DCMotorCfg(
            joint_names_expr=["ankle_pitch_.*", "ankle_roll_.*"],
            effort_limit=45.0,
            # Converted ~916 deg/s to ~16.0 rad/s
            velocity_limit=16.0,
            stiffness={
                "ankle_pitch_.*": 40.0,
                "ankle_roll_.*": 40.0,
            },
            damping={
                "ankle_pitch_.*": 2.0,
                "ankle_roll_.*": 2.0,
            },
            armature=0.03,
            saturation_effort=45.0,
        ),
        # Group 3: Torso Joints - Max Force 45 Nm
        "torso": DCMotorCfg(
            joint_names_expr=["torso_yaw", "torso_roll"],
            effort_limit=45.0,
            # Converted ~916 deg/s to ~16.0 rad/s
            velocity_limit=16.0,
            stiffness=100.0,
            damping=2.5,
            armature=0.01,
            saturation_effort=45.0,
        ),
    },
    # Regex to find the Prim path in the stage
    prim_path="/World/envs/env_.*/Robot",
)
