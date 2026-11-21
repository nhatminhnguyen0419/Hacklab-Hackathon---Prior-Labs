"""
Builds the new prosthetic-arm model: sim/scene.xml (floor, torso, objects, carrier) plus
two MuJoCo Menagerie models attached in code (MjSpec):
  Kinova Gen3 arm (7 joints)      -> hangs from the scene's "shoulder" site, names prefixed "arm_"
  TetherIA Aero Hand Open (right) -> on the arm's wrist ("pinch_site"), names prefixed "hand_"

  python sim/build.py        # self-check: compiles, prints the model, renders sim/preview.png
"""

import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_XML = os.path.join(HERE, "menagerie", "kinova_gen3", "gen3.xml")
HAND_XML = os.path.join(HERE, "menagerie", "tetheria_aero_hand_open", "right_hand.xml")
SCENE_XML = os.path.join(HERE, "scene.xml")

ARM_JOINTS = [f"arm_joint_{i}" for i in range(1, 8)]
HAND_TENDONS = ["hand_right_index_A_tendon", "hand_right_middle_A_tendon", "hand_right_ring_A_tendon",
                "hand_right_pinky_A_tendon", "hand_right_th1_A_tendon", "hand_right_th2_A_tendon"]
THUMB_ABD = "hand_right_thumb_A_cmc_abd"


def _spec(path):
    spec = mujoco.MjSpec.from_file(path)
    for key in list(spec.keys):  # their keyframes don't fit the combined model
        spec.delete(key)
    return spec


ARM_SCALE = 0.6  # the Gen3 is ~1.2 m long; 0.6 gives a human arm's ~0.7 m from shoulder to wrist


def _scale(spec, s):
    """Uniformly scale a model: lengths by s, masses by s^3, inertias by s^5."""
    for mesh in spec.meshes:
        mesh.scale = np.asarray(mesh.scale) * s
    for body in spec.bodies:
        body.pos = np.asarray(body.pos) * s
        body.ipos = np.asarray(body.ipos) * s
        body.mass = body.mass * s ** 3
        body.inertia = np.asarray(body.inertia) * s ** 5
        body.fullinertia = np.asarray(body.fullinertia) * s ** 5
    for el in list(spec.geoms) + list(spec.sites):
        el.pos = np.asarray(el.pos) * s
        if el.type != mujoco.mjtGeom.mjGEOM_MESH:
            el.size = np.asarray(el.size) * s


# N per tendon motor (0 = the model's own, unlimited). Expert over 40 episodes: unlimited 30/40 with
# 150-515 N squeeze; 40 N 32/40 with 54-124 N; 15 N 13/40; 6 N 4/40
GRIP_FORCE = float(os.environ.get("GRIP_FORCE", 40))


def build_spec():
    scene, arm, hand = _spec(SCENE_XML), _spec(ARM_XML), _spec(HAND_XML)
    _scale(arm, ARM_SCALE)
    if GRIP_FORCE:  # a prosthetic squeezes gently: cap each tendon motor's pull
        for act in hand.actuators:
            if "tendon" in act.name:
                act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                act.forcerange = [-GRIP_FORCE, GRIP_FORCE]
    scene.site("shoulder").attach_body(arm.body("base_link"), "arm_", "")
    scene.site("arm_pinch_site").attach_body(hand.body("tetheria_mount"), "hand_", "")  # after: names stay "hand_*"
    # the base is fixed to the world, so MuJoCo doesn't skip its contact with the first link: they
    # touch at the joint and locked joint 1 with millions of newtons
    scene.add_exclude(bodyname1="arm_base_link", bodyname2="arm_shoulder_link")
    # the spec's one motion sensor: an IMU in the thumb
    scene.body("hand_right_thumb_proximal_link").add_site(name="thumb_imu", size=[0.004] * 3)
    for kind, name in ((mujoco.mjtSensor.mjSENS_ACCELEROMETER, "thumb_acc"), (mujoco.mjtSensor.mjSENS_GYRO, "thumb_gyro")):
        scene.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="thumb_imu")
    return scene


def load_model():
    return build_spec().compile()


def grip_ctrl(model, g):
    """One grip value g (0 open .. 1 closed) -> the hand's 7 motor targets: a power-grasp synergy,
    as prosthetic hands are driven. Short tendons flex the fingers; the thumb swings to oppose."""
    g = float(np.clip(g, 0.0, 1.0))
    out = {}
    for name in HAND_TENDONS:
        lo, hi = model.actuator_ctrlrange[model.actuator(name).id]
        out[name] = hi - g * (hi - lo)
    lo, hi = model.actuator_ctrlrange[model.actuator(THUMB_ABD).id]
    out[THUMB_ABD] = lo + 0.7 * (hi - lo)  # thumb kept in opposition, a fixed pose (prosthetic "power grip")
    return out


if __name__ == "__main__":
    m = load_model()
    d = mujoco.MjData(m)
    print(f"bodies {m.nbody}, joints {m.njnt}, actuators {m.nu}: {[m.actuator(i).name for i in range(m.nu)]}")
    for name, v in grip_ctrl(m, 0.0).items():
        d.ctrl[m.actuator(name).id] = v
    mujoco.mj_forward(m, d)
    for _ in range(500):
        mujoco.mj_step(m, d)
    hp = d.site("hand_grasp_site").xpos
    print("hand grasp_site at", np.round(hp, 3), " arm joints", np.round([d.joint(j).qpos[0] for j in ARM_JOINTS], 2))
    assert np.all(np.isfinite(d.qpos)), "simulation blew up"
    r = mujoco.Renderer(m, 480, 640)
    r.update_scene(d, camera="demo")
    from PIL import Image
    Image.fromarray(r.render()).save(os.path.join(HERE, "preview.png"))
    print("saved sim/preview.png")
