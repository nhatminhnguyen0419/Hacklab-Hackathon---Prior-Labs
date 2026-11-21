"""
Inverse kinematics for the arm: damped least squares on the hand's grasp site (position +
orientation), the standard way to steer a 7-joint arm. Used by the scripted expert and to turn the
agent's Cartesian actions into joint targets.

  python sim/ik.py      # self-check: reaches a handshake pose in the reach zone
"""

import mujoco
import numpy as np

SITE = "hand_grasp_site"
# Handshake pose (spec section 7's 0 deg): fingers forward (+x), palm toward the body's midline (+y),
# thumb up (+z). Columns = the grasp site's x (out of the palm), y (across the palm), z (fingers).
HANDSHAKE = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)


def rot_about(axis, angle):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), angle)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def hand_frame(wrist):
    """Handshake pose turned by `wrist` rad about the finger axis (+ = palm turns down, spec section 7)."""
    return rot_about(HANDSHAKE[:, 2], -wrist) @ HANDSHAKE


def ik_step(m, d, dofs, target_pos, target_R, damping=0.05, rot_weight=0.3):
    """One damped-least-squares step: joint change (for `dofs`) moving the site toward the target."""
    site = m.site(SITE).id
    pos, R = d.site_xpos[site], d.site_xmat[site].reshape(3, 3)
    err_p = target_pos - pos
    q_err = np.zeros(4)
    q_cur, q_tgt = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q_cur, R.reshape(9))
    mujoco.mju_mat2Quat(q_tgt, np.asarray(target_R, float).reshape(9))
    mujoco.mju_negQuat(q_err, q_cur)
    mujoco.mju_mulQuat(q_err, q_tgt, q_err)
    err_r = np.zeros(3)
    mujoco.mju_quat2Vel(err_r, q_err, 1.0)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, site)
    J = np.vstack([jp[:, dofs], rot_weight * jr[:, dofs]])
    e = np.concatenate([err_p, rot_weight * err_r])
    return J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)


def solve(m, d, qadr, dofs, target_pos, target_R, iters=200, tol=2e-3):
    """Joint angles that put the site at the target (kinematics only; d.qpos is restored)."""
    saved = d.qpos.copy()
    lo, hi = m.jnt_range[[m.dof_jntid[i] for i in dofs]].T
    limited = m.jnt_limited[[m.dof_jntid[i] for i in dofs]].astype(bool)
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        dq = ik_step(m, d, dofs, target_pos, target_R)
        q = d.qpos[qadr] + np.clip(dq, -0.2, 0.2)
        q[limited] = np.clip(q[limited], lo[limited], hi[limited])
        d.qpos[qadr] = q
        if np.linalg.norm(dq) < tol:
            break
    mujoco.mj_kinematics(m, d)
    err = float(np.linalg.norm(d.site(SITE).xpos - target_pos))
    q = d.qpos[qadr].copy()
    d.qpos[:] = saved
    mujoco.mj_kinematics(m, d)
    return q, err


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build import ARM_JOINTS, load_model
    m = load_model()
    d = mujoco.MjData(m)
    qadr = [m.joint(j).qposadr[0] for j in ARM_JOINTS]
    dofs = [m.joint(j).dofadr[0] for j in ARM_JOINTS]
    for wrist in (0.0, 0.8, -0.8):
        target = np.array([0.40, -0.20, 1.10])
        q, err = solve(m, d, qadr, dofs, target, hand_frame(wrist))
        d.qpos[qadr] = q
        mujoco.mj_kinematics(m, d)
        R = d.site(SITE).xmat.reshape(3, 3)
        ang = np.degrees(np.arccos(np.clip((np.trace(R.T @ hand_frame(wrist)) - 1) / 2, -1, 1)))
        print(f"wrist {wrist:+.1f}: position error {err * 1000:.1f} mm, orientation error {ang:.1f} deg")
        assert err < 0.01 and ang < 5, "IK did not reach the handshake pose"
    print("OK")
