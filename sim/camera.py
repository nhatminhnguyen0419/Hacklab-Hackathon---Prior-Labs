"""
Depth camera worn at the user's head ("eye" in sim/scene.xml): the arm sees the object in pixels
instead of being told where it is. Each frame: a depth image and a segmentation image (which pixels
are the object), the object's pixels back-projected to 3D, their centre = the seen position.
Position, velocity and acceleration come from a quadratic fit over the last 0.4 s of frames.

  python -m sim.camera     # self-check: tracking error vs the simulator's truth over a few hand-overs
"""

import math

import mujoco
import numpy as np

W, H = 240, 180          # pixels (low resolution keeps the simulation fast; 160x120 lost the pen and ball)
DEPTH_NOISE = 0.002      # m at 1 m depth, growing with depth squared (stereo depth cameras)
MIN_PIXELS = 8           # fewer object pixels than this: not seen this frame (too far or hidden by a hand)
WINDOW = 0.4             # s of frames for the position/velocity/acceleration fit


class DepthCamera:
    def __init__(self, m, name="eye"):
        self.m, self.cam = m, m.camera(name).id
        samples, m.vis.quality.offsamples = m.vis.quality.offsamples, 0  # anti-aliasing blends segmentation ids
        self.r = mujoco.Renderer(m, H, W)                                 # at edges into invalid ones (crash)
        m.vis.quality.offsamples = samples
        f = 0.5 * H / math.tan(math.radians(m.cam_fovy[self.cam]) / 2)  # focal length in pixels
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        self.rx, self.ry = (u - W / 2) / f, -(v - H / 2) / f  # ray slopes in the camera frame (looks along -z)
        self.opt = mujoco.MjvOption()
        self.opt.sitegroup[:] = 0  # sites (the reach-zone box, markers) render solid in depth/segmentation: hide

    def see(self, d, body, rng, radius=0.0):
        """Seen 3D centre of the body (world frame), or None if not enough pixels. The camera sees only the
        near surface: its centre is pushed back along the line of sight by 2/3 of the radius (exact for a sphere)."""
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt)
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0  # no effect on depth/segmentation, much faster
        self.r.enable_depth_rendering()
        depth = self.r.render()
        self.r.disable_depth_rendering()
        self.r.enable_segmentation_rendering()
        seg = self.r.render()
        self.r.disable_segmentation_rendering()
        ids = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)  # other pixel types carry other ids
        mask = (ids >= 0) & (self.m.geom_bodyid[np.clip(ids, 0, self.m.ngeom - 1)] == body)
        self.depth, self.mask = depth, mask  # last frame, for display (server.py)
        if mask.sum() < MIN_PIXELS:
            return None
        z = depth[mask] + rng.normal(0, DEPTH_NOISE, mask.sum()) * depth[mask] ** 2
        pts = np.column_stack([self.rx[mask] * z, self.ry[mask] * z, -z])  # camera frame
        c = pts.mean(axis=0)
        c += c / np.linalg.norm(c) * (2 / 3) * radius
        return d.cam_xpos[self.cam] + d.cam_xmat[self.cam].reshape(3, 3) @ c

    def picture(self):
        """Last frame as an RGB image: depth in grey (near = bright), the object's pixels in pink."""
        if getattr(self, "depth", None) is None:
            return None
        g = (255 * np.clip((3.0 - self.depth) / 2.6, 0, 1)).astype(np.uint8)  # 0.4 m white .. 3 m black
        img = np.repeat(g[..., None], 3, axis=2)
        img[self.mask] = (255, 77, 191)
        return img

    def close(self):
        self.r.close()


class Tracker:
    """Position, velocity, acceleration from noisy seen positions: a quadratic fit per axis over WINDOW s."""

    def __init__(self):
        self.t, self.p = [], []

    def add(self, t, p):
        if p is not None:
            self.t.append(t)
            self.p.append(p)

    def estimate(self, t_now):
        """(p, v, a) at t_now, or None before 4 frames in the window. Hidden frames: extrapolates the fit."""
        keep = [i for i, t in enumerate(self.t) if t_now - t <= WINDOW + 1e-9]
        if len(keep) < 4:
            return None
        dt = np.array([self.t[i] for i in keep]) - t_now
        P = np.array([self.p[i] for i in keep])
        c2, c1, c0 = np.polyfit(dt, P, 2)  # each (3,): p(dt) = c0 + c1 dt + c2 dt^2
        return c0, c1, 2 * c2


if __name__ == "__main__":
    from sim.env import ArmEnv
    env = ArmEnv(camera=True)
    errs = {"position": [], "velocity": []}
    seen_frames = 0
    for seed in range(5):
        env.reset(seed=10_000 + seed)
        for _ in range(env.n_steps):
            env.step(env.expert_action())
            if env.seen is not None and env.weld_on and env.t_touch is None and env.t_carrier0 is not None:
                errs["position"].append(np.linalg.norm(env.seen[0] - env.opos))
                errs["velocity"].append(np.linalg.norm(env.seen[1] - env.ovel))
            if env.t_touch is not None:
                break
        seen_frames += len(env.tracker.t)
    for k, e in errs.items():
        print(f"{k} error: median {np.median(e) * 100:.1f} cm{'/s' if k == 'velocity' else ''}, "
              f"90th pct {np.percentile(e, 90) * 100:.1f}  ({len(e)} frames)")
    assert np.median(errs["position"]) < 0.05, "camera position off by more than 5 cm"
    print(f"OK: {seen_frames} frames with the object seen")
