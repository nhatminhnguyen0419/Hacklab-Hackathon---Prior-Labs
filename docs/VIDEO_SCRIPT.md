# 3-minute video script

The video is three parts, stitched by `python make_final_video.py`:

| Time | Part | Who | File you provide |
|---|---|---|---|
| 0:00–0:20 | **On camera:** who we are, the problem | one team member | `video_parts/intro.mp4` |
| 0:20–2:40 | **Footage + voice-over:** idea, demo, results | anyone (voice only) | `video_parts/voiceover.m4a` (or .mp3/.wav) |
| 2:40–3:00 | **On camera:** what’s next, GitHub, thanks | one team member | `video_parts/outro.mp4` |

The footage (`demo_replays/broll.mp4`, 2:20) is already made, and its sections match the timings below. Record the voice-over while watching it, so each line lands on its picture. Any part you don’t record is replaced by a text card, so the video always comes out at 3:00.

Speaking pace: about 150 words a minute. Each block below fits its time slot with room to breathe.

---

## 0:00–0:20 · On camera (intro.mp4)

> Hi, we’re The Big Gang: Minh, Quan and Kazuki.
> Imagine someone who has lost an arm at the shoulder. When a friend hands them a cup, a prosthetic arm should take it safely, and only when they ask.
> We built that arm in simulation, with TabPFN as its spatial awareness.

---

## 0:20–2:40 · Voice-over (voiceover.m4a, 2:20 long)

**0:20 · The problem** *(footage: “The problem” card)*
> The user says “ready”. Nothing moves before that. Then the arm has to understand an object that is still moving in someone’s hand.

**0:30 · Robots run on tables** *(footage: a table of rows)*
> Robots run on tables. Every moment of our arm, the object’s position and speed, the forces on the fingers, is one row. Past rows become TabPFN’s context.

**0:42 · Three questions** *(footage: three questions)*
> From those rows, TabPFN answers three questions, with no training run: what is it, where will it stop, and will this grasp hold?

**0:52 · Hand-over 1, bottle**
> Here’s a real run. A person hands over a bottle. TabPFN identifies it. The pink dot is where TabPFN predicts it will stop, updated every fifth of a second as the hand slows down.

**1:08 · Hand-over 2, cup**
> Our reinforcement-learning agent never sees the simulator’s answer. It acts only on TabPFN’s prediction, moving a real robot-arm model, a Kinova Gen3, and turning the wrist to suit the object.

**1:24 · Hand-over 3, pen**
> The hand is the open-source TetherIA Aero Hand, tendon-driven like real prosthetics. The grasp is pure contact physics. Nothing is glued.

**1:40 · Hand-over 4, seen through the depth camera** *(footage: depth picture in the corner)*
> This time the arm sees the object only through a depth camera at the user's head, the pink pixels in the corner. TabPFN learned the camera's errors in context. At first touch, it predicts whether the grasp will hold.

**1:56 · The re-grasp** *(footage: “When TabPFN says risky”)*
> And this is why that matters. TabPFN gives this grasp a low chance. So the hand opens and grasps again, while the person keeps holding it. This time it holds.

**2:12 · Results** *(footage: three result cards)*
> On hand-overs it never saw, TabPFN predicts the stop point to six millimetres, six times better than linear regression.
> It ranks risky grasps with an AUC of zero point nine six.
> And the arm, acting on these predictions and re-grasping on TabPFN’s warning, succeeds ninety-two percent of the time; eighty-one percent when it sees the object only through the camera.

---

## 2:40–3:00 · On camera (outro.mp4)

> Next, we’d use real hand-over recordings as TabPFN’s context, and run the same pipeline on a real depth camera and arm.
> Everything, code, models and data, is on our GitHub. Thanks for watching!

---

## Recording tips

- **Camera clips:** landscape (horizontal), eye level, a quiet room, light in front of you, not behind. A phone is fine. Aim for 18–20 s each; the tool trims anything longer than 20 s.
- **Voice-over:** a phone’s voice-memo app is fine. Start talking right after you press record, and play `demo_replays/broll.mp4` on a screen while you read, so the lines line up. Up to 2:20; a shorter recording ends in silence over the last footage.
- **Then run** `python make_final_video.py` → `demo_replays/final_video.mp4` (3:00, 1280×720).

Numbers quoted above are from held-out tests in this repository: stop point 0.6 cm median (linear regression 3.6 cm, 300 hand-overs), hold prediction ROC AUC 0.961 (598 grasps), grasp success 92% with the re-grasp (89% without, 200 episodes); with the depth camera, stop point 2.1 cm (linear regression 4.6 cm) and grasp success 81%.
