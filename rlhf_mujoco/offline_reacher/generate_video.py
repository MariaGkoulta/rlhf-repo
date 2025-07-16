#!/usr/bin/env python3
import sys
import gymnasium as gym
import cv2
from stable_baselines3 import PPO

def record_and_playback(policy, env_name="Reacher-v4", fps=30):
    """
    Records one episode of the policy in memory, then allows replaying it in a loop.

    Controls in the window:
    - During playback, press 'q' to quit immediately.
    - After playback finishes, press 'r' to replay or 'q' to exit.
    """
    # Record frames
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    done = False
    frames = []

    print("Recording episode...")
    while not done:
        frame = env.render()
        frames.append(frame)
        action, _ = policy.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()
    print(f"Recording complete: {len(frames)} frames captured.")

    # Playback loop
    cv2.namedWindow("Policy Playback", cv2.WINDOW_NORMAL)
    delay = int(1000 / fps)

    while True:
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Policy Playback", bgr)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
        # finished playing once
        print("Playback finished. Press 'r' to replay or 'q' to quit.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            continue
        else:
            break

    cv2.destroyAllWindows()


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <policy_path>", file=sys.stderr)
        sys.exit(1)

    policy_path = sys.argv[1]
    policy = PPO.load(policy_path)
    record_and_playback(policy)


if __name__ == "__main__":
    main()
