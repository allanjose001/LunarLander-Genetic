import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset()

done = False
episode_reward = 0

while not done:
    # ação aleatorio só para testar o ambiente
    action = env.action_space.sample()

    # envia a ação para o ambiente e recebe os resultados
    obs, reward, done, truncated, info = env.step(action)

    episode_reward += reward

    print("Observação: ", obs)
    print("Recompensa: ", reward)
    print("Ação: ", action)
    print("-"*40)

    if truncated:
        break

print("Recompensa total do episódio: ", episode_reward)

env.close()