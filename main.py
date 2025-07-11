import gymnasium as gym
import numpy as np
import pygame

KEY_ACTIONS = {
    pygame.K_UP: 2,      # motor principal
    pygame.K_LEFT: 1,    # motor esquerdo
    pygame.K_RIGHT: 3,   # motor direito
    pygame.K_DOWN: 0     # sem ação
}

# Inicializa pygame
pygame.init()
screen = pygame.display.set_mode((400, 100))
pygame.display.set_caption("Controle LunarLander pelo teclado (setas)")

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset()

done = False
episode_reward = 0
action = 0  # ação padrão

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key in KEY_ACTIONS:
                action = KEY_ACTIONS[event.key]

    # envia a ação para o ambiente e recebe os resultados
    obs, reward, done, truncated, info = env.step(action)
    episode_reward += reward

    print("Observação: ", obs)
    print("Recompensa: ", reward)
    print("Ação: ", action)
    print("-"*40)

    if truncated:
        break

    pygame.time.wait(50)  # reduz velocidade do loop para facilitar controle

print("Recompensa total do episódio: ", episode_reward)