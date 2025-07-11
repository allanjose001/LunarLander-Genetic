import gymnasium as gym
import pygame

KEY_ACTIONS = {
    pygame.K_UP: 2,      # motor principal
    pygame.K_LEFT: 1,    # motor esquerdo
    pygame.K_RIGHT: 3,   # motor direito
    pygame.K_DOWN: 0     # sem ação
}

def main():
    pygame.init()
    screen = pygame.display.set_mode((400, 100))
    pygame.display.set_caption("Controle LunarLander pelo teclado (setas)")

    env = gym.make("LunarLander-v3", render_mode="human")
    obs, info = env.reset()
    done = False
    episode_reward = 0
    action = 0  # ação padrão

    print("Use as setas do teclado para controlar o LunarLander!")
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key in KEY_ACTIONS:
                    action = KEY_ACTIONS[event.key]
            elif event.type == pygame.KEYUP:
                if event.key in KEY_ACTIONS:
                    action = 0  # desliga motor ao soltar a tecla

        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward

        if truncated:
            break

        pygame.time.wait(50)  # reduz velocidade do loop para facilitar controle

    print("Recompensa total do episódio:", episode_reward)
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()