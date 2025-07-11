import numpy as np

def combined_policy(obs, genes):
    """
    Política combinada: função linear com bias + regras condicionais.
    genes[0:8]   -> pesos para observações
    genes[8]     -> bias
    genes[9]     -> limite para velocidade vertical (motor principal)
    genes[10]    -> limite para ângulo positivo (motor esquerdo)
    genes[11]    -> limite para ângulo negativo (motor direito)
    """

    # Função linear com bias
    linear_score = np.dot(obs, genes[:8]) + genes[8]

    # Calcula scores para cada ação (0: nada, 1: motor esquerdo, 2: motor principal, 3: motor direito)
    scores = np.zeros(4)
    scores[0] = 0  # ação 0: nada

    # Ação 1: motor esquerdo (se ângulo > limite)
    scores[1] = linear_score
    if obs[4] < genes[10]:
        scores[1] += abs(obs[4] - genes[10])  # reforça ação se condição for satisfeita

    # Ação 2: motor principal (se vel_y < limite)
    scores[2] = linear_score
    if obs[3] < genes[9]:
        scores[2] += abs(obs[3] - genes[9])

    # Ação 3: motor direito (se ângulo < limite)
    scores[3] = linear_score
    if obs[4] > genes[11]:
        scores[3] += abs(obs[4] - genes[11])

    # Motor esquerdo: aciona se estiver muito à direita
    if obs[0] > genes[12]:  # genes[12] é o limite para acionar motor esquerdo
        scores[1] += abs(obs[0] - genes[12])

    # Motor direito: aciona se estiver muito à esquerda
    if obs[0] < genes[13]:  # genes[13] é o limite para acionar motor direito
        scores[3] += abs(obs[0] - genes[13])

    # Se ambas as pernas tocam o solo e genes[14] > 0, força ação 0 (desligar motores)
    if obs[6] == 1 and obs[7] == 1 and genes[14] > 0:
        return 0

    # Seleciona ação com maior score
    action = int(np.argmax(scores))
    return action