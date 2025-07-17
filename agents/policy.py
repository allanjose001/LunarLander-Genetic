import numpy as np

def combined_policy(obs, genes):
    """
    Política combinada: função linear com bias + regras condicionais.
    genes[0:8]   -> pesos para observações
    genes[8]     -> bias
    genes[9]     -> limite para velocidade vertical (motor principal)
    genes[10]    -> limite para ângulo positivo (motor esquerdo)
    genes[11]    -> limite para ângulo negativo (motor direito)
    genes[12]    -> limite para acionar motor esquerdo (posição x)
    genes[13]    -> limite para acionar motor direito (posição x)
    genes[14]    -> força ação 0 se ambas as pernas estiverem no solo
    genes[15]    -> limite de suavidade do pouso (velocidade vertical máxima para desaceleração)
    genes[16]    -> limite de altitude para iniciar desaceleração vertical
    genes[17]    -> tolerância para considerar centralizado antes do pouso
    """
    linear_score = np.dot(obs, genes[:8]) + genes[8]

    scores = np.zeros(4)
    scores[0] = 0
    scores[1] = linear_score
    scores[2] = linear_score
    scores[3] = linear_score

    # Ação 1: motor esquerdo (se ângulo > limite)
    if obs[4] < genes[10]:
        scores[1] += abs(obs[4] - genes[10])

    # Ação 2: motor principal (se vel_y < limite)
    if obs[3] < genes[9]:
        scores[2] += abs(obs[3] - genes[9])

    # Reforça motor principal se vel_y < genes[15] e altitude baixa (próximo do solo)
    altitude_threshold = genes[16]
    if obs[1] < altitude_threshold and obs[3] < genes[15]:
        scores[2] += abs(obs[3] - genes[15])

    # Ação 3: motor direito (se ângulo < limite)
    if obs[4] > genes[11]:
        scores[3] += abs(obs[4] - genes[11])

    # Motor esquerdo: aciona se estiver muito à direita
    if obs[0] > genes[12]:  # genes[12] é o limite para acionar motor esquerdo
        scores[1] += abs(obs[0] - genes[12])

    # Motor direito: aciona se estiver muito à esquerda
    if obs[0] < genes[13]:  # genes[13] é o limite para acionar motor direito
        scores[3] += abs(obs[0] - genes[13])

    center_tolerance = genes[17]
    
    # Se ambas as pernas tocam o solo, está centralizado e genes[14] > 0, força ação 0 (desligar motores)
    if obs[6] == 1 and obs[7] == 1 and abs(obs[0]) <= center_tolerance and genes[14] > 0:
        return 0

    # Seleciona ação com maior score
    action = int(np.argmax(scores))
    
    """
    # Se include_aux_reward for True, retorna também a recompensa auxiliar para pouso suave
    if include_aux_reward:
        soft_landing_factor = genes[15]
        aux_reward = soft_landing_factor * (1 - abs(obs[3])) if obs[6] == 1 and obs[7] == 1 else 0.0
        return action, aux_reward
    """
    return action
