# Go RL Baseline

Base en Python/NumPy para investigar aprendizaje por refuerzo en Go sobre hardware convencional. Incluye reglas de Go estándar, Atari Go, un agente aleatorio, una interfaz de consola y pruebas unitarias.

## Archivos

- `go_env.py`: reglas, estado, codificación para ML y `RandomAgent`.
- `play.py`: humano contra agente aleatorio.
- `test_go_env.py`: pruebas de captura, suicidio, ko, Atari Go y utilidades.

## Instalación y ejecución

```bash
python -m pip install numpy
python -m unittest -v test_go_env.py
python play.py --size 9 --mode atari --human-color black --seed 7
python play.py --size 9 --mode standard --human-color white --komi 7.5
```

## API mínima

```python
from go_env import GoEnv, RandomAgent

env = GoEnv(size=9, game_mode="atari")
agent = RandomAgent(seed=0)

while not env.done:
    legal = env.legal_moves()
    move = agent.select_move(env, legal)
    observation, reward, done, info = env.step(move)

print(env.winner, info)
```

Las acciones de tablero son `fila * N + columna`. El pass es `N*N` y solo existe en modo estándar. `action_mask()` devuelve una máscara directamente utilizable para anular logits o Q-values de acciones ilegales.

## Representación para redes

`encode_state()` devuelve `(N, N, 3)` con planos negro, blanco y vacío. Para entrenamiento real conviene usar:

```python
x = env.encode_state(
    channels_last=False,
    include_to_play=True,
    include_history=True,
)
```

Esto produce seis planos: negro actual, blanco actual, vacío, jugador al turno, negro anterior y blanco anterior. Los dos planos históricos permiten representar la información del ko simple; sin historial, el tablero por sí solo no es completamente markoviano.

## Integración con Gym/Gymnasium

La clase ya expone `reset()`, `step()`, `action_size`, `legal_moves()` y `action_mask()`. Un adaptador puede declarar un espacio `Discrete(env.action_size)` y un espacio `Box(0, 1, shape=(N, N, C))`. En un entorno de dos jugadores hay que fijar una convención: controlar un solo color y usar al rival dentro del wrapper, o producir transiciones de autojuego con el color del actor asociado a cada estado.

Esquema de un wrapper opcional:

```python
# Requiere gymnasium, no forma parte de las dependencias del núcleo.
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GymGoWrapper(gym.Env):
    def __init__(self, size=9, game_mode="standard"):
        self.game = GoEnv(size=size, game_mode=game_mode)
        channels = 6
        self.action_space = spaces.Discrete(self.game.action_size)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(channels, size, size),
            dtype=np.float32,
        )

    def _obs(self):
        return self.game.encode_state(
            channels_last=False,
            include_to_play=True,
            include_history=True,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._obs(), {"action_mask": self.game.action_mask()}

    def step(self, action):
        _, reward, terminated, info = self.game.step(action)
        info["action_mask"] = self.game.action_mask()
        return self._obs(), reward, terminated, False, info
```

## Extensión hacia un agente de aprendizaje

Para DQN, una CNN puede producir `action_size` Q-values; antes de elegir la acción se asigna `-inf` a acciones donde `action_mask()` sea falsa. DQN es posible, pero Go es adversarial, tiene recompensas escasas y un espacio combinatorio grande.

Para un AlphaZero simplificado, la red tendría dos cabezas: una política de `action_size` logits y un valor escalar en `[-1, 1]`. Cada nodo de MCTS usa `env.copy()` para simular acciones. Durante autojuego se guardan `(estado, política_MCTS, jugador_al_turno)` y, al terminar, se convierte `env.winner` en un objetivo de valor desde la perspectiva del jugador guardado.

## Decisiones y límites

- Los grupos y libertades se calculan por búsqueda en profundidad sobre vecinos ortogonales. Es deliberadamente simple y verificable; no está optimizado con cadenas incrementales o Zobrist hashing.
- El ko implementado es ko simple, exactamente una repetición inmediata. No hay superko.
- El conteo usa área determinista: piedras más regiones vacías rodeadas por un solo color, con komi configurable. No existe una fase humana para marcar piedras muertas; en autojuego deben capturarse antes de los dos pases.
- La generación legal simula cada intersección. Es suficiente como referencia en 9x9 y funcional en 19x19, aunque MCTS intensivo requerirá cachés, hashing y actualizaciones incrementales.
