"""Jugadores para ``GoEnv``: contrato común, agente humano y registro por nombre.

Contrato mínimo: cualquier objeto con

    select_move(env, legal_moves) -> int

sirve como jugador. ``RandomAgent`` (en go_env.py) ya lo cumple, así que un agente
nuevo (MCTS, red neuronal, etc.) solo necesita ese método. Atributos opcionales que
``match.play_game`` sabe aprovechar si existen:

- ``name``: texto para mostrar en mensajes de error o en la CLI.
- ``is_human``: True si la decisión viene de una persona (la CLI lo usa para
  decidir si imprime ayuda de coordenadas y si permite varias partidas).
- ``reset()``: se llama al comenzar cada partida (útil para limpiar un árbol MCTS).
- ``last_policy``: vector de probabilidades de la última decisión; si
  ``record_states=True`` queda guardado en el ``GameRecord`` (objetivo de política
  para entrenar tipo AlphaZero).

Para añadir un agente propio y poder elegirlo con ``--black``/``--white`` en la CLI:

    register_agent("mcts", lambda seed: MCTSAgent(seed=seed))
"""
from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from go_env import PLAYER_NAMES, GoEnv, RandomAgent, parse_human_move


class GameInterrupted(Exception):
    """El jugador humano pidió abandonar la partida."""


@runtime_checkable
class Agent(Protocol):
    """Contrato mínimo de un jugador."""

    def select_move(self, env: GoEnv, legal_moves: Sequence[int]) -> int:
        ...


class HumanAgent:
    """Pide la jugada por consola (formato ``D4`` o ``3,4``; ``pass``; ``quit``).

    ``input_fn`` y ``output_fn`` se pueden inyectar para pruebas automáticas.
    Si ``input_fn`` es None se usa ``input`` al momento de preguntar.
    """

    name = "humano"
    is_human = True

    def __init__(self, input_fn: Optional[Callable[[str], str]] = None, output_fn=print):
        self._input_fn = input_fn
        self._output = output_fn

    def select_move(self, env: GoEnv, legal_moves: Sequence[int]) -> int:
        ask = self._input_fn or input
        prompt = f"Turno de {PLAYER_NAMES[env.current_player]}> "
        while True:
            raw = ask(prompt).strip()
            if raw.lower() in {"quit", "exit", "salir"}:
                raise GameInterrupted()
            try:
                move = parse_human_move(
                    raw,
                    size=env.size,
                    allow_pass=(env.game_mode == "standard"),
                )
            except ValueError as exc:
                self._output(f"Entrada inválida: {exc}")
                continue

            if move not in legal_moves:
                self._output(
                    "Movimiento ilegal: la intersección puede estar ocupada, "
                    "ser suicidio o violar ko."
                )
                continue
            return move


# --------------------------------------------------------------------------
# Registro por nombre
# --------------------------------------------------------------------------
AgentFactory = Callable[[Optional[int]], Agent]
_FACTORIES: dict[str, AgentFactory] = {}


def register_agent(name: str, factory: AgentFactory) -> None:
    """Registra una fábrica ``factory(seed) -> Agent`` bajo ``name``."""
    _FACTORIES[name.strip().lower()] = factory


def available_agents() -> list[str]:
    """Nombres registrados (los que acepta la CLI)."""
    return sorted(_FACTORIES)


def make_agent(name: str, seed: Optional[int] = None) -> Agent:
    """Crea un agente a partir de su nombre registrado."""
    key = name.strip().lower()
    try:
        factory = _FACTORIES[key]
    except KeyError:
        raise ValueError(
            f"Agente desconocido {name!r}. Disponibles: {', '.join(available_agents())}."
        ) from None
    return factory(seed)


def agent_label(agent: object) -> str:
    """Nombre legible del agente."""
    return str(getattr(agent, "name", type(agent).__name__))


def is_human(agent: object) -> bool:
    """True si el agente representa a una persona."""
    return bool(getattr(agent, "is_human", False))


register_agent("random", lambda seed: RandomAgent(seed))
register_agent("human", lambda seed: HumanAgent())
