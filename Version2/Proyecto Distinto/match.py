"""Partidas entre dos jugadores cualesquiera (humano o máquina) sobre ``GoEnv``.

Piezas:
- ``play_game``: juega UNA partida completa entre dos agentes y devuelve un
  ``GameRecord``. Sirve igual para humano vs humano, humano vs máquina y
  máquina vs máquina (autojuego).
- ``run_match``: varias partidas entre dos agentes, con estadísticas.
- ``generate_selfplay``: partidas de un agente contra sí mismo guardando los
  estados vistos, para construir datos de entrenamiento.
- ``records_to_arrays``: convierte registros en arreglos NumPy listos para una red.

``make_env`` es una función sin argumentos que devuelve un ``GoEnv`` NUEVO. Ahí
se decide el tablero inicial (vacío, o con ``load_position``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from agents import agent_label
from go_env import BLACK, PLAYER_NAMES, WHITE, AreaScore, GoEnv, IllegalMoveError, MoveResult

# Misma codificación que recomienda el README: 6 planos, canales primero.
DEFAULT_ENCODING: dict[str, Any] = dict(
    channels_last=False, include_to_play=True, include_history=True
)


@dataclass
class GameRecord:
    """Resumen de una partida jugada con ``play_game``.

    ``states[i]``, ``masks[i]`` y ``policies[i]`` (si se registraron) describen lo
    que vio el agente ANTES de elegir ``moves[i]``, jugada de ``players[i]``.
    """

    size: int
    game_mode: str
    moves: list[int]
    players: list[int]
    finished: bool
    winner: Optional[int]
    reason: Optional[str]
    score: Optional[AreaScore]
    captures: dict[int, int]
    states: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    policies: list[Optional[np.ndarray]] = field(default_factory=list)

    @property
    def num_moves(self) -> int:
        return len(self.moves)

    def value_for(self, player: int) -> float:
        """Resultado desde la perspectiva de ``player``: +1 gana, -1 pierde, 0 empate."""
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def training_samples(self) -> list[tuple[np.ndarray, int, int, float]]:
        """Lista de ``(estado, jugada, jugador, valor)``.

        Requiere haber jugado con ``record_states=True``. Una partida cortada por
        ``max_moves`` (``finished=False``) no tiene resultado y devuelve lista vacía.
        """
        if not self.finished:
            return []
        if len(self.states) != len(self.moves):
            raise ValueError("Este registro no guardó estados: use record_states=True.")
        return [
            (s, m, p, self.value_for(p))
            for s, m, p in zip(self.states, self.moves, self.players)
        ]


def play_game(
    env: GoEnv,
    black: Any,
    white: Any,
    *,
    max_moves: Optional[int] = None,
    on_move: Optional[Callable[[GoEnv, MoveResult], None]] = None,
    record_states: bool = False,
    encode_kwargs: Optional[dict[str, Any]] = None,
) -> GameRecord:
    """Juega una partida desde el estado actual de ``env`` hasta que termine.

    - ``black`` / ``white``: agentes (ver agents.py). Pueden ser el mismo objeto.
    - ``max_moves``: tope de jugadas para evitar partidas infinitas con agentes que
      nunca pasan. Por defecto ``10 * N * N``. Si se alcanza, el registro queda con
      ``finished=False`` y ``reason="max_moves"``.
    - ``on_move(env, result)``: se llama tras cada jugada (para imprimir, animar...).
    - ``record_states``: guarda estado codificado, máscara de legales y, si el agente
      expone ``last_policy``, esa política.

    Si el jugador al turno no tiene colocaciones legales (solo posible en Atari Go
    desde una posición cargada) se aplica la regla de GoEnv: pierde.
    La excepción ``GameInterrupted`` de un ``HumanAgent`` se propaga hacia afuera.
    """
    limit = max_moves if max_moves is not None else 10 * env.size * env.size
    enc = dict(DEFAULT_ENCODING if encode_kwargs is None else encode_kwargs)
    agents = {BLACK: black, WHITE: white}

    for agent in {id(a): a for a in (black, white)}.values():
        reset = getattr(agent, "reset", None)
        if callable(reset):
            reset()

    moves: list[int] = []
    players: list[int] = []
    states: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    policies: list[Optional[np.ndarray]] = []
    truncated = False
    stuck_player: Optional[int] = None

    while not env.done:
        if len(moves) >= limit:
            truncated = True
            break

        legal = env.legal_moves()
        player = env.current_player
        if not legal:
            stuck_player = player
            break

        agent = agents[player]
        state = env.encode_state(**enc) if record_states else None
        move = agent.select_move(env, legal)

        try:
            result = env.play_move(move)
        except IllegalMoveError as exc:
            raise IllegalMoveError(
                f"{agent_label(agent)} ({PLAYER_NAMES[player]}) eligió una jugada "
                f"ilegal: {move!r}. {exc}"
            ) from exc

        moves.append(int(move))
        players.append(player)
        if record_states:
            mask = np.zeros(env.action_size, dtype=bool)
            mask[legal] = True
            states.append(state)
            masks.append(mask)
            policy = getattr(agent, "last_policy", None)
            policies.append(None if policy is None else np.array(policy, copy=True))

        if on_move is not None:
            on_move(env, result)

    if stuck_player is not None:
        finished, winner, reason = True, env.opponent(stuck_player), "no_legal_moves"
    elif truncated:
        finished, winner, reason = False, None, "max_moves"
    else:
        finished, winner, reason = True, env.winner, env.terminal_reason

    return GameRecord(
        size=env.size,
        game_mode=env.game_mode,
        moves=moves,
        players=players,
        finished=finished,
        winner=winner,
        reason=reason,
        score=env.final_score,
        captures=dict(env.captures),
        states=states,
        masks=masks,
        policies=policies,
    )


@dataclass(frozen=True)
class MatchStats:
    """Resultados agregados de ``run_match`` (A y B son los agentes recibidos)."""

    games: int
    wins_a: int
    wins_b: int
    draws: int
    unfinished: int
    black_wins: int
    white_wins: int
    mean_moves: float

    @property
    def win_rate_a(self) -> float:
        return self.wins_a / self.games if self.games else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "win_rate_a": self.win_rate_a}


def run_match(
    make_env: Callable[[], GoEnv],
    agent_a: Any,
    agent_b: Any,
    n_games: int,
    *,
    alternate_colors: bool = True,
    max_moves: Optional[int] = None,
) -> MatchStats:
    """Juega ``n_games`` partidas A vs B. Con ``alternate_colors`` A empieza con negro
    en las partidas pares y con blanco en las impares (compensa la ventaja de negro)."""
    wins_a = wins_b = draws = unfinished = black_wins = white_wins = 0
    total_moves = 0

    for i in range(n_games):
        a_is_black = (i % 2 == 0) or not alternate_colors
        black, white = (agent_a, agent_b) if a_is_black else (agent_b, agent_a)
        a_color = BLACK if a_is_black else WHITE

        record = play_game(make_env(), black, white, max_moves=max_moves)
        total_moves += record.num_moves

        if not record.finished:
            unfinished += 1
        elif record.winner is None:
            draws += 1
        else:
            if record.winner == BLACK:
                black_wins += 1
            else:
                white_wins += 1
            if record.winner == a_color:
                wins_a += 1
            else:
                wins_b += 1

    return MatchStats(
        games=n_games,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
        unfinished=unfinished,
        black_wins=black_wins,
        white_wins=white_wins,
        mean_moves=total_moves / n_games if n_games else 0.0,
    )


def generate_selfplay(
    make_env: Callable[[], GoEnv],
    agent: Any,
    n_games: int,
    *,
    max_moves: Optional[int] = None,
    encode_kwargs: Optional[dict[str, Any]] = None,
) -> list[GameRecord]:
    """``n_games`` partidas de ``agent`` contra sí mismo, guardando estados."""
    return [
        play_game(
            make_env(),
            agent,
            agent,
            max_moves=max_moves,
            record_states=True,
            encode_kwargs=encode_kwargs,
        )
        for _ in range(n_games)
    ]


def records_to_arrays(records: list[GameRecord]) -> dict[str, Optional[np.ndarray]]:
    """Apila las muestras de entrenamiento de varias partidas terminadas.

    Devuelve ``states`` (M, C, N, N) float32, ``actions`` (M,), ``players`` (M,),
    ``values`` (M,) float32, ``masks`` (M, A) bool y ``policies`` (M, A) float32
    (None si algún agente no expuso ``last_policy``).
    """
    states, actions, players, values, masks, policies = [], [], [], [], [], []
    for rec in records:
        samples = rec.training_samples()
        for (s, m, p, v), mask, pol in zip(samples, rec.masks, rec.policies):
            states.append(s)
            actions.append(m)
            players.append(p)
            values.append(v)
            masks.append(mask)
            policies.append(pol)

    if not states:
        raise ValueError("No hay muestras: ninguna partida terminada con estados guardados.")

    all_have_policy = all(p is not None for p in policies)
    return {
        "states": np.stack(states).astype(np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "players": np.asarray(players, dtype=np.int8),
        "values": np.asarray(values, dtype=np.float32),
        "masks": np.stack(masks),
        "policies": np.stack(policies).astype(np.float32) if all_have_policy else None,
    }
