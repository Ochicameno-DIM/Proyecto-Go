"""Interfaz de consola: humano vs máquina, humano vs humano o máquina vs máquina."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from agents import (
    GameInterrupted,
    available_agents,
    is_human,
    make_agent,
)
from go_env import (
    BLACK,
    EMPTY,
    WHITE,
    GoEnv,
    PLAYER_NAMES,
    format_move,
)
from match import GameRecord, play_game, run_match


_SYMBOLS = {".": EMPTY, "-": EMPTY, "0": EMPTY,
            "x": BLACK, "1": BLACK,
            "o": WHITE, "2": WHITE}


def load_position_file(env: GoEnv, path: str, to_play: int) -> None:
    """Carga en ``env`` un tablero escrito en un archivo de texto.

    Formato: una línea por fila, de ARRIBA hacia ABAJO (igual que se imprime).
    ``.`` vacío, ``X`` negro, ``O`` blanco. Los espacios se ignoran y las líneas
    que empiezan con ``#`` son comentarios.
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cells = [c for c in line if not c.isspace()]
        try:
            rows.append([_SYMBOLS[c.lower()] for c in cells])
        except KeyError as exc:
            raise ValueError(f"Símbolo no reconocido en la línea {line!r}: {exc}")

    if len(rows) != env.size or any(len(r) != env.size for r in rows):
        raise ValueError(
            f"El archivo debe tener {env.size} filas de {env.size} símbolos "
            f"(encontré {len(rows)} filas)."
        )

    env.load_position(rows, current_player=to_play)

    # load_position no valida alcanzabilidad: evitamos al menos grupos sin libertades.
    for r in range(env.size):
        for c in range(env.size):
            if env.board[r, c] != EMPTY:
                _, libs = env._group_and_liberties(env.board, r, c)
                if not libs:
                    raise ValueError(
                        "Posición inválida: hay un grupo sin libertades "
                        f"(piedra en fila {r + 1}, columna {c + 1} contando desde arriba)."
                    )


def _ask_choice(prompt: str, choices: dict[str, str], default: str) -> str:
    """Solicita una opción hasta recibir una entrada válida."""
    while True:
        raw = input(prompt).strip().lower()
        if not raw:
            return default
        if raw in choices:
            return choices[raw]
        print(f"Opción inválida. Valores permitidos: {', '.join(choices)}.")


def _announce_result(env: GoEnv, result) -> None:
    """Imprime información de una jugada y, si corresponde, el resultado."""
    player_name = PLAYER_NAMES[result.player].capitalize()
    if result.was_pass:
        print(f"{player_name} pasa.")
    else:
        print(f"{player_name} juega {format_move(result.move, env.size)}.")

    if not result.was_pass:
        if result.captured_count:
            print(
                f"Captura: {player_name} retiró {result.captured_count} "
                f"piedra(s)."
            )
        else:
            print("Capturas: ninguna.")

    print()
    print(env.render())
    print()

    if not result.done:
        return

    if result.winner is None:
        print("La partida termina en empate.")
    else:
        print(f"Ganador: {PLAYER_NAMES[result.winner].capitalize()}.")

    if result.reason == "first_capture":
        print("Motivo: primera captura en Atari Go.")
    elif result.reason == "no_legal_moves":
        print("Motivo: el siguiente jugador no tiene movimientos legales.")
    elif result.reason == "double_pass" and result.score is not None:
        score = result.score
        print(
            "Conteo de área: "
            f"negro {score.black_score:.1f}, "
            f"blanco {score.white_score:.1f} "
            f"(komi {score.komi:.1f})."
        )


def _summary_line(record: GameRecord) -> str:
    """Resumen de una línea de una partida terminada o cortada."""
    if not record.finished:
        return f"La partida se cortó por límite de jugadas ({record.num_moves} jugadas)."
    winner = "empate" if record.winner is None else PLAYER_NAMES[record.winner]
    return f"Resultado: {winner}. Motivo: {record.reason}. Jugadas: {record.num_moves}."


def build_parser() -> argparse.ArgumentParser:
    names = available_agents()
    parser = argparse.ArgumentParser(
        description=(
            "Juega Go estándar o Atari Go. Cada color lo puede llevar una persona "
            f"o una máquina ({', '.join(names)})."
        )
    )
    parser.add_argument("--size", type=int, default=9, help="Tamaño N del tablero.")
    parser.add_argument(
        "--mode",
        choices=("standard", "atari"),
        help="Variante. Si se omite, se pregunta al comenzar.",
    )
    parser.add_argument(
        "--black",
        choices=names,
        help="Quién juega con negro (por defecto 'random' si se usa --black/--white).",
    )
    parser.add_argument(
        "--white",
        choices=names,
        help="Quién juega con blanco (por defecto 'random' si se usa --black/--white).",
    )
    parser.add_argument(
        "--human-color",
        choices=("black", "white"),
        help="Atajo: humano con este color y 'random' con el otro. "
        "Si se omite junto con --black/--white, se pregunta al comenzar.",
    )
    parser.add_argument("--komi", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--position",
        help="Archivo de texto con el tablero inicial (X negro, O blanco, . vacío).",
    )
    parser.add_argument(
        "--to-play",
        choices=("black", "white"),
        default="black",
        help="Quién mueve primero desde la posición cargada (por defecto negro).",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Número de partidas (solo cuando ambos jugadores son máquinas).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Segundos de pausa tras cada jugada (para mirar partidas de máquinas).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Máquina vs máquina: no imprime cada jugada, solo el resumen final.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.human_color and (args.black or args.white):
        parser.error("--human-color no se combina con --black/--white.")
    if args.games < 1:
        parser.error("--games debe ser al menos 1.")
    if args.delay < 0:
        parser.error("--delay no puede ser negativo.")

    mode = args.mode or _ask_choice(
        "Modo [standard/atari] (standard): ",
        {"standard": "standard", "s": "standard", "atari": "atari", "a": "atari"},
        "standard",
    )

    if args.black or args.white:
        black_name = args.black or "random"
        white_name = args.white or "random"
    else:
        human_color_name = args.human_color or _ask_choice(
            "Color humano [black/white] (black): ",
            {"black": "black", "b": "black", "white": "white", "w": "white"},
            "black",
        )
        if human_color_name == "black":
            black_name, white_name = "human", "random"
        else:
            black_name, white_name = "random", "human"

    # Semillas consecutivas para las máquinas; así --seed 7 con un humano y una
    # máquina da RandomAgent(7), igual que antes.
    agents = {}
    offset = 0
    for color, name in ((BLACK, black_name), (WHITE, white_name)):
        agent_seed = None if args.seed is None else args.seed + offset
        agents[color] = make_agent(name, agent_seed)
        if not is_human(agents[color]):
            offset += 1
    has_human = any(is_human(a) for a in agents.values())

    if has_human and args.games != 1:
        parser.error("--games solo aplica cuando ambos jugadores son máquinas.")

    to_play = BLACK if args.to_play == "black" else WHITE

    def make_env() -> GoEnv:
        env = GoEnv(size=args.size, game_mode=mode, komi=args.komi)
        if args.position:
            load_position_file(env, args.position, to_play)
        return env

    try:
        env = make_env()
    except (OSError, ValueError) as exc:
        print(f"No se pudo cargar la posición: {exc}")
        return
    if args.position:
        print(
            f"Posición cargada desde {args.position}; "
            f"mueve {PLAYER_NAMES[env.current_player]}."
        )

    print()
    print(f"Modo: {mode}; tablero: {env.size}x{env.size}.")
    print(f"Negro: {black_name}; blanco: {white_name}.")
    if has_human:
        print("Coordenadas: D4 (fila desde abajo, sin columna I) o 3,4")
        print("(fila desde arriba, columna desde la izquierda; ambas 1-based).")
        if mode == "standard":
            print("Escriba 'pass' para pasar.")
        print("Escriba 'quit' para abandonar la interfaz.")
    print()

    # Varias partidas entre máquinas: solo estadísticas.
    if not has_human and args.games > 1:
        stats = run_match(
            make_env, agents[BLACK], agents[WHITE], args.games, alternate_colors=False
        )
        print(
            f"{stats.games} partidas: negro ganó {stats.black_wins}, "
            f"blanco ganó {stats.white_wins}, empates {stats.draws}, "
            f"cortadas {stats.unfinished}; jugadas promedio {stats.mean_moves:.1f}."
        )
        return

    quiet = args.quiet and not has_human
    if not quiet:
        print(env.render())
        print()

    def on_move(env_: GoEnv, result) -> None:
        if not quiet:
            _announce_result(env_, result)
        if args.delay > 0 and not has_human:
            time.sleep(args.delay)

    try:
        record = play_game(env, agents[BLACK], agents[WHITE], on_move=on_move)
    except GameInterrupted:
        print("Partida interrumpida.")
        return

    # La jugada final ya se anunció en on_move salvo en estos casos.
    if quiet or not record.finished or record.num_moves == 0:
        if quiet:
            print(env.render())
            print()
        print(_summary_line(record))


if __name__ == "__main__":
    main()
