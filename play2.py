"""Interfaz de consola para jugar contra el agente aleatorio."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from go_env import (
    BLACK,
    EMPTY,
    WHITE,
    GoEnv,
    IllegalMoveError,
    PLAYER_NAMES,
    RandomAgent,
    format_move,
    parse_human_move,
)


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


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Juega Go estándar o Atari Go contra RandomAgent."
    )
    parser.add_argument("--size", type=int, default=9, help="Tamaño N del tablero.")
    parser.add_argument(
        "--mode",
        choices=("standard", "atari"),
        help="Variante. Si se omite, se pregunta al comenzar.",
    )
    parser.add_argument(
        "--human-color",
        choices=("black", "white"),
        help="Color humano. Si se omite, se pregunta al comenzar.",
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
    args = parser.parse_args(argv)

    mode = args.mode or _ask_choice(
        "Modo [standard/atari] (standard): ",
        {"standard": "standard", "s": "standard", "atari": "atari", "a": "atari"},
        "standard",
    )
    human_color_name = args.human_color or _ask_choice(
        "Color humano [black/white] (black): ",
        {"black": "black", "b": "black", "white": "white", "w": "white"},
        "black",
    )
    human_player = BLACK if human_color_name == "black" else WHITE

    env = GoEnv(size=args.size, game_mode=mode, komi=args.komi)
    machine = RandomAgent(seed=args.seed)

    if args.position:
        try:
            load_position_file(
                env, args.position, BLACK if args.to_play == "black" else WHITE
            )
        except (OSError, ValueError) as exc:
            print(f"No se pudo cargar la posición: {exc}")
            return
        print(f"Posición cargada desde {args.position}; mueve {PLAYER_NAMES[env.current_player]}.")

    print()
    print(f"Modo: {mode}; tablero: {env.size}x{env.size}.")
    print("Coordenadas: D4 (fila desde abajo, sin columna I) o 3,4")
    print("(fila desde arriba, columna desde la izquierda; ambas 1-based).")
    if mode == "standard":
        print("Escriba 'pass' para pasar.")
    print("Escriba 'quit' para abandonar la interfaz.")
    print()
    print(env.render())
    print()

    while not env.done:
        legal_moves = env.legal_moves()
        current = env.current_player

        if current == human_player:
            while True:
                raw = input(f"Turno de {PLAYER_NAMES[current]}> ").strip()
                if raw.lower() in {"quit", "exit", "salir"}:
                    print("Partida interrumpida.")
                    return
                try:
                    move = parse_human_move(
                        raw,
                        size=env.size,
                        allow_pass=(env.game_mode == "standard"),
                    )
                except ValueError as exc:
                    print(f"Entrada inválida: {exc}")
                    continue

                if move not in legal_moves:
                    print(
                        "Movimiento ilegal: la intersección puede estar ocupada, "
                        "ser suicidio o violar ko."
                    )
                    continue
                break
        else:
            move = machine.select_move(env, legal_moves)
            print(f"La máquina ({PLAYER_NAMES[current]}) está jugando...")

        try:
            result = env.play_move(move)
        except IllegalMoveError as exc:
            # No debería ocurrir después de usar legal_moves, pero deja la CLI
            # protegida ante modificaciones futuras del entorno.
            print(f"No se pudo aplicar la jugada: {exc}")
            continue

        _announce_result(env, result)


if __name__ == "__main__":
    main()
