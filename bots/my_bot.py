from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel


def choose_direction(game: Game) -> tuple[float, float]:
    if game.state.visible_food:
        target = min(
            game.state.visible_food,
            key=lambda food: (food.pos[0] - game.state.me.x) ** 2
            + (food.pos[1] - game.state.me.y) ** 2,
        )
        return (target.pos[0] - game.state.me.x, target.pos[1] - game.state.me.y)
    return (1.0, 0.0)


def main() -> None:
    game = Game()

    while True:
        query = game.get_next_query()
        match query:
            case QueryMovePlayer():
                dx, dy = choose_direction(game)
                game.send_move(
                    MovePlayer(
                        player_id=game.state.me.player_id,
                        direction=DirectionModel(x=dx, y=dy),
                    )
                )
            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")


if __name__ == "__main__":
    main()
