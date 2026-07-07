"""
SYNCS Bot Battle 2026 - Agar.io competition bot.

Strategy in one paragraph
--------------------------
The bot is a *potential-field* controller with a small state machine on top.
Every tick it builds a single desired movement vector from weighted, normalised
sub-vectors (flee, hunt, food, wall, virus) and, separately, decides whether to
press "split". Behaviour is driven by the exact engine rules (reverse-engineered
from the public engine):

  * Eating is BLOB-vs-BLOB: blob A eats blob B iff  rA >= 1.2 * rB  AND B's
    centre is inside A's radius. So all threat/prey maths use *individual* blob
    radii, not the aggregated player radius.
  * Bigger blobs are SLOWER (speed = 1.1 / (1 + r*0.08), min 0.25). A lone
    bigger enemy can almost never catch a fleeing smaller blob on open ground -
    the real danger is a SPLIT lunge, so danger zones expand when an enemy is
    big enough to split-kill us.
  * Mass decays ~0.2% per tick, so we must keep eating; past ~radius 4-6 food
    alone can't sustain us and we must hunt players.
  * Viruses pop us into up to 16 pieces if we're big enough to consume one
    (mass > 1.5*1.2 = 1.8). We avoid them when big - and the bigger we are the
    wider the berth we give, since a pop is more catastrophic the more mass we
    lose - and we shelter near them when small and chased (a big chaser can't
    follow without popping).
  * A virus pop leaves an enemy as a tight knot of small blobs: the single most
    profitable target in the game. We actively hunt those knots and split into
    them to sweep up multiple pieces at once.

Only two actions exist in this engine: move (direction) and split. There is no
mass-eject / feed mechanic, so virus-feeding tricks don't apply.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

# --------------------------------------------------------------------------- #
# Engine constants - imported live so we stay in sync if the engine is patched
# mid-competition (they said they might). Fallbacks match the shipped values.
# --------------------------------------------------------------------------- #
try:
    from lib.config.arena import ARENA_SIZE, VIRUS_SIZE, MAX_BLOB_COUNT
    from lib.config.player import (
        EAT_SIZE_RATIO,
        BASE_PLAYER_SPEED,
        PLAYER_SPEED_RADIUS_FACTOR,
        MIN_PLAYER_SPEED,
        SPLIT_MIN_MASS,
    )
except Exception:  # pragma: no cover - defensive fallback
    ARENA_SIZE = 60.0
    VIRUS_SIZE = 1.5
    MAX_BLOB_COUNT = 16
    EAT_SIZE_RATIO = 1.2
    BASE_PLAYER_SPEED = 1.1
    PLAYER_SPEED_RADIUS_FACTOR = 0.08
    MIN_PLAYER_SPEED = 0.25
    SPLIT_MIN_MASS = 2.0

SQRT2 = math.sqrt(2.0)
CENTER = ARENA_SIZE / 2.0
VIRUS_CONSUME_MASS = VIRUS_SIZE * EAT_SIZE_RATIO  # blob mass above which we pop on a virus

# --------------------------------------------------------------------------- #
# Tunables. All the "personality" of the bot lives here - tweak and re-simulate.
# --------------------------------------------------------------------------- #
class Cfg:
    # How many ticks ahead we assume an enemy can walk toward us.
    THREAT_PREDICT_TICKS = 4.0
    # Extra reach we grant a split-capable enemy (the lunge distance).
    SPLIT_LUNGE_REACH = 6.5
    # Safety padding added to every danger zone.
    THREAT_MARGIN = 1.2

    # --- Aggression / prey ------------------------------------------------- #
    # We walk toward prey we're at least this much bigger than (the eat ratio is
    # already baked in). Walking can't catch an optimal fleer (smaller = faster),
    # so this mostly commits us to careless / cornered / distracted targets.
    # Lowered from 1.25 -> hunt more eagerly.
    CHASE_MARGIN = 1.15
    # Minimum mass before we consider a split-kill. Lowered from 2.6 -> we
    # split-kill more readily.
    SPLIT_KILL_MIN_MASS = 2.2
    # Score multiplier applied to a split-killable target (was inline 1.4).
    SPLIT_KILL_SCORE_MULT = 1.6

    # --- Virus-pop / cluster ganking -------------------------------------- #
    # Enemy blobs within this distance of each other count as one cluster.
    CLUSTER_RADIUS = 4.0
    # This many eatable enemy blobs bunched together == a "popped player" signal.
    CLUSTER_MIN_BLOBS = 3
    # If the knot sits this close to a virus, it's almost certainly a fresh pop.
    CLUSTER_VIRUS_DIST = 5.0
    # Score multiplier for a cluster target - this is the juiciest prey around.
    W_CLUSTER_BONUS = 2.2
    # Extra multiplier when the cluster is confirmed next to a virus.
    W_CLUSTER_VIRUS_BONUS = 1.3

    # Distance we keep clear of walls, added on top of our radius.
    WALL_MARGIN = 3.5

    # --- Virus avoidance (scales with our size) --------------------------- #
    # Base clearance we keep from a virus (when big enough to pop), on top of radii.
    VIRUS_MARGIN = 1.5
    # Extra clearance per unit of our radius: a pop is more catastrophic the
    # bigger we are, so large blobs peel away from viruses much earlier.
    VIRUS_MARGIN_PER_RADIUS = 0.5
    # Mass above which we treat viruses as a serious hazard (not a mild nuisance).
    VIRUS_BIG_MASS = 4.0

    # Combination weights (applied to already-normalised sub-vectors).
    W_PREY = 1.7            # was 1.35 - hunt harder (weight when small)
    W_FOOD = 1.0            # food weight when small
    # Diet shift: the bigger we are, the more we favour hunting over farming,
    # since food can't sustain a large blob against mass decay. We lerp the prey
    # and food weights from their "small" values toward these as mass grows.
    W_PREY_BIG = 3.2        # prey weight once we're fully "grown"
    W_FOOD_BIG = 0.2        # food weight once we're fully "grown"
    DIET_SHIFT_START_MASS = 2.5   # at/below this, weights stay at the small values
    DIET_SHIFT_FULL_MASS = 10.0   # at/above this, weights reach the *_BIG values
    W_WALL = 0.55
    W_VIRUS = 0.9           # normal virus-avoidance weight
    W_VIRUS_BIG = 2.0       # virus-avoidance weight once we're "large"
    # While fleeing:
    FLEE_WALL = 0.45
    FLEE_VIRUS = 0.55
    FLEE_VIRUS_BIG = 1.1    # flee viruses harder when large
    FLEE_SHELTER = 0.7
    FLEE_FOOD = 0.15

    # Endgame: in the final stretch, if we're comfortably ahead we play safer.
    ENDGAME_ROUNDS = 200


# --------------------------------------------------------------------------- #
# Small vector helpers
# --------------------------------------------------------------------------- #
def _norm(vx: float, vy: float) -> tuple[float, float]:
    m = math.hypot(vx, vy)
    if m < 1e-12:
        return (0.0, 0.0)
    return (vx / m, vy / m)


def _speed(radius: float) -> float:
    return max(MIN_PLAYER_SPEED, BASE_PLAYER_SPEED / (1.0 + radius * PLAYER_SPEED_RADIUS_FACTOR))


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


# --------------------------------------------------------------------------- #
# The brain
# --------------------------------------------------------------------------- #
class Bot:
    def __init__(self) -> None:
        self._last_dir: tuple[float, float] = (1.0, 0.0)
        self._rng = random.Random(1234)

    # -- utility describing one of our own blobs ---------------------------- #
    @staticmethod
    def _blob_xy(b) -> tuple[float, float]:
        return (b.pos[0], b.pos[1])

    def decide(self, state) -> tuple[float, float, bool]:
        me = state.me
        my_blobs = list(me.blobs.values())
        if not my_blobs:
            # Dead / no info - harmless default (also can't be submitted when dead).
            return (*self._last_dir, False)

        primary = max(my_blobs, key=lambda b: b.radius)
        px, py = self._blob_xy(primary)
        primary_mass = primary.radius * primary.radius

        enemies = list(state.visible_blobs)
        viruses = list(state.visible_viruses)
        foods = list(state.visible_food)

        # ---------------- THREATS ----------------------------------------- #
        # Sum world-space repulsion over every (my blob, dangerous enemy) pair.
        threat_x = threat_y = 0.0
        threat_level = 0.0
        for b in my_blobs:
            bx, by = self._blob_xy(b)
            br = b.radius
            bmass = br * br
            for e in enemies:
                er = e.radius
                if er < br * EAT_SIZE_RATIO:
                    continue  # this enemy blob can't eat this blob of mine
                ex, ey = e.pos[0], e.pos[1]
                d = _dist(bx, by, ex, ey)
                split_capable = (er * er >= SPLIT_MIN_MASS) and ((er / SQRT2) >= br * EAT_SIZE_RATIO)
                reach = er + _speed(er) * Cfg.THREAT_PREDICT_TICKS + Cfg.THREAT_MARGIN
                if split_capable:
                    reach += Cfg.SPLIT_LUNGE_REACH
                if d >= reach:
                    continue
                sev = (reach - d) / reach  # 0..1, closer = scarier
                if split_capable:
                    sev *= 1.5
                w = sev * bmass  # protect our heavier blobs preferentially
                ux, uy = _norm(bx - ex, by - ey)  # push away from enemy
                threat_x += ux * w
                threat_y += uy * w
                threat_level = max(threat_level, sev)

        threatened = (threat_x * threat_x + threat_y * threat_y) > 1e-9

        # ---------------- WALL avoidance ---------------------------------- #
        wx = wy = 0.0
        margin = primary.radius + Cfg.WALL_MARGIN
        if px < margin:
            wx += (margin - px) / margin
        if px > ARENA_SIZE - margin:
            wx -= (px - (ARENA_SIZE - margin)) / margin
        if py < margin:
            wy += (margin - py) / margin
        if py > ARENA_SIZE - margin:
            wy -= (py - (ARENA_SIZE - margin)) / margin

        # ---------------- VIRUS handling ---------------------------------- #
        # Avoid viruses for any of our blobs big enough to pop on them. The
        # clearance grows with blob size: the bigger we are, the more a pop
        # costs us, so we start peeling away from viruses much earlier.
        vx = vy = 0.0
        for b in my_blobs:
            if b.radius * b.radius <= VIRUS_CONSUME_MASS:
                continue
            bx, by = self._blob_xy(b)
            keep_base = b.radius + Cfg.VIRUS_MARGIN + Cfg.VIRUS_MARGIN_PER_RADIUS * b.radius
            for v in viruses:
                keep = keep_base + v.radius
                d = _dist(bx, by, v.pos[0], v.pos[1])
                if d < keep:
                    ux, uy = _norm(bx - v.pos[0], by - v.pos[1])
                    s = (keep - d) / keep
                    vx += ux * s
                    vy += uy * s

        # Shelter: when small enough to be virus-safe and being chased, run to a
        # nearby virus - big chasers can't follow without popping.
        sh_x = sh_y = 0.0
        if threatened and primary_mass <= VIRUS_CONSUME_MASS and viruses:
            nearest_v = min(viruses, key=lambda v: _dist(px, py, v.pos[0], v.pos[1]))
            sh_x, sh_y = _norm(nearest_v.pos[0] - px, nearest_v.pos[1] - py)

        # ---------------- FOOD (cluster seeking) -------------------------- #
        fx = fy = 0.0
        for f in foods:
            dx = f.pos[0] - px
            dy = f.pos[1] - py
            w = 1.0 / (dx * dx + dy * dy + 1.0)  # strongly favour nearby food
            fx += dx * w
            fy += dy * w

        # ---------------- PREY selection ---------------------------------- #
        # All enemy blobs our primary can outright eat.
        eatable = [e for e in enemies if primary.radius >= e.radius * EAT_SIZE_RATIO]

        best_prey = None
        best_aim: Optional[tuple[float, float]] = None
        best_score = 0.0
        prey_split = False

        for e in eatable:
            er = e.radius
            ex, ey = e.pos[0], e.pos[1]
            d = _dist(px, py, ex, ey)

            split_kill = (
                primary_mass >= Cfg.SPLIT_KILL_MIN_MASS
                and (primary.radius / SQRT2) >= er * EAT_SIZE_RATIO
                and d <= (primary.radius + Cfg.SPLIT_LUNGE_REACH)
                and len(my_blobs) < MAX_BLOB_COUNT
            )
            walkable = primary.radius >= er * EAT_SIZE_RATIO * Cfg.CHASE_MARGIN
            if not (split_kill or walkable):
                continue

            # Baseline: prefer high mass, nearby, and split-killable prey.
            score = (er * er) / (d + 1.0)
            if split_kill:
                score *= Cfg.SPLIT_KILL_SCORE_MULT

            aim = (ex, ey)

            # Virus-pop detection: a tight knot of small enemy blobs around e.
            # We value the *aggregate* mass of the whole knot and aim at its
            # centroid so a split lunge sweeps up several pieces at once.
            cluster = [o for o in eatable
                       if _dist(ex, ey, o.pos[0], o.pos[1]) <= Cfg.CLUSTER_RADIUS]
            if len(cluster) >= Cfg.CLUSTER_MIN_BLOBS:
                knot_mass = sum(o.radius * o.radius for o in cluster)
                score = (knot_mass / (d + 1.0)) * Cfg.W_CLUSTER_BONUS
                if split_kill:
                    score *= Cfg.SPLIT_KILL_SCORE_MULT
                if viruses and min(_dist(ex, ey, v.pos[0], v.pos[1])
                                   for v in viruses) <= Cfg.CLUSTER_VIRUS_DIST:
                    score *= Cfg.W_CLUSTER_VIRUS_BONUS  # confirmed fresh pop
                aim = (sum(o.pos[0] for o in cluster) / len(cluster),
                       sum(o.pos[1] for o in cluster) / len(cluster))

            if score > best_score:
                best_score = score
                best_prey = e
                best_aim = aim
                prey_split = split_kill

        prey_x = prey_y = 0.0
        if best_prey is not None and best_aim is not None:
            prey_x, prey_y = _norm(best_aim[0] - px, best_aim[1] - py)

        # Never split when a real threat is present (splitting = vulnerable).
        # A knot of small popped blobs never trips `threatened`, so we're still
        # free to gank it - unless a genuine bigger enemy is also lurking.
        do_split = bool(best_prey is not None and prey_split and not threatened)

        # Endgame caution: if we're clearly the biggest and time's almost up,
        # stop taking split risks and just farm/survive.
        rounds_left = max(0, state.max_rounds - state.round)
        if rounds_left <= Cfg.ENDGAME_ROUNDS and enemies:
            biggest_enemy = max((e.radius for e in enemies), default=0.0)
            if me.radius > biggest_enemy * 1.25:
                do_split = False

        # ---------------- COMBINE ----------------------------------------- #
        big = primary_mass >= Cfg.VIRUS_BIG_MASS
        w_virus = Cfg.W_VIRUS_BIG if big else Cfg.W_VIRUS
        flee_virus = Cfg.FLEE_VIRUS_BIG if big else Cfg.FLEE_VIRUS

        # Diet shift: interpolate prey/food weights by our mass so a big blob
        # chases players and a small one farms food.
        span = Cfg.DIET_SHIFT_FULL_MASS - Cfg.DIET_SHIFT_START_MASS
        t = (primary_mass - Cfg.DIET_SHIFT_START_MASS) / span if span > 1e-9 else 1.0
        t = max(0.0, min(1.0, t))
        w_prey = Cfg.W_PREY + (Cfg.W_PREY_BIG - Cfg.W_PREY) * t
        w_food = Cfg.W_FOOD + (Cfg.W_FOOD_BIG - Cfg.W_FOOD) * t

        if do_split and best_prey is not None:
            # Aim the lunge straight at the prey (or knot centroid).
            dirx, diry = prey_x, prey_y
        elif threatened:
            fxn, fyn = _norm(threat_x, threat_y)
            dirx = fxn + Cfg.FLEE_WALL * _norm(wx, wy)[0] + flee_virus * _norm(vx, vy)[0] \
                + Cfg.FLEE_SHELTER * sh_x + Cfg.FLEE_FOOD * _norm(fx, fy)[0]
            diry = fyn + Cfg.FLEE_WALL * _norm(wx, wy)[1] + flee_virus * _norm(vx, vy)[1] \
                + Cfg.FLEE_SHELTER * sh_y + Cfg.FLEE_FOOD * _norm(fx, fy)[1]
        else:
            pxn, pyn = _norm(prey_x, prey_y)
            fxn, fyn = _norm(fx, fy)
            wxn, wyn = _norm(wx, wy)
            vxn, vyn = _norm(vx, vy)
            dirx = w_prey * pxn + w_food * fxn + Cfg.W_WALL * wxn + w_virus * vxn
            diry = w_prey * pyn + w_food * fyn + Cfg.W_WALL * wyn + w_virus * vyn

        # ---------------- EXPLORE fallback -------------------------------- #
        if (dirx * dirx + diry * diry) < 1e-9:
            # Nothing interesting in view: drift toward centre with smooth jitter
            # so we don't get stuck against a wall or oscillate.
            cx, cy = _norm(CENTER - px, CENTER - py)
            jitter = self._rng.uniform(-0.6, 0.6)
            ca, sa = math.cos(jitter), math.sin(jitter)
            lx, ly = self._last_dir
            base_x = 0.6 * lx + 0.4 * cx
            base_y = 0.6 * ly + 0.4 * cy
            dirx = base_x * ca - base_y * sa
            diry = base_x * sa + base_y * ca

        dirx, diry = _norm(dirx, diry)
        if dirx == 0.0 and diry == 0.0:
            dirx, diry = self._last_dir
        self._last_dir = (dirx, diry)
        return (dirx, diry, do_split)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main() -> None:
    game = Game()
    bot = Bot()
    while True:
        query = game.get_next_query()
        match query:
            case QueryMovePlayer():
                try:
                    dx, dy, split = bot.decide(game.state)
                except Exception:
                    # Never crash / never time out -> never get banned.
                    dx, dy, split = 1.0, 0.0, False
                if dx == 0.0 and dy == 0.0:
                    dx = 1.0
                game.send_move(
                    MovePlayer(
                        player_id=game.state.me.player_id,
                        direction=DirectionModel(x=dx, y=dy),
                        split=split,
                    )
                )
            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")


if __name__ == "__main__":
    main()