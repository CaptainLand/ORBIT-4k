from __future__ import annotations

import numpy as np

EMPTY, TAP, LN_START, LN_END = 0, 1, 2, 3


def validate_and_repair(chart: np.ndarray) -> tuple[np.ndarray, list[str]]:
    result = chart.astype(np.uint8, copy=True)
    active = [False] * 4
    repairs: list[str] = []
    for tick in range(len(result)):
        for lane in range(4):
            state = int(result[tick, lane])
            if state == LN_START:
                if active[lane]:
                    result[tick, lane] = TAP
                    repairs.append(f"tick {tick} lane {lane}: nested LN_START -> TAP")
                else:
                    active[lane] = True
            elif state == LN_END:
                if not active[lane]:
                    result[tick, lane] = EMPTY
                    repairs.append(f"tick {tick} lane {lane}: orphan LN_END -> EMPTY")
                else:
                    active[lane] = False
            elif state == TAP and active[lane]:
                result[tick, lane] = EMPTY
                repairs.append(f"tick {tick} lane {lane}: TAP inside active LN -> EMPTY")
    for lane, is_active in enumerate(active):
        if is_active:
            end_tick = len(result) - 1
            result[end_tick, lane] = LN_END
            repairs.append(f"lane {lane}: closed unterminated LN at final tick")
    return result, repairs
