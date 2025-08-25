# wsafety/alert.py
# Prints a terminal alert when the male-to-female ratio is >= threshold.

import time
from typing import Optional, Tuple


class RatioAlert:
    def __init__(
        self,
        threshold: float = 3.0,
        cooldown_seconds: float = 10.0,
        require_female: bool = True,
    ):
        """
        threshold: trigger when (male_count / female_count) >= threshold
        cooldown_seconds: minimum time between repeated alerts while condition persists
        require_female: if True, only compute ratio when female_count >= 1.
                        if False, treat F=0 as infinite ratio (will alert if male_count > 0).
        """
        self.threshold = float(threshold)
        self.cooldown_seconds = float(cooldown_seconds)
        self.require_female = bool(require_female)

        self._last_alert_ts: float = 0.0
        self._above: bool = False  # was condition true last frame?

    def _now(self) -> float:
        return time.monotonic()

    def _should_print(self) -> bool:
        return (self._now() - self._last_alert_ts) >= self.cooldown_seconds

    def update(self, male_count: int, female_count: int) -> Tuple[bool, Optional[str]]:
        """
        Call this once per frame with current counts.
        Returns (triggered, message). If triggered is True, it also prints the message.
        """
        # Compute ratio with guardrails
        ratio: Optional[float] = None
        note = ""
        if female_count > 0:
            ratio = male_count / float(female_count)
        elif not self.require_female and male_count > 0:
            ratio = float("inf")
            note = " (F=0 → ratio treated as ∞)"

        # Determine current state
        current_above = (ratio is not None) and (ratio >= self.threshold)

        triggered = False
        message = None

        # Rising edge or cooldown-based repeat
        if current_above and (not self._above or self._should_print()):
            self._last_alert_ts = self._now()
            triggered = True
            if ratio == float("inf"):
                ratio_str = "∞"
            elif ratio is None:
                ratio_str = "N/A"
            else:
                ratio_str = f"{ratio:.2f}"

            message = (
                f"ALERT: High M/F ratio (≥ {self.threshold:.2f}){note} | "
                f"M={male_count}, F={female_count}, ratio={ratio_str}"
            )
            print(message)

        # Update state for next call
        self._above = current_above
        return triggered, message