"""Circuit breaker: max tool rounds + same-tool loop detection.

**Unstable** internals; session behavior is part of the stable ``AgentRouter``
contract (``session_id`` / ``reset_session`` / ``session_snapshot``). See ``API.md``.

Session semantics
-----------------
* **With ``session_id``**: history and trip state persist across ``route()``
  calls on the same router instance. Use one id per user conversation / agent
  run. Call ``reset(session_id)`` when the conversation ends.
* **Without ``session_id``**: only the ``history`` argument of the *current*
  call is checked (stateless). Safe for one-shot API / eval -calls do not
  share a fake ``_default`` session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from laya_thalamus.config import CircuitBreakerConfig


@dataclass
class SessionState:
    session_id: str
    history: list[str] = field(default_factory=list)
    tripped: bool = False
    trip_reason: str | None = None
    route_count: int = 0
    last_decision_id: str | None = None


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self.config = config or CircuitBreakerConfig()
        self._sessions: dict[str, SessionState] = {}

    def get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def check(
        self,
        session_id: str | None,
        history: list[str] | None = None,
    ) -> tuple[bool, str | None]:
        """Return (ok, reason). ok=False means terminate."""
        if session_id:
            state = self.get_session(session_id)
            hist = list(history) if history is not None else list(state.history)
            if state.tripped:
                return False, state.trip_reason or "circuit already tripped"
        else:
            state = None
            hist = list(history) if history is not None else []

        if len(hist) >= self.config.max_tool_rounds:
            reason = f"max tool rounds reached ({self.config.max_tool_rounds})"
            if state is not None:
                state.tripped = True
                state.trip_reason = reason
            return False, reason

        n = self.config.max_same_tool_repeats
        if n > 0 and len(hist) >= n:
            tail = hist[-n:]
            if len(set(tail)) == 1 and tail[0] not in (None, "none"):
                reason = f"same tool '{tail[0]}' repeated {n} times"
                if state is not None:
                    state.tripped = True
                    state.trip_reason = reason
                return False, reason

        return True, None

    def record(self, session_id: str | None, tool_name: str) -> None:
        if not session_id:
            return
        self.get_session(session_id).history.append(tool_name)

    def note_decision(self, session_id: str | None, decision_id: str) -> None:
        if not session_id:
            return
        state = self.get_session(session_id)
        state.route_count += 1
        state.last_decision_id = decision_id

    def reset(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)

    def sync_history(self, session_id: str | None, history: list[str]) -> None:
        if not session_id:
            return
        state = self.get_session(session_id)
        state.history = list(history)
        state.tripped = False
        state.trip_reason = None

    def snapshot(self, session_id: str) -> dict:
        state = self.get_session(session_id)
        return {
            "session_id": state.session_id,
            "history": list(state.history),
            "tripped": state.tripped,
            "trip_reason": state.trip_reason,
            "route_count": state.route_count,
            "last_decision_id": state.last_decision_id,
        }
