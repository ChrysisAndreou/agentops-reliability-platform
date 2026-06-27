"""
Simulated memory agent — deterministic, configurable agent for memory evaluation.

Produces realistic MemoryRecallResult objects following a configurable
MemoryProfile without requiring LLM API keys. Suitable for CI/CD testing,
benchmark development, and evaluation methodology demonstration.

For production use, swap in a real agent with memory capabilities.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from .state import (
    PRODUCTION_MEMORY,
    MemoryContext,
    MemoryEntry,
    MemoryProfile,
    MemoryRecallResult,
    MemoryStore,
    MemoryType,
)


class SimulatedMemoryAgent:
    """A simulated agent with configurable memory behaviour for evaluation.

    Maintains a MemoryStore across turns and produces recall results
    according to a MemoryProfile. Deterministic given the same (task_id,
    profile, seed) combination.

    Usage:
        store = MemoryStore()
        agent = SimulatedMemoryAgent(profile=PRODUCTION_MEMORY, store=store)

        # Inject memories over turns
        store.inject(MemoryEntry(id="m1", content="User prefers dark mode",
                     memory_type=MemoryType.SEMANTIC, injected_at_turn=2))

        # Test recall at a later turn
        result = agent.recall("m1", turn=8, task_id="test-1")
        print(f"Correct: {result.correct}, Confidence: {result.confidence}")
    """

    def __init__(
        self,
        profile: MemoryProfile = PRODUCTION_MEMORY,
        store: MemoryStore | None = None,
        seed: int | None = None,
    ):
        self.profile = profile
        self.store = store or MemoryStore()
        self._seed = seed
        self._contexts: list[MemoryContext] = []

    def recall(
        self,
        target_id: str,
        turn: int,
        task_id: str = "default",
    ) -> MemoryRecallResult:
        """Test recall of a specific memory at a given conversation turn."""
        rng = random.Random(self._make_seed(target_id, turn, task_id))

        target = self.store.get_by_id(target_id)
        if target is None:
            return MemoryRecallResult(
                task_id=task_id,
                turn=turn,
                target_memory_id=target_id,
                memory_type=MemoryType.SEMANTIC,
                target_content="",
                recalled_content="",
                correct=False,
                partial=False,
                confidence=0.0,
                latency_ms=0,
                error="Memory not found in store",
            )

        # Calculate recall probability with decay
        turns_since_injection = turn - target.injected_at_turn
        decay = min(
            1.0,
            self.profile.decay_per_turn * turns_since_injection,
        )
        importance_bonus = target.importance * self.profile.importance_weight
        recall_prob = max(
            self.profile.min_recall_prob,
            self.profile.base_recall_prob - decay + importance_bonus,
        )

        # Determine recall outcome
        roll = rng.random()
        is_confused = rng.random() < self.profile.confusion_rate
        is_hallucinated = rng.random() < self.profile.hallucination_rate

        if roll < recall_prob and not is_confused:
            # Correct recall
            recalled = target.content
            correct = True
            partial = False
            confidence = 0.85 + rng.uniform(0, 0.15)
        elif roll < recall_prob and is_confused:
            # Confused: partial recall with wrong details
            words = target.content.split()
            if len(words) > 2:
                rng.shuffle(words)
            recalled = " ".join(words)
            correct = False
            partial = True
            confidence = 0.5 + rng.uniform(0, 0.3)
        elif is_hallucinated:
            # Fabricated memory
            recalled = self._generate_hallucination(target, rng)
            correct = False
            partial = False
            confidence = 0.6 + rng.uniform(0, 0.3)
        else:
            # Forgotten
            recalled = ""
            correct = False
            partial = False
            confidence = 0.1 + rng.uniform(0, 0.2)

        # Calculate latency
        latency = (
            self.profile.base_recall_latency_ms
            + self.profile.latency_per_turn_ms * turns_since_injection
            + rng.uniform(-100, 100)
        )

        return MemoryRecallResult(
            task_id=task_id,
            turn=turn,
            target_memory_id=target_id,
            memory_type=target.memory_type,
            target_content=target.content,
            recalled_content=recalled,
            correct=correct,
            partial=partial,
            confidence=min(1.0, max(0.0, confidence)),
            latency_ms=max(50, latency),
        )

    def recall_all(
        self,
        turn: int,
        task_id: str = "default",
        memory_type: MemoryType | None = None,
    ) -> list[MemoryRecallResult]:
        """Test recall of all memories injected up to this turn."""
        entries = self.store.get_by_turn(turn)
        if memory_type:
            entries = [e for e in entries if e.memory_type == memory_type]
        return [self.recall(e.id, turn, task_id) for e in entries]

    def run_conversation(
        self,
        turns: list[dict[str, Any]],
        task_id: str = "default",
    ) -> list[MemoryRecallResult]:
        """Simulate a full multi-turn conversation with memory injection and recall.

        Each turn dict can have:
        - "inject": list of MemoryEntry to inject this turn
        - "recall": list of memory IDs to test recall for this turn
        """
        results: list[MemoryRecallResult] = []

        for turn_idx, turn_data in enumerate(turns):
            # Inject memories
            for entry in turn_data.get("inject", []):
                entry.injected_at_turn = turn_idx
                self.store.inject(entry)

            # Test recall
            for target_id in turn_data.get("recall", []):
                result = self.recall(target_id, turn_idx, task_id=f"{task_id}-t{turn_idx}")
                results.append(result)

            # Track context
            active = [e.id for e in self.store.get_by_turn(turn_idx)]
            context = MemoryContext(
                turn=turn_idx,
                active_memories=active,
                recalled_correctly=[
                    r.target_memory_id for r in results
                    if r.turn == turn_idx and r.correct
                ],
                recalled_incorrectly=[
                    r.target_memory_id for r in results
                    if r.turn == turn_idx and not r.correct and not self._is_hallucination(r, turn_idx)
                ],
                forgotten=[
                    r.target_memory_id for r in results
                    if r.turn == turn_idx and not r.correct
                ],
            )
            self._contexts.append(context)

        return results

    def _make_seed(self, target_id: str, turn: int, task_id: str) -> int:
        """Deterministic seed for reproducibility."""
        s = self._seed if self._seed is not None else 42
        h = hashlib.sha256(
            f"{task_id}:{target_id}:{turn}:{self.profile.name}:{s}".encode()
        ).digest()
        return int.from_bytes(h[:4], "big")

    def _generate_hallucination(self, target: MemoryEntry, rng: random.Random) -> str:
        """Generate a plausible but fabricated memory."""
        templates = [
            f"I recall {target.content[:30]} was mentioned in the previous session",
            f"Based on our earlier discussion, {target.content[:30]} should be prioritized",
            f"The user previously stated that {target.content[:30]}",
        ]
        return rng.choice(templates)

    def _is_hallucination(self, result: MemoryRecallResult, turn: int) -> bool:
        """Check if a result appears to be a hallucination."""
        if result.correct or result.partial:
            return False
        return len(result.recalled_content) > 0 and result.target_content == ""

    def reset(self) -> None:
        """Reset the agent's memory for a fresh conversation."""
        self.store = MemoryStore()
        self._contexts = []
