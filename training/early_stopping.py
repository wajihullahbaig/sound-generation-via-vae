class EarlyStopping:
    """Stop when a monitored loss has not improved by min_delta for `patience` checks."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.bad_checks = 0

    def step(self, loss: float) -> bool:
        """Record a new loss; returns True if it is the best so far."""
        if loss < self.best - self.min_delta:
            self.best, self.bad_checks = loss, 0
            return True
        self.bad_checks += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_checks >= self.patience

    def state_dict(self) -> dict:
        """State needed to resume training."""
        return {'best': self.best, 'bad_checks': self.bad_checks}

    def load_state_dict(self, state: dict) -> None:
        """Restore state saved by state_dict()."""
        self.best, self.bad_checks = state['best'], state['bad_checks']
