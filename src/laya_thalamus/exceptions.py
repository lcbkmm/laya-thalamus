"""Custom exceptions."""


class LayaRouterError(Exception):
    """Base error for laya-thalamus."""


class ModelNotLoadedError(LayaRouterError):
    """Backend model is not available."""


class ToolLimitExceeded(LayaRouterError):
    """Too many tools registered for a single Laya choice head."""


class CircuitTripped(LayaRouterError):
    """Loop / max-round protection fired."""


class ValidationError(LayaRouterError):
    """Input failed safety / length checks."""
