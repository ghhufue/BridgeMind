class SolverError(Exception):
    """An error that can cross the solver process boundary."""

    def __init__(self, code, message, field=None, retryable=False, native_code=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.retryable = retryable
        self.native_code = native_code

    def to_dict(self):
        return {key: getattr(self, key) for key in
                ("code", "message", "field", "retryable", "native_code")}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

