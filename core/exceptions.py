class OpGradNonImplemented(Exception):
    def __init__(self, message: str = "`grad_fn` not applicable to this op"):
        super().__init__(message)