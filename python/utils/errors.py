class KernelError(Exception):
    def __init__(self, kernel_name):
        self.message = "ERROR: The given kernel name is not implemented."
        super().__init__(self.message)
