class _DummyExpander:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass
    def markdown(self, *args, **kwargs):
        pass
    def write(self, *args, **kwargs):
        pass
    def json(self, *args, **kwargs):
        pass

# Basic Streamlit function stubs
session_state = {}

def subheader(*args, **kwargs):
    pass

def pyplot(*args, **kwargs):
    pass

def caption(*args, **kwargs):
    pass

def expander(*args, **kwargs):
    return _DummyExpander()

def markdown(*args, **kwargs):
    pass

def columns(n):
    return [_DummyExpander() for _ in range(n)]

def header(*args, **kwargs):
    pass

def write(*args, **kwargs):
    pass

def image(*args, **kwargs):
    pass

def json(*args, **kwargs):
    pass

def metric(*args, **kwargs):
    pass

def progress(*args, **kwargs):
    pass

def button(*args, **kwargs):
    return False

def spinner(*args, **kwargs):
    class _Spinner:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
    return _Spinner()

def download_button(*args, **kwargs):
    pass

def rerun():
    pass
