from src.classical_models import get_models

def test_get_models():
    assert len(get_models()) > 0
