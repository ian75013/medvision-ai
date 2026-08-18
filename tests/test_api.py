"""Test de la sonde de santé de l'API.

Minimal à dessein : c'est cet endpoint que les probes Kubernetes interrogent. S'il cesse de
répondre 200, le pod est redémarré en boucle — la panne la plus brutale du déploiement, et
la plus facile à prévenir.
"""

from fastapi.testclient import TestClient

from src.api.main import app


def test_health_endpoint() -> None:
    """``GET /health`` répond 200 et le corps exact attendu par les probes."""
    client = TestClient(app)
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
