"""API FastAPI de MedVision-AI : application, routes v2 et services d'état.

* :mod:`src.api.main` — ``create_app()``, l'état applicatif et les endpoints historiques ;
* ``src.api.routes`` — les routes de l'API v2, toutes préfixées ``/api``, consommées par le
  front Angular ;
* ``src.api.services`` — l'état vivant : registre versionné, cache borné de sessions ONNX,
  index d'images, diffusion d'événements, veille DVC.

POURQUOI l'état est porté par ``app.state`` et non par des variables de module : les tests
doivent pouvoir construire une application isolée, avec ses propres dossiers d'artefacts et
de données, sans qu'un état global fuite d'un test à l'autre. C'est aussi ce qui permet de
faire tourner plusieurs workers uvicorn sans qu'ils se marchent dessus.

Le détail des deux surfaces d'API — historique et v2 — est dans :mod:`src.api.main`.
"""
