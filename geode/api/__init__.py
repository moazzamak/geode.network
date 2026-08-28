"""The GEODE API subpackage: the application layer over the product
core (register -> route -> ledger -> settlement), local-only."""
from geode.api.service import app, create_app

__all__ = ["app", "create_app"]
