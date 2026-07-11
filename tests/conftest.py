"""Shared pytest fixtures for unit/integration/e2e tests."""
import json
import base64
import hashlib
import hmac
import os
import sys
import time
import types
from pathlib import Path

import pandas as pd
import pytest


def _install_jwt_stub() -> None:
    if "jwt" in sys.modules:
        return

    jwt_module = types.ModuleType("jwt")

    class InvalidTokenError(Exception):
        pass

    class ExpiredSignatureError(Exception):
        pass

    def _b64encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    def _b64decode(data: str) -> bytes:
        padding = "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(data + padding)

    def encode(payload, secret, algorithm=None):
        header = {"alg": algorithm or "HS256", "typ": "JWT"}
        header_part = _b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        payload_part = _b64encode(json.dumps(payload, default=str, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{header_part}.{payload_part}".encode("ascii")
        signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        return f"{header_part}.{payload_part}.{_b64encode(signature)}"

    def decode(token, secret, algorithms=None):
        try:
            header_part, payload_part, signature_part = token.split(".")
            signing_input = f"{header_part}.{payload_part}".encode("ascii")
            expected_signature = hmac.new(
                secret.encode("utf-8"), signing_input, hashlib.sha256
            ).digest()
            if not hmac.compare_digest(expected_signature, _b64decode(signature_part)):
                raise InvalidTokenError("invalid signature")
            payload = json.loads(_b64decode(payload_part).decode("utf-8"))
            if payload.get("exp") is not None and float(payload["exp"]) < time.time():
                raise ExpiredSignatureError("token expired")
            return payload
        except ExpiredSignatureError:
            raise
        except Exception as exc:
            raise InvalidTokenError(str(exc)) from exc

    jwt_module.InvalidTokenError = InvalidTokenError
    jwt_module.ExpiredSignatureError = ExpiredSignatureError
    jwt_module.encode = encode
    jwt_module.decode = decode
    sys.modules["jwt"] = jwt_module


def _install_ariadne_stub() -> None:
    try:
        import ariadne  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    ariadne_module = types.ModuleType("ariadne")

    def graphql_sync(schema, data, context_value=None):
        return False, {"errors": [{"message": "GraphQL is unavailable in tests"}]}

    def gql(schema_text):
        return schema_text

    class QueryType:
        def field(self, name):
            def decorator(func):
                return func

            return decorator

    def make_executable_schema(type_defs, query):
        return {"type_defs": type_defs, "query": query}

    ariadne_module.graphql_sync = graphql_sync
    ariadne_module.gql = gql
    ariadne_module.QueryType = QueryType
    ariadne_module.make_executable_schema = make_executable_schema
    sys.modules["ariadne"] = ariadne_module

    explorer_module = types.ModuleType("ariadne.explorer")

    class ExplorerGraphiQL:
        def html(self, *_args, **_kwargs):
            return "<html><body>GraphiQL unavailable in tests</body></html>"

    explorer_module.ExplorerGraphiQL = ExplorerGraphiQL
    sys.modules["ariadne.explorer"] = explorer_module


_install_jwt_stub()
_install_ariadne_stub()


@pytest.fixture
def sample_dataframe():
    """A small but valid dataset matching the expected training schema.

    NOTE: adjust column names here if backend.preprocessing.REQUIRED_FEATURE_COLUMNS
    differs from ["country", "year", "hicp", "unemployment_rate"] + target "hpi".
    """
    rows = []
    countries = ["FR", "DE", "BE", "ES"]
    for country in countries:
        for year in range(2015, 2021):
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "hicp": 100 + (year - 2015) * 1.5,
                    "unemployment_rate": 8.0 + (year - 2015) * 0.1,
                    "hpi": 100 + (year - 2015) * 5,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_csv(tmp_path, sample_dataframe):
    """Write the sample dataframe to a temp CSV and return its path."""
    csv_path = tmp_path / "training_table.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_csv_with_dvc_file(sample_csv):
    """Same as sample_csv, plus a fake .dvc sidecar file for hash extraction tests."""
    dvc_content = {
        "outs": [
            {
                "md5": "b52b097a27380a9510dc5344da77b6",
                "size": 12345,
                "path": sample_csv.name,
            }
        ]
    }
    dvc_path = Path(f"{sample_csv}.dvc")
    import yaml

    with open(dvc_path, "w") as f:
        yaml.safe_dump(dvc_content, f)
    return sample_csv


@pytest.fixture
def tiny_dataframe():
    """Dataset too small to pass the minimum-rows check (< 20 rows)."""
    return pd.DataFrame(
        {
            "country": ["FR"] * 5,
            "year": list(range(2015, 2020)),
            "hicp": [100.0] * 5,
            "unemployment_rate": [8.0] * 5,
            "hpi": [100.0] * 5,
        }
    )


@pytest.fixture
def missing_columns_dataframe():
    """Dataset missing a required column (unemployment_rate)."""
    return pd.DataFrame(
        {
            "country": ["FR"] * 25,
            "year": list(range(2000, 2025)),
            "hicp": [100.0] * 25,
            "hpi": [100.0] * 25,
        }
    )


@pytest.fixture
def auth_headers():
    """Valid user-scope JWT for tests that hit authenticated prediction routes."""
    import jwt

    secret = os.environ.get("JWT_SECRET", "dev-secret-change-me")
    token = jwt.encode(
        {
            "email": "test@example.com",
            "name": "Test User",
            "scope": "user",
            "exp": time.time() + 3600,
            "iat": time.time(),
        },
        secret,
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}
