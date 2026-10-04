"""The store's own HTTP API, which every chat write goes through so its stock,
day-total and notification logic stays in one place.

Products and sales are open endpoints; day management (start day, expenses,
end day) needs a logged-in staff user, so the client logs in with the
configured mobile/password (or uses a configured token) and retries once on a
401 when the token has expired.
"""

import os

import httpx

from app.actions.base import ActionFailedError


class StoreApi:
    def __init__(
        self,
        options: dict,
        http_client: httpx.AsyncClient | None = None,
    ):
        self._url_env = options.get("api_url_env", "SPORTS_STORE_API_URL")
        self._token_env = options.get("api_token_env", "SPORTS_STORE_API_TOKEN")
        self._mobile_env = options.get("api_mobile_env", "SPORTS_STORE_API_MOBILE")
        self._password_env = options.get("api_password_env", "SPORTS_STORE_API_PASSWORD")
        self._http = http_client
        self._token: str | None = None

    async def get(self, path: str, *, auth: bool = False, params: dict | None = None) -> httpx.Response:
        return await self._send("GET", path, auth=auth, params=params)

    async def post(self, path: str, body: dict | None = None, *, auth: bool = False) -> httpx.Response:
        return await self._send("POST", path, auth=auth, json=body)

    async def put(self, path: str, body: dict | None = None, *, auth: bool = False) -> httpx.Response:
        return await self._send("PUT", path, auth=auth, json=body)

    async def _send(self, method: str, path: str, *, auth: bool, **kwargs) -> httpx.Response:
        base_url = self._base_url()
        headers = {"Authorization": f"Bearer {await self._bearer(base_url)}"} if auth else {}
        response = await self._request(method, f"{base_url}{path}", headers=headers, **kwargs)
        if auth and response.status_code == 401 and self._can_log_in():
            self._token = None  # expired: log in again, once
            headers = {"Authorization": f"Bearer {await self._bearer(base_url)}"}
            response = await self._request(method, f"{base_url}{path}", headers=headers, **kwargs)
        return response

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        try:
            return await self._client().request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise ActionFailedError(
                f"Couldn't reach the store API at {self._base_url()} ({type(exc).__name__}). Nothing was changed."
            ) from exc

    def _base_url(self) -> str:
        base_url = os.environ.get(self._url_env, "").strip().rstrip("/")
        if not base_url:
            raise ActionFailedError(
                f"This needs the store API: set {self._url_env} (e.g. http://localhost:8001). Nothing was changed."
            )
        return base_url

    def _can_log_in(self) -> bool:
        return bool(os.environ.get(self._mobile_env) and os.environ.get(self._password_env))

    async def _bearer(self, base_url: str) -> str:
        if self._token:
            return self._token
        if not self._can_log_in():
            token = os.environ.get(self._token_env, "").strip()
            if not token:
                raise ActionFailedError(
                    "Day management needs a store login: set "
                    f"{self._mobile_env} and {self._password_env} (or {self._token_env})."
                )
            self._token = token
            return token

        response = await self._request(
            "POST",
            f"{base_url}/auth/login",
            json={"mobile": os.environ[self._mobile_env], "password": os.environ[self._password_env]},
        )
        if response.status_code >= 400:
            raise ActionFailedError(
                f"The store rejected the chat's login (HTTP {response.status_code}). "
                f"Check {self._mobile_env} and {self._password_env}."
            )
        self._token = response.json()["token"]
        return self._token

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http


def error_detail(response: httpx.Response) -> str:
    """The store's own error message (FastAPI's `detail`), or the HTTP status."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, str) and detail:
        return detail
    return f"HTTP {response.status_code}"
