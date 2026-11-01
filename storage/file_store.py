"""Private Cloudflare R2 storage through its S3-compatible API."""
from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Iterator
from urllib.parse import quote
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from config.settings import settings


class StorageConfigurationError(RuntimeError):
    """Raised when Cloudflare R2 credentials or bucket settings are missing."""


def _required_settings() -> tuple[str, str, str, str]:
    account_id, access_key = settings.R2_ACCOUNT_ID, settings.R2_ACCESS_KEY_ID
    secret_key, bucket = settings.R2_SECRET_ACCESS_KEY, settings.R2_BUCKET_NAME
    if not all((account_id, access_key, secret_key, bucket)):
        raise StorageConfigurationError(
            "Cloudflare R2 is not configured. Set R2_ACCOUNT_ID, "
            "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and R2_BUCKET_NAME."
        )
    endpoint = settings.R2_ENDPOINT_URL or f"https://{account_id}.r2.cloudflarestorage.com"
    return endpoint, access_key, secret_key, bucket


@lru_cache(maxsize=1)
def _client():
    endpoint, access_key, secret_key, _ = _required_settings()
    return boto3.client(
        "s3", endpoint_url=endpoint, aws_access_key_id=access_key,
        aws_secret_access_key=secret_key, region_name="auto",
        config=Config(retries={"max_attempts": 3, "mode": "standard"},
                      connect_timeout=8, read_timeout=60, max_pool_connections=4),
    )


def _bucket() -> str:
    return _required_settings()[3]


def build_memory_object_key(user_id: str, memory_id: str, category: str, filename: str) -> str:
    """Build a deterministic, traversal-safe key for one memory object."""
    if not user_id or not user_id.strip() or "/" in user_id or chr(92) in user_id:
        raise ValueError("Invalid user ID for object storage.")
    if not memory_id or not memory_id.strip() or "/" in memory_id or chr(92) in memory_id:
        raise ValueError("Invalid memory ID for object storage.")
    if category not in {"original", "extracted"}:
        raise ValueError("Unsupported memory object category.")
    safe_name = Path(filename.replace(chr(92), "/")).name
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("Invalid object filename.")
    return f"memory-vault/users/{user_id}/memories/{memory_id}/{category}/{safe_name}"


def upload_file(file_obj: BinaryIO, key: str, content_type: str) -> str:
    """Stream a file-like object to R2 using a single transfer worker."""
    if not key or not content_type:
        raise ValueError("Object key and content type are required.")
    file_obj.seek(0)
    _client().upload_fileobj(
        file_obj, _bucket(), key, ExtraArgs={"ContentType": content_type},
        Config=TransferConfig(multipart_threshold=8 * 1024 * 1024,
                              max_concurrency=1, use_threads=False),
    )
    return key


def upload_text(content: str, key: str) -> str:
    """Store extracted UTF-8 text in R2."""
    if not isinstance(content, str):
        raise TypeError("Extracted content must be text.")
    _client().put_object(Bucket=_bucket(), Key=key, Body=content.encode("utf-8"),
                         ContentType="text/plain; charset=utf-8")
    return key


def download_file(key: str, destination: str | Path) -> Path:
    """Download an R2 object directly to disk, not application RAM."""
    if not key:
        raise ValueError("Object key is required.")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(_bucket(), key, str(path))
    return path


@contextmanager
def download_to_temp(key: str, suffix: str = "") -> Iterator[Path]:
    """Yield a temporary local copy and always remove it on exit."""
    handle = NamedTemporaryFile(prefix="memory-vault-", suffix=suffix, delete=False)
    path = Path(handle.name)
    handle.close()
    try:
        download_file(key, path)
        yield path
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def download_text(key: str, encoding: str = "utf-8") -> str:
    """Retrieve extracted text only for operations that explicitly need it."""
    response = _client().get_object(Bucket=_bucket(), Key=key)
    body = response["Body"]
    try:
        return body.read().decode(encoding)
    finally:
        body.close()


def delete_file(key: str | None) -> None:
    """Delete an object; deleting an absent key is harmless."""
    if key:
        _client().delete_object(Bucket=_bucket(), Key=key)


def file_exists(key: str) -> bool:
    """Check object existence without downloading its contents."""
    if not key:
        return False
    try:
        _client().head_object(Bucket=_bucket(), Key=key)
        return True
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return False
        raise


def generate_signed_url(key: str, expires_in: int = 900,
                        download_filename: str | None = None) -> str:
    """Create a short-lived URL for a private R2 object."""
    if not key:
        raise ValueError("Object key is required.")
    if not 1 <= int(expires_in) <= 3600:
        raise ValueError("Signed URL expiry must be between 1 and 3600 seconds.")
    params = {"Bucket": _bucket(), "Key": key}
    if download_filename:
        params["ResponseContentDisposition"] = (
            f'attachment; filename="{quote(Path(download_filename).name)}"'
        )
    return _client().generate_presigned_url("get_object", Params=params,
                                            ExpiresIn=int(expires_in))
