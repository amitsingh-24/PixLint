from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Optional


class SecurityError(Exception):
    """Security-related error"""
    pass


class PathTraversalError(SecurityError):
    """Path traversal attempt detected"""
    pass


class InvalidCredentialError(SecurityError):
    """Invalid or missing credentials"""
    pass


class RateLimitExceeded(SecurityError):
    """Rate limit exceeded"""
    pass


class PathSecurityError(SecurityError):
    """Path validation failed"""
    pass


class PipelineSecurityError(SecurityError):
    """Pipeline security error"""
    pass


class CredentialManager:
    """Secure credential management - credentials only from environment variables"""

    _instance: "CredentialManager | None" = None
    _lock = threading.Lock()

    # Allowed credential keys from environment
    ALLOWED_CREDENTIAL_KEYS = frozenset({
        # AWS
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_REGION", "AWS_ENDPOINT_URL",
        # GCS
        "GOOGLE_APPLICATION_CREDENTIALS", "GCS_CREDENTIALS",
        # Azure
        "AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT", "AZURE_STORAGE_KEY",
        # Database
        "DATABASE_URL", "POSTGRES_PASSWORD", "MYSQL_PASSWORD",
        # Generic API keys
        "HF_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "WANDB_API_KEY", "CLEARML_API_KEY",
    })

    def __new__(cls) -> "CredentialManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._credential_cache: dict[str, str] = {}

    def get_credential(self, key: str) -> Optional[str]:
        """Get credential from environment only - never from parameters"""
        if key not in self.ALLOWED_CREDENTIAL_KEYS:
            raise ValueError(f"Credential key not allowed: {key}")

        if key not in self._credential_cache:
            value = os.environ.get(key)
            if value is None:
                raise ValueError(f"Required credential not found in environment: {key}")
            self._credential_cache[key] = value
        return self._credential_cache[key]

    def get_aws_credentials(self) -> dict:
        """Get AWS credentials safely"""
        return {
            "aws_access_key_id": self.get_credential("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": self.get_credential("AWS_SECRET_ACCESS_KEY"),
            "aws_session_token": os.environ.get("AWS_SESSION_TOKEN"),
            "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        }

    def get_gcs_credentials(self) -> dict:
        """Get GCS credentials"""
        return {
            "credentials_path": self.get_credential("GOOGLE_APPLICATION_CREDENTIALS"),
        }

    def get_azure_credentials(self) -> dict:
        """Get Azure credentials"""
        return {
            "connection_string": self.get_credential("AZURE_STORAGE_CONNECTION_STRING"),
        }

    def clear_cache(self) -> None:
        """Clear credential cache (for testing)"""
        self._credential_cache.clear()


class PathValidator:
    """Path validation and sanitization to prevent path traversal"""

    # Allowed base directories - configurable via environment
    _allowed_bases: list[Path] = []
    _lock = threading.Lock()

    # Dangerous patterns that indicate path traversal attempts
    DANGEROUS_PATTERNS = [
        r'\.\.',           # Parent directory traversal
        r'~/',             # Home directory expansion
        r'\$\{',           # Variable expansion
        r'`',              # Command substitution
        r'\$\{|\(\)',      # Variable/command substitution
        r'\|',             # Pipe
        r';',              # Command separator
        r'&',              # Background execution
        r'`',              # Backticks
        r'\x00',           # Null byte
    ]

    # Dangerous file extensions
    DANGEROUS_EXTENSIONS = frozenset({
        '.exe', '.bat', '.cmd', '.com', '.scr', '.pif',
        '.msi', '.dll', '.sys', '.drv',
        '.sh', '.bash', '.zsh', '.fish',
        '.pyc', '.pyo', '.pyd',
        '.so', '.dylib', '.dll',
    })

    # Maximum path length
    MAX_PATH_LENGTH = 4096

    # Allowed file extensions for dataset operations
    ALLOWED_EXTENSIONS = frozenset({
        # Images
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif',
        # Annotations
        '.json', '.xml', '.txt', '.csv', '.yaml', '.yml',
        # Models
        '.pt', '.pth', '.onnx', '.pb', '.tflite',
        # Archives (for import/export)
        '.zip', '.tar', '.tar.gz', '.tgz',
    })

    @classmethod
    def configure_allowed_bases(cls, bases: list[str]) -> None:
        """Configure allowed base directories"""
        with cls._lock:
            cls._allowed_bases = [Path(b).resolve() for b in bases if os.path.exists(b)]

    @classmethod
    def get_allowed_bases(cls) -> list[Path]:
        """Get configured allowed bases, auto-discover if none set"""
        with cls._lock:
            if not cls._allowed_bases:
                # Auto-discover common data directories
                candidates = [
                    "/data", "/workspace", "/data/datasets",
                    os.path.expanduser("~/data"),
                    "/tmp", "/private/tmp", "/private/var",  # macOS symlink handling
                    os.environ.get("CV_DATA_DIR", ""),
                    os.environ.get("CV_WORKSPACE", ""),
                ]
                cls._allowed_bases = [Path(p).resolve() for p in candidates if p and os.path.exists(p)]
        return cls._allowed_bases

    @classmethod
    def validate_path(cls, path: str, base: Optional[str] = None) -> Path:
        """
        Validate and resolve a path, preventing path traversal attacks.

        Args:
            path: The path to validate
            base: Optional base directory to restrict to

        Returns:
            Resolved Path object

        Raises:
            PathTraversalError: If path traversal detected
            PathSecurityError: If path fails validation
        """
        if not path or not isinstance(path, str):
            raise ValueError("Path must be a non-empty string")

        # Check path length
        if len(path) > 4096:
            raise ValueError("Path exceeds maximum length of 4096")

        # Check for dangerous patterns
        for pattern in PathValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, path):
                raise PathTraversalError(f"Dangerous pattern detected in path: {path}")

        # Check for null bytes
        if '\x00' in path:
            raise PathSecurityError("Null byte detected in path")

        # Normalize path
        try:
            path_obj = Path(path).resolve()
        except Exception as e:
            raise PathTraversalError(f"Invalid path: {e}")

        # Check extension
        ext = path_obj.suffix.lower()
        if ext in PathValidator.DANGEROUS_EXTENSIONS:
            raise PathSecurityError(f"Dangerous file extension: {ext}")

        # Verify path is within allowed bases
        if not PathValidator._is_path_allowed(Path(path).resolve()):
            # Check if it's a subdirectory of an allowed base
            path_allowed = any(
                str(Path(path).resolve()).startswith(str(base)) for base in PathValidator.get_allowed_bases()
            )
            if not path_allowed and os.environ.get("CV_ALLOW_ALL_PATHS", "false").lower() != "true":
                raise PathTraversalError(
                    f"Path not within allowed directories: {path}. "
                    f"Allowed bases: {[str(b) for b in PathValidator.get_allowed_bases()]}"
                )

        return path_obj

    @classmethod
    def _is_path_allowed(cls, path: Path) -> bool:
        """Check if path is within allowed bases"""
        for base in cls._allowed_bases:
            try:
                Path(path).relative_to(base)
                return True
            except ValueError:
                continue
        return False

    @classmethod
    def validate_dataset_path(cls, path: str) -> Path:
        """Validate a dataset directory path"""
        path_obj = cls.validate_path(path)
        if not path_obj.exists():
            raise ValueError(f"Dataset path does not exist: {path}")
        if not path_obj.is_dir():
            raise ValueError(f"Dataset path is not a directory: {path}")
        return path_obj

    @classmethod
    def validate_output_path(cls, path: str, allow_create: bool = True) -> Path:
        """Validate an output path for writing.

        Unlike a read path, the target may not exist yet, so we cannot call
        ``Path.resolve(strict=True)``. We still enforce the same anti-traversal
        rules and confine the resolved path to the configured allowed bases so
        that a tool caller cannot write outside the sandbox.
        """
        if not path or not isinstance(path, str):
            raise ValueError("Path must be a non-empty string")
        if len(path) > cls.MAX_PATH_LENGTH:
            raise ValueError(f"Path exceeds maximum length of {cls.MAX_PATH_LENGTH}")
        if '\x00' in path:
            raise PathSecurityError("Null byte detected in path")

        # Reject obvious traversal / injection tokens before resolving.
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, path):
                raise PathTraversalError(f"Dangerous pattern detected in path: {path}")

        try:
            path_obj = Path(path).resolve()
        except Exception as e:
            raise PathTraversalError(f"Invalid path: {e}")

        ext = path_obj.suffix.lower()
        if ext in cls.DANGEROUS_EXTENSIONS:
            raise PathSecurityError(f"Dangerous file extension: {ext}")

        # Confine the resolved path within an allowed base (unless explicitly
        # disabled for trusted single-user / local deployments).
        if not cls._is_within_allowed_bases(path_obj):
            if os.environ.get("CV_ALLOW_ALL_PATHS", "false").lower() != "true":
                raise PathTraversalError(
                    f"Output path not within allowed directories: {path}. "
                    f"Allowed bases: {[str(b) for b in cls.get_allowed_bases()]}"
                )

        if path_obj.exists():
            if not path_obj.is_dir():
                raise ValueError(f"Output path exists but is not a directory: {path}")
        elif not allow_create:
            raise ValueError(f"Output directory does not exist: {path}")
        else:
            # Verify the nearest existing ancestor is writable.
            parent = path_obj.parent
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            if not parent.exists():
                raise ValueError(f"Parent directory does not exist: {path_obj.parent}")
            if not os.access(parent, os.W_OK):
                raise ValueError(f"Parent directory not writable: {parent}")

        return path_obj

    @classmethod
    def _is_within_allowed_bases(cls, path: Path) -> bool:
        """True if ``path`` resolves inside one of the configured allowed bases."""
        resolved = path.resolve()
        for base in cls.get_allowed_bases():
            try:
                resolved.relative_to(base)
                return True
            except ValueError:
                continue
        return False

    @classmethod
    def validate_file_path(cls, path: str, allowed_extensions: Optional[set[str]] = None) -> Path:
        """Validate a file path for reading"""
        path_obj = cls.validate_path(path)
        if not path_obj.exists():
            raise ValueError(f"File does not exist: {path}")
        if not path_obj.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if allowed_extensions:
            ext = path_obj.suffix.lower()
            if ext not in allowed_extensions:
                raise ValueError(f"File extension not allowed: {path_obj.suffix}")

        return path_obj

    @classmethod
    def safe_join(cls, base: str, *paths: str) -> Path:
        """Safely join paths, ensuring result stays within base"""
        cls.validate_path(base)  # validates base; raises if disallowed
        result = Path(base)
        for p in paths:
            result = result / p
        return cls.validate_path(str(result), base=str(Path(base).resolve()))


class InputValidator:
    """Input validation for tool parameters"""

    # Maximum lengths
    MAX_STRING_LENGTH = 10000
    MAX_PATH_LENGTH = 4096
    MAX_JSON_SIZE = 10 * 1024 * 1024  # 10MB

    # Regex patterns
    DATASET_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]{1,100}$')
    DATASET_ID_PATTERN_LOOSE = re.compile(r'^[a-zA-Z0-9_\-\.]{1,200}$')
    FORMAT_PATTERN = re.compile(r'^(coco|voc|yolo|kitti|csv|folder|tfrecord|webdataset|labelme|cvat)$', re.IGNORECASE)
    MODEL_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]{1,50}$')
    PIPELINE_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]{1,100}$')

    # Format validation
    SUPPORTED_FORMATS = frozenset({
        "coco", "voc", "yolo", "kitti", "csv", "folder",
        "tfrecord", "webdataset", "labelme", "cvat", "fiftyone",
        "pytorch", "tensorflow", "ultralytics", "hdf5",
    })

    # Model names
    SUPPORTED_MODELS = frozenset({
        "resnet50", "resnet101", "clip-vit-base", "clip-vit-large",
        "dinov2", "sam", "sam2", "grounding-dino",
    })

    @classmethod
    def validate_dataset_id(cls, dataset_id: str) -> str:
        """Validate dataset ID"""
        if not dataset_id:
            raise ValueError("Dataset ID cannot be empty")
        if len(dataset_id) > 200:
            raise ValueError("Dataset ID too long")
        if not re.match(r'^[a-zA-Z0-9_\-\.]{1,200}$', dataset_id):
            raise ValueError(f"Invalid dataset ID format: {dataset_id}")
        return dataset_id

    @classmethod
    def validate_format(cls, format_str: str) -> str:
        """Validate format string"""
        if not format_str:
            raise ValueError("Format cannot be empty")
        format_lower = format_str.lower()
        if format_lower not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format_str}. Supported: {sorted(cls.SUPPORTED_FORMATS)}")
        return format_lower

    @classmethod
    def validate_model_name(cls, model: str) -> str:
        """Validate model name"""
        if not re.match(r'^[a-zA-Z0-9_\-\.]{1,50}$', model):
            raise ValueError(f"Invalid model name: {model}")
        return model

    @classmethod
    def validate_json_size(cls, data: str) -> dict:
        """Validate and parse JSON with size limit"""
        if len(data) > 10 * 1024 * 1024:  # 10MB
            raise ValueError("JSON payload too large (max 10MB)")
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    @classmethod
    def validate_path(cls, path: str) -> str:
        """Validate path string"""
        if not path:
            raise ValueError("Path cannot be empty")
        if len(path) > 4096:
            raise ValueError("Path too long")
        PathValidator.validate_path(path)
        return path

    @classmethod
    def validate_positive_int(cls, value: Any, max_value: int = 1000000) -> int:
        """Validate positive integer"""
        try:
            val = int(value)
        except (ValueError, TypeError):
            raise ValueError(f"Expected positive integer, got {value}")
        if val <= 0:
            raise ValueError("Value must be positive")
        if val > max_value:
            raise ValueError(f"Value exceeds maximum of {max_value}")
        return val

    @classmethod
    def validate_positive_float(cls, value: Any, max_value: float = 1e9) -> float:
        """Validate positive float"""
        try:
            val = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Expected positive float, got {value}")
        if val <= 0:
            raise ValueError("Value must be positive")
        if val > max_value:
            raise ValueError(f"Value exceeds maximum of {max_value}")
        return val

    @classmethod
    def validate_bbox(cls, bbox: list) -> list[float]:
        """Validate bounding box [x1, y1, x2, y2]"""
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("Bounding box must be list of 4 numbers")
        try:
            bbox = [float(x) for x in bbox]
        except (ValueError, TypeError):
            raise ValueError("Bounding box values must be numbers")
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box: x2 > x1 and y2 > y1 required")
        if any(v < 0 for v in bbox):
            raise ValueError("Bounding box coordinates must be non-negative")
        return bbox

    @classmethod
    def validate_ratio_dict(cls, ratios: dict) -> dict[str, float]:
        """Validate split ratios sum to 1.0"""
        if not isinstance(ratios, dict):
            raise ValueError("Ratios must be a dictionary")
        if not ratios:
            raise ValueError("Ratios cannot be empty")

        total = 0.0
        for key, value in ratios.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Ratio keys must be non-empty strings")
            try:
                val = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Ratio values must be numbers, got {value}")
            if val < 0:
                raise ValueError("Ratios must be non-negative")
            total += val

        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        return {k: float(v) for k, v in ratios.items()}


class CredentialFilter:
    """Filter sensitive data from logs and outputs"""

    SENSITIVE_PATTERNS = [
        (r'(?i)(aws_secret_access_key|aws_secret_key|secret_key|secret|password|pwd|token|api_key|apikey)\s*[:=]\s*["\']?([^\s"\']+)', r'\1=***REDACTED***'),
        (r'(?i)(aws_access_key_id|access_key_id)\s*[:=]\s*["\']?([A-Z0-9]{16,})', r'\1=***REDACTED***'),
        (r'(?i)(connection_string|connectionstring)\s*[:=]\s*["\']?([^\s"\']+)', r'\1=***REDACTED***'),
        (r'(?i)(password|pwd)\s*[:=]\s*["\']?([^\s"\']+)', r'\1=***REDACTED***'),
        (r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?([^\s"\']+)', r'\1=***REDACTED***'),
        (r'(?i)(token)\s*[:=]\s*["\']?([^\s"\']+)', r'\1=***REDACTED***'),
        (r'(mongodb|postgresql|mysql)://([^:]+):([^@]+)@', r'\1://\2:***@'),
    ]

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Remove sensitive information from text"""
        if not isinstance(text, str):
            return str(text)

        result = text
        for pattern, replacement in cls.SENSITIVE_PATTERNS:
            result = re.sub(pattern, replacement, result)
        return result

    @classmethod
    def sanitize_dict(cls, data: dict) -> dict:
        """Recursively sanitize dictionary"""
        if not isinstance(data, dict):
            return data

        result: dict[str, Any] = {}
        for key, value in data.items():
            key_str = cls.sanitize(str(key))
            if isinstance(value, dict):
                result[key_str] = cls.sanitize_dict(value)
            elif isinstance(value, list):
                result[key_str] = [cls.sanitize_dict(v) if isinstance(v, dict) else cls.sanitize(str(v)) for v in value]
            elif isinstance(value, str):
                result[key_str] = cls.sanitize(value)
            else:
                result[key_str] = value
        return result


class RateLimiter:
    """Thread-safe rate limiter with sliding window"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed, record if so"""
        now = time.time()
        window_start = time.time() - self.window_seconds

        with self._lock:
            if key not in self._requests:
                self._requests[key] = []
            # Remove old requests
            self._requests[key] = [ts for ts in self._requests[key] if ts > window_start]

            if len(self._requests[key]) >= self.max_requests:
                return False

            self._requests[key].append(now)
            return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in current window"""
        with self._lock:
            if key not in self._requests:
                return self.max_requests
            current = len([ts for ts in self._requests[key] if ts > time.time() - self.window_seconds])
            return max(0, self.max_requests - current)

    def reset(self, key: Optional[str] = None) -> None:
        """Reset rate limit for key or all keys"""
        with self._lock:
            if key:
                self._requests.pop(key, None)
            else:
                self._requests.clear()


class ResourceLimiter:
    """Resource limits for tool execution"""

    def __init__(
        self,
        max_cpu_seconds: int = 300,
        max_memory_mb: int = 2048,
        max_disk_mb: int = 10240,
        max_execution_time: int = 300,
        max_concurrent: int = 4,
    ):
        self.max_cpu_seconds = max_cpu_seconds
        self.max_memory_mb = max_memory_mb
        self.max_disk_mb = max_disk_mb
        self.max_execution_time = max_execution_time
        self.max_concurrent = max_concurrent

        self._active_count = 0
        self._lock = threading.Lock()
        self._start_times: dict[int, float] = {}

    def acquire(self, tool_name: str) -> bool:
        """Acquire resource slot"""
        with self._lock:
            if self._active_count >= self.max_concurrent:
                return False
            self._active_count += 1
            self._start_times[threading.get_ident()] = time.time()
            return True

    def release(self) -> None:
        with self._lock:
            self._active_count = max(0, self._active_count - 1)
            self._start_times.pop(threading.get_ident(), None)

    def check_limits(self, tool_name: str) -> None:
        """Check if tool execution should be terminated"""
        thread_id = threading.current_thread().ident
        if thread_id in self._start_times:
            elapsed = time.time() - self._start_times[thread_id]
            if elapsed > self.max_execution_time:
                raise SecurityError(f"Tool execution time exceeded: {tool_name} ran for {elapsed:.1f}s")


class SecurityAuditor:
    """Security audit logging"""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self._lock = threading.Lock()
        self._events: list[dict] = []

    def log_security_event(
        self,
        event_type: str,
        tool_name: str,
        dataset_id: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[dict] = None,
        severity: str = "INFO",
    ) -> None:
        """Log a security event"""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "tool_name": tool_name,
            "dataset_id": dataset_id,
            "user_id": user_id,
            "details": details or {},
            "severity": severity,
        }

        with self._lock:
            self._events.append(event)
            if self.log_file:
                try:
                    with open(self.log_file, "a") as f:
                        f.write(json.dumps(event) + "\n")
                except Exception:
                    pass  # Don't fail on logging errors

    def log_tool_call(
        self,
        tool_name: str,
        dataset_id: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Log a tool call"""
        self.log_security_event(
            event_type="TOOL_CALL",
            tool_name=tool_name,
            dataset_id=dataset_id,
            details={"success": success, "error": error},
            severity="INFO" if success else "ERROR",
        )

    def log_security_violation(
        self,
        violation_type: str,
        tool_name: str,
        dataset_id: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        """Log a security violation"""
        self.log_security_event(
            event_type="SECURITY_VIOLATION",
            tool_name=tool_name,
            dataset_id=dataset_id,
            details={"violation_type": violation_type, **(details or {})},
            severity="CRITICAL",
        )

    def get_events(self, since: Optional[float] = None) -> list[dict]:
        """Get security events"""
        with self._lock:
            if since is None:
                return self._events.copy()
            return [e for e in self._events if e["timestamp"] >= since]


# Global instances
credential_manager = CredentialManager()
path_validator = PathValidator()
input_validator = InputValidator()
credential_filter = CredentialFilter()

# Rate limiters
TOOL_RATE_LIMITER = RateLimiter(max_requests=100, window_seconds=60)
DATASET_RATE_LIMITER = RateLimiter(max_requests=50, window_seconds=60)

# Resource limiter
RESOURCE_LIMITER = ResourceLimiter(
    max_cpu_seconds=300,
    max_memory_mb=2048,
    max_disk_mb=10240,
    max_execution_time=300,
    max_concurrent=4,
)

# Security auditor
SECURITY_AUDITOR = SecurityAuditor(
    log_file=os.environ.get("CV_SECURITY_LOG_FILE", "/tmp/pixlint_security.log")
)


# Convenience functions
def validate_dataset_id(dataset_id: str) -> str:
    """Validate dataset ID"""
    return InputValidator.validate_dataset_id(dataset_id)


def validate_path(path: str, base: Optional[str] = None) -> Path:
    """Validate and resolve a path"""
    return PathValidator.validate_path(path, base)


def validate_dataset_path(path: str) -> Path:
    """Validate dataset directory path"""
    return PathValidator.validate_dataset_path(path)


def validate_output_path(path: str, allow_create: bool = True) -> Path:
    """Validate output path"""
    return PathValidator.validate_output_path(path, allow_create)


def validate_file_path(path: str, allowed_extensions: Optional[set[str]] = None) -> Path:
    """Validate a file path for reading (confined to allowed bases)."""
    return PathValidator.validate_file_path(path, allowed_extensions)


def validate_format(format_str: str) -> str:
    """Validate format string"""
    return InputValidator.validate_format(format_str)


def validate_model_name(model: str) -> str:
    """Validate model name"""
    return InputValidator.validate_model_name(model)


def validate_json(data: str) -> dict:
    """Validate and parse JSON with size limit"""
    return InputValidator.validate_json_size(data)


def sanitize_output(text: str) -> str:
    """Sanitize output for logging"""
    return CredentialFilter.sanitize(text)


def sanitize_dict(data: dict) -> dict:
    """Sanitize dictionary for logging"""
    return CredentialFilter.sanitize_dict(data)


def validate_positive_int(value: Any, max_value: int = 1000000) -> int:
    """Validate positive integer"""
    return InputValidator.validate_positive_int(value, max_value)


def validate_ratio_dict(ratios: dict) -> dict[str, float]:
    """Validate split ratios sum to 1.0"""
    return InputValidator.validate_ratio_dict(ratios)


# Rate limiting decorators
def rate_limited(max_requests: int = 100, window_seconds: int = 60, key_prefix: str = ""):
    """Decorator for rate limiting tool calls"""
    limiter = RateLimiter(max_requests, window_seconds)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract dataset_id for rate limiting key
            dataset_id = kwargs.get("dataset_id", "global")
            key = f"{key_prefix}:{dataset_id}"

            if not limiter.is_allowed(key):
                raise SecurityError(f"Rate limit exceeded for {func.__name__}")

            return func(*args, **kwargs)
        return wrapper
    return decorator


# Context manager for resource limiting
@contextmanager
def limited_execution(tool_name: str):
    """Context manager for resource-limited tool execution"""
    if not RESOURCE_LIMITER.acquire(tool_name):
        raise SecurityError(f"Maximum concurrent executions reached for {tool_name}")

    try:
        yield
    finally:
        RESOURCE_LIMITER.release()


# Export main classes and functions
__all__ = [
    # Exceptions
    "SecurityError",
    "PathTraversalError",
    "InvalidCredentialError",
    "RateLimitExceeded",
    "PathSecurityError",
    "PipelineSecurityError",

    # Core classes
    "CredentialManager",
    "PathValidator",
    "InputValidator",
    "CredentialFilter",
    "RateLimiter",
    "ResourceLimiter",
    "SecurityAuditor",

    # Global instances
    "credential_manager",
    "path_validator",
    "input_validator",
    "credential_filter",
    "TOOL_RATE_LIMITER",
    "DATASET_RATE_LIMITER",
    "RESOURCE_LIMITER",
    "SECURITY_AUDITOR",

    # Validators
    "validate_dataset_id",
    "validate_path",
    "validate_dataset_path",
    "validate_output_path",
    "validate_file_path",
    "validate_format",
    "validate_model_name",
    "validate_json",
    "validate_positive_int",
    "validate_ratio_dict",
    "sanitize_output",
    "sanitize_dict",

    # Rate limiting
    "rate_limited",
    "limited_execution",
]
