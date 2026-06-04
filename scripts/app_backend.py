"""Local JSON API for the NeuroWave desktop app prototype."""

import argparse
from dataclasses import asdict
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import traceback
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.app_inference import AppInferenceRequest, run_app_inference


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_BYTES = 100 * 1024 * 1024
ARTIFACT_RESULT_FIELDS = (
    "target_crop_wav",
    "predicted_patch_json",
    "predicted_wav",
    "target_spectrogram",
    "predicted_spectrogram",
    "summary",
)
RUN_FOLDER_RESULT_FIELDS = ("run_dir",)


def health_response():
    return {
        "status": "ok",
        "service": "neurowave-app-backend",
    }


def error_response(message, status=HTTPStatus.BAD_REQUEST, details=None):
    payload = {
        "error": {
            "message": str(message),
            "status": int(status),
        }
    }
    if details:
        payload["error"]["details"] = details
    return payload


def parse_json_body(body):
    if not body:
        raise ValueError("request body must not be empty")
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON body: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def predict_response(payload):
    request = AppInferenceRequest(**payload)
    return asdict(run_app_inference(request))


def artifact_paths_from_result(result):
    if not isinstance(result, dict):
        return []
    paths = []
    for field in ARTIFACT_RESULT_FIELDS:
        value = result.get(field)
        if isinstance(value, str) and value:
            paths.append(Path(value))
    return paths


def run_folder_paths_from_result(result):
    if not isinstance(result, dict):
        return []
    paths = []
    for field in RUN_FOLDER_RESULT_FIELDS:
        value = result.get(field)
        if isinstance(value, str) and value:
            paths.append(Path(value))
    return paths


def resolve_artifact_request_path(raw_path):
    if not raw_path:
        raise ValueError("artifact path query parameter is required")
    return Path(raw_path).resolve()


def open_folder(path):
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
        return
    raise RuntimeError("open folder is only supported on Windows in this prototype")


def response_for_exception(error, include_tracebacks=False):
    if isinstance(error, (ValueError, TypeError)):
        return HTTPStatus.BAD_REQUEST, error_response(error, HTTPStatus.BAD_REQUEST)
    if isinstance(error, FileNotFoundError):
        return HTTPStatus.NOT_FOUND, error_response(error, HTTPStatus.NOT_FOUND)

    details = traceback.format_exc() if include_tracebacks else None
    return (
        HTTPStatus.INTERNAL_SERVER_ERROR,
        error_response(
            "backend prediction failed",
            HTTPStatus.INTERNAL_SERVER_ERROR,
            details=details,
        ),
    )


class NeuroWaveBackendHandler(BaseHTTPRequestHandler):
    server_version = "NeuroWaveAppBackend/0.1"

    def do_OPTIONS(self):
        self._send_json({}, status=HTTPStatus.NO_CONTENT)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(health_response())
            return
        if parsed.path == "/artifact":
            self._send_artifact(parsed)
            return
        self._send_json(
            error_response(f"unknown endpoint: {parsed.path}", HTTPStatus.NOT_FOUND),
            status=HTTPStatus.NOT_FOUND,
        )

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/predict":
            self._handle_predict()
            return
        if parsed.path == "/open-folder":
            self._handle_open_folder()
            return
        self._send_json(
            error_response(f"unknown endpoint: {parsed.path}", HTTPStatus.NOT_FOUND),
            status=HTTPStatus.NOT_FOUND,
        )

    def _handle_predict(self):
        try:
            payload = parse_json_body(self._read_request_body())
            result = self.server.predict_function(payload)
            self.server.register_artifacts(result)
            self.server.register_run_folders(result)
            self._send_json(result)
        except Exception as error:
            status, response = response_for_exception(
                error,
                include_tracebacks=getattr(self.server, "debug_errors", False),
            )
            self._send_json(response, status=status)

    def _handle_open_folder(self):
        try:
            payload = parse_json_body(self._read_request_body())
            path = resolve_artifact_request_path(payload.get("path", ""))
            if not self.server.is_registered_run_folder(path):
                self._send_json(
                    error_response("run folder is not available", HTTPStatus.FORBIDDEN),
                    status=HTTPStatus.FORBIDDEN,
                )
                return
            if not path.exists() or not path.is_dir():
                self._send_json(
                    error_response("run folder not found", HTTPStatus.NOT_FOUND),
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            self.server.open_folder_function(str(path))
            self._send_json({"status": "ok", "path": str(path)})
        except Exception as error:
            status, response = response_for_exception(
                error,
                include_tracebacks=getattr(self.server, "debug_errors", False),
            )
            self._send_json(
                response,
                status=status,
            )

    def _read_request_body(self):
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("Content-Length must be an integer") from error
        if length <= 0:
            raise ValueError("request body must not be empty")
        if length > MAX_REQUEST_BYTES:
            raise ValueError("request body is too large")
        return self.rfile.read(length)

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = b"" if status == HTTPStatus.NO_CONTENT else json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_artifact(self, parsed):
        try:
            query = parse_qs(parsed.query)
            path_values = query.get("path", [])
            path = resolve_artifact_request_path(path_values[0] if path_values else "")
        except ValueError as error:
            self._send_json(
                error_response(error, HTTPStatus.BAD_REQUEST),
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        if not self.server.is_registered_artifact(path):
            self._send_json(
                error_response("artifact is not available", HTTPStatus.FORBIDDEN),
                status=HTTPStatus.FORBIDDEN,
            )
            return
        if not path.exists() or not path.is_file():
            self._send_json(
                error_response("artifact file not found", HTTPStatus.NOT_FOUND),
                status=HTTPStatus.NOT_FOUND,
            )
            return

        size = path.stat().st_size
        if size > MAX_ARTIFACT_BYTES:
            self._send_json(
                error_response("artifact file is too large", HTTPStatus.REQUEST_ENTITY_TOO_LARGE),
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return

        content_type, _encoding = mimetypes.guess_type(path.name)
        if path.suffix.lower() == ".wav":
            content_type = "audio/wav"
        if path.suffix.lower() == ".json":
            content_type = "application/json"

        body = path.read_bytes()
        self.send_response(int(HTTPStatus.OK))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        if getattr(self.server, "quiet", False):
            return
        super().log_message(format, *args)


class NeuroWaveBackendServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address,
        handler_class=NeuroWaveBackendHandler,
        predict_function=predict_response,
        open_folder_function=open_folder,
        quiet=False,
        debug_errors=False,
    ):
        super().__init__(server_address, handler_class)
        self.predict_function = predict_function
        self.open_folder_function = open_folder_function
        self.quiet = quiet
        self.debug_errors = debug_errors
        self._artifact_paths = set()
        self._run_folder_paths = set()
        self._artifact_lock = threading.Lock()

    def register_artifacts(self, result):
        paths = {path.resolve() for path in artifact_paths_from_result(result)}
        if not paths:
            return
        with self._artifact_lock:
            self._artifact_paths.update(paths)

    def register_run_folders(self, result):
        paths = {path.resolve() for path in run_folder_paths_from_result(result)}
        if not paths:
            return
        with self._artifact_lock:
            self._run_folder_paths.update(paths)

    def is_registered_artifact(self, path):
        resolved = Path(path).resolve()
        with self._artifact_lock:
            return resolved in self._artifact_paths

    def is_registered_run_folder(self, path):
        resolved = Path(path).resolve()
        with self._artifact_lock:
            return resolved in self._run_folder_paths


def create_server(
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    predict_function=predict_response,
    open_folder_function=open_folder,
    quiet=False,
    debug_errors=False,
):
    return NeuroWaveBackendServer(
        (host, port),
        predict_function=predict_function,
        open_folder_function=open_folder_function,
        quiet=quiet,
        debug_errors=debug_errors,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host address to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind.")
    parser.add_argument("--quiet", action="store_true", help="Disable HTTP access logs.")
    parser.add_argument(
        "--debug-errors",
        action="store_true",
        help="Include tracebacks in 500 responses.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = create_server(
        host=args.host,
        port=args.port,
        quiet=args.quiet,
        debug_errors=args.debug_errors,
    )
    print(f"NeuroWave app backend listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping NeuroWave app backend.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
