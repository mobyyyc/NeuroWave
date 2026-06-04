"""Local JSON API for the NeuroWave desktop app prototype."""

import argparse
from dataclasses import asdict
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisynth.app_inference import AppInferenceRequest, run_app_inference


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_REQUEST_BYTES = 2 * 1024 * 1024


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
        if self.path == "/health":
            self._send_json(health_response())
            return
        self._send_json(
            error_response(f"unknown endpoint: {self.path}", HTTPStatus.NOT_FOUND),
            status=HTTPStatus.NOT_FOUND,
        )

    def do_POST(self):
        if self.path != "/predict":
            self._send_json(
                error_response(f"unknown endpoint: {self.path}", HTTPStatus.NOT_FOUND),
                status=HTTPStatus.NOT_FOUND,
            )
            return

        try:
            payload = parse_json_body(self._read_request_body())
            result = self.server.predict_function(payload)
            self._send_json(result)
        except Exception as error:
            status, response = response_for_exception(
                error,
                include_tracebacks=getattr(self.server, "debug_errors", False),
            )
            self._send_json(response, status=status)

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
        quiet=False,
        debug_errors=False,
    ):
        super().__init__(server_address, handler_class)
        self.predict_function = predict_function
        self.quiet = quiet
        self.debug_errors = debug_errors


def create_server(
    host=DEFAULT_HOST,
    port=DEFAULT_PORT,
    predict_function=predict_response,
    quiet=False,
    debug_errors=False,
):
    return NeuroWaveBackendServer(
        (host, port),
        predict_function=predict_function,
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
