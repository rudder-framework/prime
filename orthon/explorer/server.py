#!/usr/bin/env python3
"""
ORTHON Explorer Server

Serves the explorer UI and provides API endpoints for browsing local directories.

Usage:
    python -m orthon.explorer.server ~/Domains
    python -m orthon.explorer.server --port 8080 ~/Domains
"""

import argparse
import json
import mimetypes
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ExplorerHandler(SimpleHTTPRequestHandler):
    """HTTP handler with API endpoints for directory browsing."""

    data_root: Path = None
    static_dir: Path = None
    sql_dir: Path = None

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        # API endpoints
        if path == '/api/files':
            self.handle_list_files()
        elif path.startswith('/api/file/'):
            self.handle_serve_file(path[10:])  # Remove '/api/file/'
        elif path == '/api/sql-reports':
            self.handle_list_sql_reports()
        elif path.startswith('/api/sql-report/'):
            self.handle_serve_sql_report(path[16:])  # Remove '/api/sql-report/'
        elif path == '/' or path == '/explorer.html':
            self.serve_static('explorer.html')
        elif path == '/flow' or path == '/flow_viz.html':
            self.serve_static('flow_viz.html')
        elif path.startswith('/static/'):
            self.serve_static(path[8:])
        else:
            # Try serving from static directory
            self.serve_static(path.lstrip('/'))

    def handle_list_files(self):
        """List all parquet and yaml files in the data directory."""
        files = []

        if self.data_root and self.data_root.exists():
            for root, dirs, filenames in os.walk(self.data_root):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]

                rel_root = Path(root).relative_to(self.data_root)

                for filename in filenames:
                    if filename.endswith(('.parquet', '.yaml', '.yml')):
                        rel_path = rel_root / filename if str(rel_root) != '.' else Path(filename)
                        full_path = Path(root) / filename

                        files.append({
                            'path': str(rel_path),
                            'name': filename,
                            'directory': str(rel_root) if str(rel_root) != '.' else '',
                            'size': full_path.stat().st_size,
                            'type': 'parquet' if filename.endswith('.parquet') else 'yaml',
                        })

        # Sort by directory then name
        files.sort(key=lambda f: (f['directory'], f['name']))

        # Group by directory for tree structure
        tree = {}
        for f in files:
            dir_key = f['directory'] or '(root)'
            if dir_key not in tree:
                tree[dir_key] = []
            tree[dir_key].append(f)

        response = {
            'root': str(self.data_root),
            'files': files,
            'tree': tree,
        }

        self.send_json(response)

    def handle_serve_file(self, rel_path: str):
        """Serve a file from the data directory."""
        if not self.data_root:
            self.send_error(404, 'No data directory configured')
            return

        file_path = self.data_root / rel_path

        # Security: ensure path is within data_root
        try:
            file_path.resolve().relative_to(self.data_root.resolve())
        except ValueError:
            self.send_error(403, 'Access denied')
            return

        if not file_path.exists():
            self.send_error(404, f'File not found: {rel_path}')
            return

        # Serve the file
        content_type, _ = mimetypes.guess_type(str(file_path))
        if content_type is None:
            if file_path.suffix == '.parquet':
                content_type = 'application/octet-stream'
            elif file_path.suffix in ('.yaml', '.yml'):
                content_type = 'text/yaml'
            else:
                content_type = 'application/octet-stream'

        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', file_path.stat().st_size)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        with open(file_path, 'rb') as f:
            self.wfile.write(f.read())

    def handle_list_sql_reports(self):
        """List SQL reports organized by pipeline stage."""
        reports = []

        # Pipeline stages in order
        stages = [
            ('typology', 'Typology', '01_typology'),
            ('signal_vector', 'Signal Vector', '02_signal_vector'),
            ('state_vector', 'State Vector', '03_state_vector'),
            ('geometry', 'Geometry', '04_geometry'),
            ('dynamics', 'Dynamics', '05_dynamics'),
            ('physics', 'Physics', '06_physics'),
        ]

        if self.sql_dir and self.sql_dir.exists():
            # Stage-specific reports (primary)
            stages_dir = self.sql_dir / 'stages'
            if stages_dir.exists():
                for stage_id, stage_name, prefix in stages:
                    for f in sorted(stages_dir.glob(f'{prefix}*.sql')):
                        reports.append({
                            'name': f.stem.replace(prefix + '_', '').replace('_', ' ').title() or stage_name,
                            'filename': f.name,
                            'category': stage_id,
                            'stage': stage_name,
                            'path': f'stages/{f.name}',
                        })

            # Also include legacy reports directory
            reports_dir = self.sql_dir / 'reports'
            if reports_dir.exists():
                for f in sorted(reports_dir.glob('*.sql')):
                    if f.name.startswith('00_'):  # Skip run_all
                        continue
                    reports.append({
                        'name': f.stem,
                        'filename': f.name,
                        'category': 'reports',
                        'stage': 'Reports',
                        'path': f'reports/{f.name}',
                    })

        self.send_json({
            'sql_dir': str(self.sql_dir) if self.sql_dir else None,
            'stages': [s[0] for s in stages] + ['reports'],
            'stage_names': {s[0]: s[1] for s in stages} | {'reports': 'Reports'},
            'reports': reports,
        })

    def handle_serve_sql_report(self, rel_path: str):
        """Serve a SQL report file."""
        if not self.sql_dir:
            self.send_error(404, 'SQL directory not configured')
            return

        file_path = self.sql_dir / rel_path

        # Security: ensure path is within sql_dir
        try:
            file_path.resolve().relative_to(self.sql_dir.resolve())
        except ValueError:
            self.send_error(403, 'Access denied')
            return

        if not file_path.exists():
            self.send_error(404, f'SQL file not found: {rel_path}')
            return

        content = file_path.read_text(encoding='utf-8')
        self.send_json({
            'name': file_path.stem,
            'filename': file_path.name,
            'path': rel_path,
            'content': content,
        })

    def serve_static(self, filename: str):
        """Serve a file from the static directory."""
        if not self.static_dir:
            self.send_error(500, 'Static directory not configured')
            return

        file_path = self.static_dir / filename

        if not file_path.exists():
            self.send_error(404, f'Not found: {filename}')
            return

        content_type, _ = mimetypes.guess_type(str(file_path))
        if content_type is None:
            content_type = 'text/html' if filename.endswith('.html') else 'application/octet-stream'

        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', file_path.stat().st_size)
        self.end_headers()

        with open(file_path, 'rb') as f:
            self.wfile.write(f.read())

    def send_json(self, data: dict):
        """Send a JSON response."""
        content = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(content))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[{self.log_date_time_string()}] {args[0]}")


def run_server(data_dir: str, port: int = 8080):
    """Run the explorer server."""
    static_dir = Path(__file__).parent / 'static'
    sql_dir = Path(__file__).parent.parent / 'sql'
    data_root = Path(data_dir).expanduser().resolve()

    if not data_root.exists():
        print(f"Error: Directory not found: {data_root}")
        sys.exit(1)

    if not static_dir.exists():
        print(f"Error: Static directory not found: {static_dir}")
        sys.exit(1)

    # Configure handler
    ExplorerHandler.data_root = data_root
    ExplorerHandler.static_dir = static_dir
    ExplorerHandler.sql_dir = sql_dir if sql_dir.exists() else None

    server = HTTPServer(('localhost', port), ExplorerHandler)

    print(f"ORTHON Explorer Server")
    print(f"  Data directory: {data_root}")
    print(f"  SQL reports:    {sql_dir}")
    print(f"  Static files:   {static_dir}")
    print(f"")
    print(f"  Explorer:       http://localhost:{port}/explorer.html")
    print(f"  Flow Viz:       http://localhost:{port}/flow")
    print(f"\nPress Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='ORTHON Explorer Server - Browse and analyze parquet files'
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='~/Domains',
        help='Directory to browse (default: ~/Domains)'
    )
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=8080,
        help='Port to listen on (default: 8080)'
    )

    args = parser.parse_args()
    run_server(args.directory, args.port)


if __name__ == '__main__':
    main()
