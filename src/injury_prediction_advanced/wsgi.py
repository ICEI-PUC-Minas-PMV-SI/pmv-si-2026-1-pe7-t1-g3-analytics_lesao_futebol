"""WSGI entrypoint for production servers.

Use a production WSGI server such as waitress or gunicorn.
Example:
  waitress-serve --listen=0.0.0.0:5000 wsgi:app
"""

from app import app
