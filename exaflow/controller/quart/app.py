from quart import Quart

from exaflow.controller.quart.api_token import require_api_token
from exaflow.controller.quart.endpoints import algorithms
from exaflow.controller.quart.error_handlers import error_handlers

app = Quart(__name__)
app.before_request(require_api_token)
app.register_blueprint(algorithms)
app.register_blueprint(error_handlers)

# Uncomment if you want to debug locally. The app.run cannot receive host and port from here, since it will ignore the env variables passed from the dockerfile run command.
# QUART_HOST = "0.0.0.0"
# QUART_PORT = 5000
# QUART_DEBUG = True  # Change to enable debug
#
# We need to pass host and port as well, otherwise the default values are used
# app.run(host=QUART_HOST, port=QUART_PORT, debug=QUART_DEBUG)
