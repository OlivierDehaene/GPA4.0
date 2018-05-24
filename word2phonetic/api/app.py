import logging.config

import settings
import decoding_utils
from flask import Flask, Blueprint
from api.restplus import api
from api.endpoints.en.client import ns as en_client_namespace
from api.endpoints.fr.client import ns as es_client_namespace
from api.endpoints.es.client import ns as fr_client_namespace
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# load logging confoguration and create log object
logging.config.fileConfig('logging.conf')
log = logging.getLogger(__name__)


def __get_flask_server_params__():
    '''
    Returns connection parameters of the Flask application

    :return: Tripple of server name, server port and debug settings
    '''
    server_name = decoding_utils.get_env_var_setting('FLASK_SERVER_NAME', settings.DEFAULT_FLASK_SERVER_NAME)
    server_port = decoding_utils.get_env_var_setting('FLASK_SERVER_PORT', settings.DEFAULT_FLASK_SERVER_PORT)

    flask_debug = decoding_utils.get_env_var_setting('FLASK_DEBUG', settings.DEFAULT_FLASK_DEBUG)
    flask_debug = True if flask_debug == '1' else False

    return server_name, server_port, flask_debug


def configure_app(flask_app, server_name, server_port):
    '''
    Configure Flask application

    :param flask_app: instance of Flask() class
    '''
    flask_app.config['SERVER_NAME'] = server_name + ':' + server_port
    flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = settings.RESTPLUS_SWAGGER_UI_DOC_EXPANSION
    flask_app.config['RESTPLUS_VALIDATE'] = settings.RESTPLUS_VALIDATE
    flask_app.config['RESTPLUS_MASK_SWAGGER'] = settings.RESTPLUS_MASK_SWAGGER
    flask_app.config['ERROR_404_HELP'] = settings.RESTPLUS_ERROR_404_HELP


def initialize_app(flask_app, server_name, server_port):
    '''
    Initialize Flask application with Flask-RestPlus

    :param flask_app: instance of Flask() class
    '''
    blueprint = Blueprint('w2p_api', __name__, url_prefix='/w2p_api')

    configure_app(flask_app, server_name, server_port)
    api.init_app(blueprint)
    api.add_namespace(en_client_namespace)
    api.add_namespace(fr_client_namespace)
    api.add_namespace(es_client_namespace)

    flask_app.register_blueprint(blueprint)


def main():
    server_name, server_port, flask_debug = __get_flask_server_params__()
    initialize_app(app, server_name, server_port)
    log.info(
        '>>>>> Starting TF Serving client at http://{}/ >>>>>'.format(app.config['SERVER_NAME'])
    )
    app.run(host="0.0.0.0")


if __name__ == '__main__':
    main()
