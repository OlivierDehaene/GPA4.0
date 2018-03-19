import logging

from flask_restplus import Api
import settings


log = logging.getLogger(__name__)

# create Flask-RestPlus API
api = Api(version='1.0',
          title='Word to phonetic serving interface',
          description='RESTful API for the word to phonetic models')


# define default error handler
@api.errorhandler
def default_error_handler(error):
    '''
    Default error handler, if something unexpected occured

    :param error: Contains specific error information
    :return: Tuple of JSON object with error information and 500 status code
    '''
    message = 'Unexpected error occured: {}'.format(error.specific)
    log.exception(message)

    if not settings.DEFAULT_FLASK_DEBUG:
        return {'message': message}, 500
