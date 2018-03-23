import os
import logging.config
import settings
import pandas as pd
import io

from flask import request, send_file
from flask_restplus import Resource
from api.restplus import api
from api.logic.w2p_serving_logic import g2p_mapping_once, g2p_mapping_file
from werkzeug.datastructures import FileStorage

logging.config.fileConfig('logging.conf')
log = logging.getLogger(__name__)

# create dedicated namespace for english client
ns = api.namespace('english', description='Operations for english word to phonetic client')

# Flask-RestPlus specific parser for image uploading
once_parser = api.parser()

once_parser.add_argument('word',
                         location='form',
                         type=str,
                         required=True)

upload_parser = api.parser()
upload_parser.add_argument('word_corpus',
                           location='files',
                           type=FileStorage,
                           required=True)
upload_parser.add_argument('progression',
                           location='files',
                           type=FileStorage,
                           required=False)


@ns.route('/once')
class W2POnceEN(Resource):
    @ns.doc(description='Predicts the most probable g-p mapping of one word',
            responses={
                200: "Success",
                400: "Bad request",
                500: "Internal server error"
            })
    @ns.expect(once_parser)
    def post(self):
        try:
            input = once_parser.parse_args()["word"]
        except Exception as inst:
            return {'message': 'something wrong with incoming request. ' +
                               'Original message: {}'.format(inst)}, 400

        try:
            results = g2p_mapping_once(input, settings.EN_MODEL_NAME, settings.EN_VOCAB_FILE)
            results_json = {'word': results[0], 'phonetic': results[1], 'mapping': results[2]}
            return {'prediction_result': results_json}, 200

        except Exception as inst:
            return {'message': 'internal error: {}'.format(inst)}, 500


@ns.route('/file')
class W2PFileEN(Resource):
    @ns.doc(description='Predicts the most probable g-p mapping and provides the lesson number of a corpus of words.\n'
                        + 'One word per line.\n'
                        + 'You can provide your own g-p progression.',
            responses={
                200: "Success",
                400: "Bad request",
                500: "Internal server error"
            })
    @ns.expect(upload_parser)
    def post(self):
        try:
            files = upload_parser.parse_args()
            corpus_file = files["word_corpus"]
            progression_file = files["progression"]
            corpus = [str(line, 'Latin-1').strip() for line in corpus_file.readlines()]

            if progression_file == None:
                gpProg = pd.read_csv(os.path.join(settings.EN_FILES, 'gp_prog.csv'))
            else:
                gpProg = pd.read_csv(io.BytesIO(progression_file.read()))

            gpProg = gpProg.loc[gpProg["GP"].notnull()]

        except Exception as inst:
            return {'message': 'something wrong with incoming request. ' +
                               'Original message: {}'.format(inst)}, 400

        try:
            return g2p_mapping_file(corpus, gpProg, settings.EN_MODEL_NAME)

        except Exception as inst:
            return {'message': 'internal error: {}'.format(inst)}, 500


@ns.route('/download_gp_progression')
class W2PDownloadGpProgEN(Resource):
    @ns.doc(description='Download g-p lesson progression',
            responses={
                200: "Success",
                400: "Bad request",
                500: "Internal server error"
            })
    def get(self):
        try:
            return send_file(os.path.join(settings.EN_FILES, "gp_prog.csv"), as_attachment=True)

        except Exception as inst:
            return {'message': 'internal error: {}'.format(inst)}, 500


@ns.route('/download_language_stats')
class W2PDownloadStatsEN(Resource):
    @ns.doc(description='Download g-p freq and consistency csv',
            responses={
                200: "Success",
                400: "Bad request",
                500: "Internal server error"
            })
    def get(self):
        try:
            return send_file(os.path.join(settings.EN_FILES, "stats.csv"), as_attachment=True)

        except Exception as inst:
            return {'message': 'internal error: {}'.format(inst)}, 500
