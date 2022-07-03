from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import current_user, login_required
from project.recommendation_flow.retriever import get_content_data, ControllerEnum

data_api = Blueprint('data_api', __name__, static_folder='../frontend/build', static_url_path='/')


@data_api.route('/get_images', methods=['GET'])
@login_required
def get_images():
    urls = get_content_data(ControllerEnum.RANDOM)
    return {
        "username": current_user.username,
        "images": [urls],
    }


@data_api.route('/ping', methods=['GET'])
def ping():
    return 'pong'
