import jwt
from flask import request
from flask_restx import Namespace, Resource, fields
from src import bcrypt
from src.api.users.crud import add_user, get_user_by_id, get_user_by_username
from src.api.users.models import User
from src.api.utils.auth_utils import get_user

auth_namespace = Namespace("auth")


user = auth_namespace.model(
    "User",
    {
        "username": fields.String(required=True),
    },
)


full_user = auth_namespace.clone(
    "Full User", user, {"password": fields.String(required=True)}
)


login = auth_namespace.model(
    "Login User",
    {
        "username": fields.String(required=True),
        "password": fields.String(required=True),
    },
)

refresh = auth_namespace.model(
    "Refresh", {"refresh_token": fields.String(required=True)}
)

tokens = auth_namespace.clone(
    "Access and refresh_tokens", refresh, {"access_token": fields.String(required=True)}
)

parser = auth_namespace.parser()
parser.add_argument("Authorization", location="headers")


class Register(Resource):
    @auth_namespace.marshal_with(tokens)
    @auth_namespace.expect(full_user, validate=True)
    @auth_namespace.response(201, "Success")
    @auth_namespace.response(400, "Sorry. That username already exists.")
    def post(self):
        post_data = request.get_json()
        username = post_data.get("username")
        password = post_data.get("password")

        user = get_user_by_username(username)
        if user:
            auth_namespace.abort(400, "Sorry. That username already exists.")
        user = add_user(username, password)

        access_token = user.encode_token(user.id, "access")
        refresh_token = user.encode_token(user.id, "refresh")

        response_object = {"access_token": access_token, "refresh_token": refresh_token}
        return response_object, 201


class Login(Resource):
    @auth_namespace.marshal_with(tokens)
    @auth_namespace.expect(login, validate=True)
    @auth_namespace.response(200, "Success")
    @auth_namespace.response(404, "User does not exist")
    def post(self):
        post_data = request.get_json()
        username = post_data.get("username")
        password = post_data.get("password")

        user = get_user_by_username(username)
        if not user or not bcrypt.check_password_hash(user.password, password):
            auth_namespace.abort(404, "User does not exist")

        access_token = user.encode_token(user.id, "access")
        refresh_token = user.encode_token(user.id, "refresh")

        response_object = {"access_token": access_token, "refresh_token": refresh_token}
        return response_object, 200


class Refresh(Resource):
    @auth_namespace.marshal_with(tokens)
    @auth_namespace.expect(refresh, validate=True)
    @auth_namespace.response(200, "Success")
    @auth_namespace.response(401, "Invalid token")
    def post(self):
        post_data = request.get_json()
        refresh_token = post_data.get("refresh_token")
        response_object = {}

        try:
            resp = User.decode_token(refresh_token)
            user = get_user_by_id(resp)

            if not user:
                auth_namespace.abort(401, "Invalid token")

            access_token = user.encode_token(user.id, "access")
            refresh_token = user.encode_token(user.id, "refresh")

            response_object = {
                "access_token": access_token,
                "refresh_token": refresh_token,
            }
            return response_object, 200
        except jwt.ExpiredSignatureError:
            auth_namespace.abort(401, "Signature expired. Please log in again.")
            return "Signature expired. Please log in again."
        except jwt.InvalidTokenError:
            auth_namespace.abort(401, "Invalid token. Please log in again.")


class Status(Resource):
    @auth_namespace.marshal_with(user)
    @auth_namespace.response(200, "Success")
    @auth_namespace.response(401, "Invalid token")
    @auth_namespace.expect(parser)
    def get(self):
        status_code, user_id, exception_message = get_user(request)
        if exception_message:
            auth_namespace.abort(status_code, exception_message)
        user = get_user_by_id(user_id)
        return user, 200


auth_namespace.add_resource(Register, "/register")
auth_namespace.add_resource(Login, "/login")
auth_namespace.add_resource(Refresh, "/refresh")
auth_namespace.add_resource(Status, "/status")
