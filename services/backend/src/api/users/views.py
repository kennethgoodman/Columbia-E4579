from flask import request
from flask_restx import Namespace, Resource, fields

from src.api.users.crud import (  # isort:skip
    get_all_users,
    get_user_by_username,
    add_user,
    get_user_by_id,
    update_user,
    delete_user,
)
from src.api.utils.auth_utils import get_user

users_namespace = Namespace("users")


user = users_namespace.model(
    "User",
    {
        "id": fields.Integer(readOnly=True),
        "username": fields.String(required=True),
        "created_date": fields.DateTime,
    },
)

user_post = users_namespace.inherit(
    "User post", user, {"password": fields.String(required=True)}
)


class UsersList(Resource):
    @users_namespace.marshal_with(user, as_list=True)
    def get(self):
        """Returns all users."""
        return get_all_users(), 200

    @users_namespace.expect(user_post, validate=True)
    @users_namespace.response(201, "<user_username> was added!")
    @users_namespace.response(400, "Sorry. That username already exists.")
    def post(self):
        """Creates a new user."""
        post_data = request.get_json()
        username = post_data.get("username")
        password = post_data.get("password")
        response_object = {}

        user = get_user_by_username(username)
        if user:
            response_object["message"] = "Sorry. That username already exists."
            return response_object, 400

        add_user(username, password)
        response_object["message"] = f"{username} was added!"
        return response_object, 201


class Users(Resource):
    @users_namespace.marshal_with(user)
    @users_namespace.response(200, "Success")
    @users_namespace.response(404, "User <user_id> does not exist")
    def get(self, user_id):
        """Returns a single user."""
        user = get_user_by_id(user_id)
        if not user:
            users_namespace.abort(404, f"User {user_id} does not exist")
        return user, 200

    @users_namespace.expect(user, validate=True)
    @users_namespace.response(200, "<user_id> was updated!")
    @users_namespace.response(400, "Sorry. That username already exists.")
    @users_namespace.response(404, "User <user_id> does not exist")
    def put(self, user_id):
        """Updates a user."""
        post_data = request.get_json()
        username = post_data.get("username")
        response_object = {}

        user = get_user_by_id(user_id)
        if not user:
            users_namespace.abort(404, f"User {user_id} does not exist")

        if get_user_by_username(username):
            response_object["message"] = "Sorry. That username already exists."
            return response_object, 400

        update_user(user, username)

        response_object["message"] = f"{user.id} was updated!"
        return response_object, 200

    @users_namespace.response(200, "<user_id> was removed!")
    @users_namespace.response(404, "User <user_id> does not exist")
    def delete(self, user_id):
        """ "Deletes a user."""
        response_object = {}
        user = get_user_by_id(user_id)

        if not user:
            users_namespace.abort(404, f"User {user_id} does not exist")

        delete_user(user)

        response_object["message"] = f"{user.username} was removed!"
        return response_object, 200


users_namespace.add_resource(UsersList, "")
users_namespace.add_resource(Users, "/<int:user_id>")
