import traceback
import uuid
from flask import request
from flask_restx import Namespace, Resource, fields
from src.api.polls.models import Poll, Choice, Vote
from src.api.utils.auth_utils import get_user
from src.api.users.models import User
from src import db

polls_namespace = Namespace("polls")

poll_model = polls_namespace.model(
    "Poll",
    {
        "id": fields.Integer(readOnly=True),
        "question": fields.String(required=True),
        "available": fields.Boolean(required=True),
        'choices': fields.List(fields.Nested(polls_namespace.model('Choice', {
            'id': fields.Integer(readOnly=True),
            'text': fields.String(required=True),
        })))
    },
)


class PollList(Resource):
    @polls_namespace.marshal_with(poll_model, as_list=True)
    @polls_namespace.response(200, "Success")
    @polls_namespace.response(401, "Unauthorized")
    def get(self):
        """Retrieves all polls (admin) or only available polls (users)."""
        request.request_id = uuid.uuid4()
        status_code, user_id, exception_message = get_user(request)
        if exception_message or user_id is None:
            print(exception_message)
            return {"message": "Unauthorized"}, 401

        try:
            if user_id == 1: # 1 is the admin
                polls = Poll.query.all()
            else:
                polls = Poll.query.filter_by(available=True).options(db.joinedload(Poll.choices)).all()
            return polls, 200
        except Exception as e:
            db.session.rollback()
            print(f"failed to retrieve polls: {e}")
            print(traceback.format_exc())
            return {"error": "Failed to retrieve polls"}, 500

    @polls_namespace.expect(poll_model)
    @polls_namespace.marshal_with(poll_model, code=201)
    @polls_namespace.response(201, "Poll created successfully.")
    @polls_namespace.response(401, "Unauthorized")
    def post(self):
        """Creates a new poll (admin only)."""
        request.request_id = uuid.uuid4()
        status_code, user_id, exception_message = get_user(request)
        if exception_message:
            print(exception_message)
            return {"message": "Unauthorized"}, 401

        try:
            if user_id != 1: # 1 is the admin
                return {"message": "Unauthorized"}, 401

            data = request.get_json()
            print('data', data)
            new_poll = Poll(
                question=data.get("question"), available=data.get("available", False)
            )
            # Create and associate new choices with the poll
            choices = []
            for choice_data in data.get('choices', []):
                choice = Choice(text=choice_data['text'])
                choices.append(choice)
                new_poll.choices.append(choice)  # Append to the relationship

            db.session.add(new_poll)  # Add the poll (choices are automatically added)
            for choice in choices:
                db.session.add(choice)
            db.session.commit() 
            return new_poll, 201
        except Exception as e:
            db.session.rollback()
            print(f"Failed to create poll: {e}")
            print(traceback.format_exc())
            return {"error": "Failed to create poll"}, 500


class PollDetail(Resource):
    @polls_namespace.marshal_with(poll_model)
    @polls_namespace.response(200, "Success")
    @polls_namespace.response(401, "Unauthorized")
    @polls_namespace.response(404, "Poll not found.")
    def get(self, poll_id):
        """Retrieves a specific poll."""
        request.request_id = uuid.uuid4()
        status_code, user_id, exception_message = get_user(request)
        if exception_message:
            print(exception_message)
            return {"message": "Unauthorized"}, 401
        try:
            poll = Poll.query.get_or_404(poll_id)
            return poll, 200
        except Exception as e:
            print(f"Failed to retrieve poll: {e}")
            print(traceback.format_exc())
            return {"error": "Failed to retrieve poll"}, 500

    @polls_namespace.expect(poll_model)
    @polls_namespace.marshal_with(poll_model)
    @polls_namespace.response(200, "Poll updated successfully.")
    @polls_namespace.response(401, "Unauthorized")
    @polls_namespace.response(404, "Poll not found.")
    def put(self, poll_id):
        """Updates a poll (admin only)."""
        request.request_id = uuid.uuid4()
        status_code, user_id, exception_message = get_user(request)
        if exception_message:
            print(exception_message)
            return {"message": "Unauthorized"}, 401

        try:
            if user_id != 1:
                return {"message": "Unauthorized"}, 401

            poll = Poll.query.get_or_404(poll_id)
            data = request.get_json()
            poll.available = data.get("available", poll.available)
            db.session.commit()
            return poll, 200
        except Exception as e:
            db.session.rollback()
            print(f"Failed to update poll: {e}")
            print(traceback.format_exc())
            return {"error": "Failed to update poll"}, 500

    @polls_namespace.response(204, "Poll deleted successfully.")
    @polls_namespace.response(401, "Unauthorized")
    @polls_namespace.response(404, "Poll not found.")
    def delete(self, poll_id):
        """Deletes a poll (admin only)."""
        request.request_id = uuid.uuid4()
        status_code, user_id, exception_message = get_user(request)
        if exception_message:
            print(exception_message)
            return {"message": "Unauthorized"}, 401
        try:
            if user_id != 1: # 1 is admin
                return {"message": "Unauthorized"}, 401

            poll = Poll.query.get_or_404(poll_id)
            db.session.delete(poll)
            db.session.commit()
            return "", 204
        except Exception as e:
            db.session.rollback()
            print(f"failed to delete poll: {e}")
            print(traceback.format_exc())
            return {"error": "Failed to delete poll"}, 500

class VoteSubmission(Resource):
    @polls_namespace.response(201, "Vote registered successfully.")
    @polls_namespace.response(400, "Bad Request - Invalid choice or missing data.")
    @polls_namespace.response(401, "Unauthorized")
    def post(self, poll_id):
        """Allows a user to vote on a poll."""
        request.request_id = uuid.uuid4()
        status_code, user_id, exception_message = get_user(request)
        if exception_message:
            print(exception_message)
            return {"message": "Unauthorized"}, 401
        try:
            data = request.get_json()
            choice_id = data.get('choice')
            if not choice_id:
                return {"message": "Missing 'choice' in request body."}, 400

            # Check if the choice is valid for this poll
            choice = Choice.query.filter_by(id=choice_id, poll_id=poll_id).first()
            if not choice:
                return {"message": "Invalid choice for this poll."}, 400

            # Check if the user has already voted on this poll
            existing_vote = Vote.query.filter_by(user_id=user_id, poll_id=poll_id).first()
            if existing_vote:
                return {"message": "You have already voted on this poll."}, 400

            new_vote = Vote(user_id=user_id, poll_id=poll_id, choice_id=choice_id)
            db.session.add(new_vote)
            db.session.commit()

            return {"message": "Vote registered successfully."}, 201

        except Exception as e:
            db.session.rollback()
            print(f"Failed to register vote: {e}")
            print(traceback.format_exc())
            return {"error": "Failed to register vote."}, 500

polls_namespace.add_resource(PollList, "")
polls_namespace.add_resource(PollDetail, "/<int:poll_id>")
polls_namespace.add_resource(VoteSubmission, '/<int:poll_id>/votes')