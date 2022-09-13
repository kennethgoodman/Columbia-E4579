from flask_restx import Namespace, Resource

ping_namespace = Namespace("ping")


class Ping(Resource):
    @ping_namespace.response(200, "Success")
    def get(self):
        """Ping Pong"""
        return {"status": "success", "message": "pong!"}


ping_namespace.add_resource(Ping, "")
