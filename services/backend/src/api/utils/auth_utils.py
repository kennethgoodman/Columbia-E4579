import jwt
from src.api.users.models import User


def get_user(request):
    auth_header = request.headers.get("Authorization")
    if auth_header:
        try:
            access_token = auth_header.split(" ")[1]
            user_id = User.decode_token(access_token)
            return 200, user_id, ""
        except jwt.ExpiredSignatureError:
            print("Expired")
            return 401, None, "Signature expired. Please log in again."
        except jwt.InvalidTokenError:
            print("Invalid")
            return 401, None, "Invalid token. Please log in again."
    else:
        print("TRequired")
        return 403, None, "Token Required"
