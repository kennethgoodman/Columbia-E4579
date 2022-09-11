from flask import current_app
from flask_admin.contrib.sqla import ModelView
from src import bcrypt


class UsersAdminView(ModelView):
    column_searchable_list = ("username",)
    column_editable_list = (
        "username",
        "created_date",
    )
    column_filters = ("username",)
    column_sortable_list = (
        "username",
        "active",
        "created_date",
    )
    column_default_sort = ("created_date", True)

    def on_model_change(self, form, model, is_created):
        model.password = bcrypt.generate_password_hash(
            model.password, current_app.config.get("BCRYPT_LOG_ROUNDS")
        ).decode()
