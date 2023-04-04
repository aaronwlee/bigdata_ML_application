from flask_wtf import Form
from wtforms.fields import StringField, PasswordField
from flask_wtf.file import FileField
from wtforms.validators import DataRequired, EqualTo, Length
from services.mongo_service import get_collection_expain

# Set your classes here.


class InsertCollectionForm(Form):
    def __init__(self, collection, form):
        super().__init__(form)
        fields = get_collection_expain(collection)
        for field in fields:
            if field != "_id":
                setattr(self, field, StringField(field))

    


class LoginForm(Form):
    name = StringField('Username', [DataRequired()])
    password = PasswordField('Password', [DataRequired()])


class ForgotForm(Form):
    email = StringField(
        'Email', validators=[DataRequired(), Length(min=6, max=40)]
    )

class FileUploadForm(Form):
    file = FileField(
        'Data File', validators=[DataRequired()]
    )
