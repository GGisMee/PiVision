from web import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()

def create_app() -> Flask:
    app = Flask(__name__, template_folder='templates')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./blueprints.db'

    db.init_app(app)

    # import and register all blueprints, merge everythin
    from blueprintapp.blueprints_.core.routes import core
    from blueprintapp.blueprints_.todos.routes import todos
    from blueprintapp.blueprints_.people.routes import people
    app.register_blueprint(core, url_prefix = '/') # startsidan utan prifix
    app.register_blueprint(todos, url_prefix = '/todos') # skapar ett prefix som måte användas varje gång för att nå alla filer i blueprinten todos
    app.register_blueprint(people, url_prefix = '/people') # skapar ett prefix som måte användas varje gång för att nå alla filer i blueprinten todos

    migrate = Migrate(app, db)
     
    return app