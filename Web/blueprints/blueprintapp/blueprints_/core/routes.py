from web import render_template, Blueprint


# like creating an app, but for blueprints
core = Blueprint('core', import_name=__name__, template_folder='templates')

@core.route('/')
def index():
    return render_template('core/index.html')
