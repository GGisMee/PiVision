from flask import request, render_template, redirect, url_for, Blueprint

# If this doesn't workthen mark main directory as sources root, see more at https://www.youtube.com/watch?v=oQ5UfJqW5Jo&list=PLCkHpammh_dhRZ2Codl6j_h1ut4Nx2lfM&index=5, 3:06:50
from blueprintapp.app import db
from blueprintapp.blueprints_.todos.models import Todo

# like creating an app, but for blueprints
todos = Blueprint('todos', import_name=__name__, template_folder='templates')

@todos.route('/')
def index():
    todos = Todo.query.all()
    return render_template('todos/index.html', todos = todos)

@todos.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'GET':
        return render_template('todos/create.html')
    elif request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        done = True if 'done' in request.form.keys() else False
        description = description if description != '' else None
        todo = Todo(title = title, description = description, done = done)
        db.session.add(todo)
        db.session.commit()
        #. todos först här eftersom vi kommer ha flera indexes och detta är en av dem.
        #. förklarligt eftersom vi har prefixen /todos i app
        return redirect(url_for('todos.index'))
        