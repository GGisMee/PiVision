from flask import request, render_template, redirect, url_for, Blueprint

# If this doesn't workthen mark main directory as sources root, see more at https://www.youtube.com/watch?v=oQ5UfJqW5Jo&list=PLCkHpammh_dhRZ2Codl6j_h1ut4Nx2lfM&index=5, 3:06:50
from blueprintapp.app import db
from blueprintapp.blueprints_.people.models import Person

# like creating an app, but for blueprints
people = Blueprint('people', import_name=__name__, template_folder='templates')

@people.route('/')
def index():
    people = Person.query.all()
    return render_template('people/index.html', people = people)

@people.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'GET':
        return render_template('people/create.html')
    elif request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        job = request.form.get('job')

        job = job if job != '' else None

        person = Person(name = name, age = age, job = job)

        db.session.add(person)
        db.session.commit()
        #. todos först här eftersom vi kommer ha flera indexes och detta är en av dem.
        #. förklarligt eftersom vi har prefixen /todos i app
        return redirect(url_for('people.index'))
        