from . import *
from app.irsystem.models.search import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Ski Resort Recommendations"
net_id = "Ava Anderson: aca76, Michael Behrens: mcb273, Cameron Haarmann: cmh332, Nicholas Mohan: nhm39, Megan Tormey: mt664"


def max_distance(s):
    # distance is a string either x or x+
    if "+" in s:
        lst = s.split("+")
        return int(lst[0])
    else:
        return int(s)


def get_version():
    v = request.args.get('version')
    if not v:
        return 2
    else:
        return int(v)


@irsystem.route('/', methods=['GET'])
def search():
    query = request.args.get('description')
    version = get_version()
    location = request.args.get('location')
    d = request.args.get('distance')
    if d:
        d = max_distance(d)
    if not query:
        data = []
        output_message = ''
    else:
        output_message = "Your search: " + query
        data = search_q(query, version=version, location=location,
                        distance=d)
    return render_template('front-end-test.html', data=data)
