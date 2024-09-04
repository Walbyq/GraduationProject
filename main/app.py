import os
import shutil
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import ai
import parser

app = Flask(__name__)
app.secret_key = '09062002'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/SashaAlina/PythonProjects/myproject/database.db'
db = SQLAlchemy(app)

query = ''
CLASS_LENGTH = 200


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    src = db.Column(db.String(200), nullable=False)
    acceptability = db.Column(db.Boolean)


@app.route('/', methods=['GET', 'POST'])
def index():
    global query
    db.create_all()
    if request.method == 'POST':
        query = request.form['query']

        # Парсим изображения
        srcs = parser.parse_with_selenium(query)

        # Переносим данные из класса в подкласс db.Model
        for src in srcs:
            image = Image(src=src)
            db.session.add(image)

        # Сохраняем сессию
        db.session.commit()
        return redirect(url_for('images', page=1))

    return render_template('index.html')


@app.route('/images', methods=['GET', 'POST'])
def images():
    # Получаем номер страницы из запроса
    page = request.args.get('page', type=int)

    # Получаем 20 элементов на указанной странице
    imgs = Image.query.paginate(page=page, per_page=20, error_out=False).items

    # Обращаемся к базе данных
    unaccept_results = db.session.query(Image).filter_by(acceptability=False).all()
    accept_results = db.session.query(Image).filter_by(acceptability=True).all()
    other_results = db.session.query(Image).filter_by(acceptability=None).all()

    # Получаем ссылки id изображений
    srcs_unaccept = [result.src for result in unaccept_results]
    srcs_accept = [result.src for result in accept_results]
    srcs_other = [result.src for result in other_results]

    id_unaccept = [result.id for result in unaccept_results]
    id_accept = [result.id for result in accept_results]
    id_other = [result.id for result in other_results]

    if request.method == 'POST':
        for img in imgs:
            if request.form.get(str(img.id)):
                img.acceptability = False
            else:
                img.acceptability = True
        # Сохраняем сессию
        db.session.commit()
        return redirect(url_for('images', page=page + 1))

    # Набираем нужное количество изображений для обучения
    if len(unaccept_results) >= CLASS_LENGTH and len(accept_results) >= CLASS_LENGTH:
        # Загружаем датасет для обучения
        title = parser.transliterate(query)
        session['title'] = title

        path = f'C:/Users/SashaAlina/PythonProjects/myproject/static/images/{title}/'
        session['path'] = path

        parser.streaming_download(path + 'unaccept', title, srcs_unaccept, id_unaccept)
        parser.streaming_download(path + 'accept', title, srcs_accept, id_accept)

        # Загружаем остальные изображения
        parser.streaming_download(path, title, srcs_other, id_other)

        # Очищаем БД, она нам больше не пригодится
        Image.query.delete()
        db.session.commit()

        return redirect(url_for('params'))

    return render_template('images.html', imgs=imgs, query=query,
                           len_accept=len(accept_results), len_unaccept=len(unaccept_results), len_class=CLASS_LENGTH)


@app.route('/model_params', methods=['GET', 'POST'])
def params():
    if request.method == 'POST':
        epochs = request.form['epochs']
        batch_size = request.form['batch_size']

        session['epochs'] = epochs
        session['batch_size'] = batch_size
        return redirect(url_for('results'))

    return render_template('model_params.html')


@app.route('/results')
def results():
    path = session.get('path')
    epochs = int(session.get('epochs'))
    batch_size = int(session.get('batch_size'))

    # Тренируем модель на приемлемых и неприемлемых изображениях
    model, class_names = ai.create_and_train_model(path, batch_size, epochs)

    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(path, filename)
            img, score = ai.get_image_score(model, image_path)
            print('Prediction score: {} ({:.2f}%)'.format(
                class_names[np.argmax(score)],
                100 * np.max(score)))

            if class_names[np.argmax(score)] == 'unaccept':
                new_path = path + 'unaccept/'
                print(new_path)
            else:
                new_path = path + 'accept/'
                print(new_path)
            shutil.move(image_path, new_path)

    return render_template('results.html')


if __name__ == '__main__':
    app.run(debug=True, port=8000)
