from crimai.app import create_app
from crimai.models import Media, db
app = create_app()
with app.app_context():
    m = Media.query.filter(Media.output_path.like('%4c93b206%')).first()
    m.output_path = m.output_path.replace('.mp4', '_h264.mp4')
    db.session.commit()
    print('Done:', m.output_path)
