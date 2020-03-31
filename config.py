# This config is tuned for the bundle of person-detection-retail-xxx and person-reidentification-retail-xxx
# models, but should be suitable for other well-trained detector and reid models
# Alse all tracking update intervals are set assumin input frequency about 30FPS

sig_config = dict(
    threshold=1.2,
    model_path='',
    similarity_model_name='',
    classify_model_name='',
    classify_reponse={'0': 'Not Receive', '1': 'Complete'},
    success_case='Complete',
    fail_case='Incomplete'
)
