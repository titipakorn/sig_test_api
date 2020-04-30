# This config is tuned for the bundle of person-detection-retail-xxx and person-reidentification-retail-xxx
# models, but should be suitable for other well-trained detector and reid models
# Alse all tracking update intervals are set assumin input frequency about 30FPS

sig_config = dict(
    threshold=1.2,
    model_path='/app/models',
    similarity_model_name='similarity_model.pkl',
    classify_model_name='classifier_model.pkl',
    similarity_response={'0': 'Not Receive',
                         '1': 'Complete', '2': 'Complete', '3': 'Complete'},
    classify_response={'0': 'No Signature',
                       '1': 'OTP', '2': 'OTP_MNP', '3': 'Signature'},
    success_case='Complete',
    fail_case='Incomplete'
)
