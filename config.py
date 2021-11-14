# This config is tuned for the bundle of person-detection-retail-xxx and person-reidentification-retail-xxx
# models, but should be suitable for other well-trained detector and reid models
# Alse all tracking update intervals are set assumin input frequency about 30FPS

# dashdot,no_signature,signature
sig_config = dict(
    accept_threshold=1.2,
    deny_threshold=1.6,
    thicker_kernel=5,
    model_path="/app/models",
    similarity_model_name="similarity_model.pkl",
    classify_model_name="classifier_model.pkl",
    # similarity_response={"0": "Not Receive", "1": "Complete", "2": "Complete", "3": "Complete"},
    similarity_response={"no_signature": "Incomplete", "dashdot": "Incomplete", "signature": "Complete"},
    classify_response={"0": "No Signature", "1": "OTP", "2": "OTP_MNP", "3": "Signature"},
    success_case="Complete",
    unknown_case="Unknown",
    fail_case="Incomplete",
    app={
        "start_width": 1194,
        "end_width": 1444,
        "start_height": 1422,
        "end_height": 1540,
    },
    device={
        "start_width": 950,
        "end_width": 1400,
        "start_height": 2000,
        "end_height": 2090,
    },
)
