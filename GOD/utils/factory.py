def get_model(model_name, args):
    name = model_name.lower()
    if name == "god":
        from models.GOD import Learner
    else:
        assert 0
    return Learner(args)