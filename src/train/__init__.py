"""Training package (WP3): config dataclass, training loop, evaluation.

Public surface:
    TrainConfig / load_config   [config.py]
    train_run(cfg, train_ds, val_ds, ...) -> best checkpoint path [loop.py]
    evaluate(model, loader, device, num_verbs) -> metrics dict    [eval_model.py]
    save_confusion_png(cm, class_names, path)                    [eval_model.py]
"""
