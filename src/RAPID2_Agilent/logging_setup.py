# logging_setup.py みたいな場所に置くと便利
import logging

def setup_logging(*, root_level=logging.WARNING, debug_loggers=()):
    # ハンドラは「通す」レベルにして、各loggerのsetLevelで絞るのがコツ
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)  # <- ここを高くすると DEBUG が出なくなる
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()            # 二重出力防止（必要なら）
    root.addHandler(handler)
    root.setLevel(root_level)

    for name in debug_loggers:
        logging.getLogger(name).setLevel(logging.DEBUG)

# 使い方（main.pyなど）
setup_logging(
    root_level=logging.WARNING,
    debug_loggers=[
        # "src.my_chem.views.custom_list_widget.command_list", 
        # "src.my_chem.presenters.main_presenter",
    ],
)
