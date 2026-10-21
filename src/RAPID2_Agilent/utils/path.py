
from importlib import resources
from . import decorators

class AppPath:
    # 常にパッケージルートを基準にパスを生成
    icon_path = resources.files("RAPID2_Agilent.resources.icons")
    atomic_weights_path = resources.files("RAPID2_Agilent.resources").joinpath("Atomic Weights and Isotopic Compositions 20220704.txt")
    @decorators.classproperty
    def AppIcon(cls, name="app_icon.png"):
        return str(cls.icon_path.joinpath(name))
