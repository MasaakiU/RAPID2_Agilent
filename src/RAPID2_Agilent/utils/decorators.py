
class classproperty:
    def __init__(self, func):
        self.func = func
    def __get__(self, owner_self, owner_cls):
        # クラス (owner_cls) を引数として関数を実行する
        return self.func(owner_cls)


