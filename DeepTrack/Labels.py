class Label:
    def __getitem__(self, key):
        n_L = Label()
        if isinstance(key, str):
            n_L.__resolve__ = lambda P: [v[key] for v in self.__resolve__(P) if key in v]
        else:
            n_L.__resolve__ = lambda P: [v[key] for v in self.__resolve__(P)]
        return n_L

    def __resolve__(self, I):
        return I
    
    def __update__(self, history):
        pass