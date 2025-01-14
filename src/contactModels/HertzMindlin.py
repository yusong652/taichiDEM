import taichi as ti


class HertzMindlin:
    def __init__(self, YoungModulus, ShearModulus, muStatic, muDynamic):
        self.YoungModulus = YoungModulus
        self.ShearModulus = ShearModulus
        self.muStatic = muStatic
        self.muDynamic = muDynamic

        