class Args:
    import numpy as np
    import constants as c

    varlist = [ ['r_min',              float],
                ['r_max',              float],
                ['r_N',              int],
                ['z_max',              float],
                ['z_N',              int],

                ['sigma0',          float],
                ['T0',              float],
                ['ps',               float],
                ['q',               float],
                ['alpha',               float],

                ['dtg',               float],
                ['dbgmodel',               str],
                ['v_f',               float],
                ['v_b',               float],
                ['rho_s',               float],
                ['a_min',               float],
                ['a_max',               float],
                ['Nf',               int],

                ['a0',               float],
                ['z0',               float],
                ['r0',               float],

                ['collisions',            bool],
                ['barrier',            str],
                ['feps',               float],
                ['f_diff',               float],
                ['f_coll',               float],
                ['randmotion',            bool],
                ['viscev',               bool],
                ['rad_vel',               str],
                ['t_tot',               float],
                ['r_end',               float],

                ]

    def __init__(self):
        pass

    def read_args(self):
        """
        Read in the simulation parameters from the file 'parameters.inp'
        """
        from configobj import ConfigObj
        import os

        config = ConfigObj('parameters.inp')
        varlist = {v[0]: v[1] for v in self.varlist}



        for name, val in config.items():
            if name not in varlist:
                print('Unknown Parameter:{}'.format(name))
                continue

            t = varlist[name] #datatype of name
            if t in [int, float]:
                setattr(self, name, t(val)) #assign simple values
            elif t in [bool]:
                if (val == 'True'):
                    val = True
                else:
                    val = False
                setattr(self, name, val)
            else: #strings and bool
                setattr(self, name, val)
