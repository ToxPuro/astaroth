import numpy as np


class WelfordVariance():

    def __init__(self):
        self.has_initial_data = False

    def add(self, new_data):
        if not self.has_initial_data:
            self.M2 = np.zeros_like(new_data)
            self.mean = new_data.copy()
            self.n = 1
            self.has_initial_data = True
        else:
            self.n += 1
            delta = new_data - self.mean
            self.mean += delta / self.n
            delta2 = new_data - self.mean
            self.M2 += delta * delta2

    def get_mean(self):
        assert(self.has_initial_data)
        assert(self.n > 0)
        return self.mean

    def get_var(self):
        assert(self.has_initial_data)
        assert(self.n > 0)
        return self.M2 / self.n


class NaiveVariance():

    def __init__(self):
        self.has_initial_data = False


    def add(self, data):
        if not self.has_initial_data:
            self.values = data.copy()
            self.values_squared = data**2
            self.has_initial_data = True
            self.n = 1
        else:
            self.values += data
            self.values_squared += data**2
            self.n += 1

    def get_mean(self):
        assert(self.has_initial_data)
        return self.values / self.n

    def get_var(self):
        assert(self.has_initial_data)
        return (self.values_squared - (self.values**2)/self.n) / self.n

def calc_urms(uux, uuy, uuz):
    print("in calc_urms")
    return np.sqrt(np.mean((uux**2 + uuy**2 + uuz**2)/3))


def calc_Re_real(u_rms, kf, nu):

    return u_rms / (kf*nu)


def calc_Re_mesh(u_rms, kf, dxyz, nu):

    return calc_Re_real(u_rms, kf, nu)*dxyz

def calc_Mach(ux, uy, uz):

    return np.sqrt(np.max(ux**2 + uy**2 + uz**2))


def test_variance():
    import statistics
    n = 10
    x = list(map(lambda x : np.linspace(0,x,10),range(1,n)))

    true_mean = [statistics.mean([arr[i] for arr in x]) for i in range(len(x[0]))]
    true_var = [statistics.variance([arr[i] for arr in x]) for i in range(len(x[0]))]
    xwelf = WelfordVariance()
    xnaive = NaiveVariance()
    for e in x:
        xwelf.add(e)
        xnaive.add(e)
    #xvar = statistics.variance(x)*((n-1)/n)
    #xmean = statistics.mean(x)
    print("mean:", true_mean, xwelf.get_mean(), xnaive.get_mean(), sep="\n")
    print()
    print("variance:", true_var, xwelf.get_var(), xnaive.get_var(), sep="\n")
