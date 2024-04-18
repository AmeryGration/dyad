def _f1(gamma, delta, period, primary_mass):
    """Return the normalization constant"""
    gamma = np.asarray(gamma)
    delta = np.asarray(delta)
    period = np.asarray(period)
    primary_mass = np.asarray(primary_mass)

    num = 1. 
    denom = (
        0.3**(delta - gamma)
        *(0.3**(gamma + 1.) - 0.1**(gamma + 1.))
        /(gamma + 1.)
        + (1. - 0.3**(delta + 1.))
        /(delta + 1.)
        # + 0.05*_moe2017_twin_excess_constant(delta, period, primary_mass)
    )
    # Handle division by zero
    mask = (denom == np.inf)
    denom[mask] = (
        0.3**(delta[mask] - gamma[mask])
        *(0.3**(gamma[mask] + 1.) - 0.1**(gamma[mask] + 1.))
        /(gamma[mask] + 1.)
    )

    res = num/denom


    return res

def _f2(gamma, delta, period, primary_mass):
    def f(gamma, delta):
        with np.errstate(invalid="raise"):
            # Handle division by zero
            try:
                res = (
                    0.3**(delta - gamma)
                    *(0.3**(gamma + 1.) - 0.1**(gamma + 1.))
                    /(gamma + 1.)
                )
            except FloatingPointError:
                # NB: natural log not common logarithm
                res = 0.3**(delta - gamma)*(np.log(0.3) - np.log(0.1)) 

        return res

    def g(gamma, delta, period, primary_mass):
        with np.errstate(invalid="raise"):
            # Handle division by zero
            try:
                res = (
                    (1. - 0.3**(delta + 1.))
                    /(delta + 1.)
                    + 0.05*_moe2017_twin_excess_constant(
                        delta, period, primary_mass
                    )
                )
            except FloatingPointError:
                # NB: natural log not common logarithm
                res = - np.log(0.3)

        return res

    gamma = np.asarray(gamma)
    delta = np.asarray(delta)
    period = np.asarray(period)
    primary_mass = np.asarray(primary_mass)

    num = 1.
    denom = (
        f(gamma, delta)
        + g(gamma, delta, period, primary_mass)
        # + 0.05*_moe2017_twin_excess_constant(delta, period, primary_mass)
    )
    res = num/denom

    return res

def _f3(gamma, delta, period, primary_mass):
    def f_1(gamma, delta):
        res = 0.3**(delta - gamma)*(np.log(0.3) - np.log(0.1))

        return res

    def f_2(gamma, delta):
        res = (
            0.3**(delta - gamma)
            *(0.3**(gamma + 1.) - 0.1**(gamma + 1.))
            /(gamma + 1.)
        )

        return res

    def g_1(gamma, delta):
        res = - np.log(0.3)*np.ones_like(gamma)

        return res

    def g_2(gamma, delta):
        res = (1. - 0.3**(delta + 1.))/(delta + 1.)

        return res

    gamma = np.asarray(gamma)
    delta = np.asarray(delta)
    period = np.asarray(period)
    primary_mass = np.asarray(primary_mass)

    condition = [
        np.isclose(gamma, -1., atol=1.e-3), ~np.isclose(gamma, -1., atol=1.e-3)
    ]
    value = [f_1(gamma, delta), f_2(gamma, delta)]
    f = np.select(condition, value)

    condition = [delta == -1., delta != -1.]
    value = [g_1(gamma, delta), g_2(gamma, delta)]
    g = np.select(condition, value)    

    num = 1.
    denom = (
        f
        + g
        # + 0.05*_moe2017_twin_excess_constant(delta, period, primary_mass)
    )
    res = num/denom

    return res

# Test utility functions: norm
primary_mass_boundary = (0.8, 1.2, 3.5, 6., 60.)
log10_period_boundary = (
    0.2, 1., 1.3, 2., 2.5, 3.4, 3.5, 4., 4.5, 5.5, 6., 6.5, 8.
)

n = 50
primary_mass = np.hstack(
    [
        np.linspace(0.8, 1.2, n),
        np.linspace(1.2, 3.5, n)[1:],
        np.linspace(3.5, 6., n)[1:],
        np.linspace(6., 60., n)[1:],
    ]
)
log10_period = np.hstack(
    [
        np.linspace(0.2 + 1.e-6, 1., n),
        np.linspace(1., 1.3, n)[1:],
        np.linspace(1.3, 2., n)[1:],
        np.linspace(2., 2.5, n)[1:],
        np.linspace(2.5, 3.4, n)[1:],
        np.linspace(3.4, 3.5, n)[1:],
        np.linspace(3.5, 4., n)[1:],
        np.linspace(4., 4.5, n)[1:],
        np.linspace(4.5, 5.5, n)[1:],
        np.linspace(5.5, 6., n)[1:],
        np.linspace(6., 6.5, n)[1:],
        np.linspace(6.5, 8., n)[1:],
    ]
)
period = 10.**log10_period

gamma = _moe2017_gamma(log10_period, primary_mass.reshape([-1, 1]))
delta = _moe2017_delta(log10_period, primary_mass.reshape([-1, 1]))

norm =_f1(
    gamma, delta, period, primary_mass.reshape([-1, 1])
)

# fig, ax, cbar = plot.plot(cbar=True)
# im = ax.pcolormesh(log10_period, primary_mass, norm)
# ax.contour(log10_period, primary_mass, norm, colors="k")
# # ax.scatter(log10_period[y_ind], primary_mass[x_ind])
# ax.vlines(log10_period_boundary, 0., 60., ls="dashed")
# ax.hlines(primary_mass_boundary, 0., 8., ls="dashed")
# # ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlim(0.2, 8.)
# ax.set_ylim(0.8, 60.)
# ax.set_xlabel(r"$x$")
# ax.set_ylabel(r"$M_{1}$")
# cbar = fig.colorbar(im, cax=cbar)
# cbar.set_label(r"$A_{q}$")
# plt.savefig("norm.pdf")
# plt.show()

# norm =_f2(
#     gamma, delta, period, primary_mass.reshape([-1, 1])
# )

# fig, ax, cbar = plot.plot(cbar=True)
# im = ax.pcolormesh(log10_period, primary_mass, norm)
# ax.contour(log10_period, primary_mass, norm, colors="k")
# # ax.scatter(log10_period[y_ind], primary_mass[x_ind])
# ax.vlines(log10_period_boundary, 0., 60., ls="dashed")
# ax.hlines(primary_mass_boundary, 0., 8., ls="dashed")
# # ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlim(0.2, 8.)
# ax.set_ylim(0.8, 60.)
# ax.set_xlabel(r"$x$")
# ax.set_ylabel(r"$M_{1}$")
# cbar = fig.colorbar(im, cax=cbar)
# cbar.set_label(r"$A_{q}$")
# plt.savefig("norm.pdf")
# plt.show()

# norm =_f3(
#     gamma, delta, period, primary_mass.reshape([-1, 1])
# )

# fig, ax, cbar = plot.plot(cbar=True)
# im = ax.pcolormesh(log10_period, primary_mass, norm)
# ax.contour(log10_period, primary_mass, norm, colors="k")
# # ax.scatter(log10_period[y_ind], primary_mass[x_ind])
# ax.vlines(log10_period_boundary, 0., 60., ls="dashed")
# ax.hlines(primary_mass_boundary, 0., 8., ls="dashed")
# # ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlim(0.2, 8.)
# ax.set_ylim(0.8, 60.)
# ax.set_xlabel(r"$x$")
# ax.set_ylabel(r"$M_{1}$")
# cbar = fig.colorbar(im, cax=cbar)
# cbar.set_label(r"$A_{q}$")
# plt.savefig("norm.pdf")
# plt.show()

# def a(x):
#     return 1./x

# def b(x):
#     return np.pi*np.ones_like(x)

# x = np.array([0., 1.])
# condition = [(x == 1.), (x != 1)]
# values = [a(x), b(x)]
# res = np.select(condition, values)
# print(res)

# def c(x):
#     res = 1./x
#     res[res == np.inf] = np.pi

#     return res

# x = np.array([0., 1.])
# print(c(x))
