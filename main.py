import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
def read_mesh(filename):
    f = open(filename)
    x = np.zeros((mx, my))
    y = np.zeros((mx, my))
    i = 0; j = 0
    for line in f:
        x[i,j] = line.split()[0]
        y[i,j] = line.split()[1]
        if (j == my-1):
            i += 1; j = 0
        else:
            j += 1
        if (i == mx): break

    f.close()
    return (x, y)

# -----------------------------------------------------------------------------
def plot_mesh():
    plt.plot(x, y, '.', color='b')
    plt.show()
    return 

# -----------------------------------------------------------------------------
def calc_normal():
    # initialize dn
    # dni/dnj_x : x-components
    # dnj/dnj_y : y-components
    # dni/dnj   : length
    # defined at boundaries
    dni_x = np.zeros((mx, my))
    dni_y = np.zeros((mx, my))
    dni   = np.zeros((mx, my))
    dnj_x = np.zeros((mx, my))
    dnj_y = np.zeros((mx, my))
    dnj   = np.zeros((mx, my))

    # compute dn_bc and |dn_bc| along constant-xi boundaries
    for i in range(0, mx):
        for j in range(1, my):
            dni_x[i,j] =  (y[i,j] - y[i,j-1])
            dni_y[i,j] = -(x[i,j] - x[i,j-1])
            dni[i,j]   = np.sqrt(dni_x[i,j]**2 + dni_y[i,j]**2)
            
    # compute dn_cd and |dn_cd| along constant-eta boundaries
    for i in range(1, mx):
        for j in range(0, my):
            dnj_x[i,j] =  (y[i-1,j] - y[i,j])
            dnj_y[i,j] = -(x[i-1,j] - x[i,j])
            dnj[i,j]   = np.sqrt(dnj_x[i,j]**2 + dnj_y[i,j]**2)

    return (dni_x, dni_y, dni, dnj_x, dnj_y, dnj)

# -----------------------------------------------------------------------------
# compute cell areas
def calc_area():
    # defined at cell-centers
    area = np.zeros((mx, my))
    for i in range(1, mx):
        for j in range(1, my):
            dxac = x[i,  j] - x[i-1,j-1]
            dyac = y[i,  j] - y[i-1,j-1]
            dxbd = x[i-1,j] - x[i,  j-1]
            dybd = y[i-1,j] - y[i,  j-1]
            area[i,j] = .5*abs(dxac*dybd - dxbd*dyac)
    return area

# -----------------------------------------------------------------------------
# compute pressure
def pressure(rho, u, v, et):
    return (gamma-1) * (et - .5 * rho * (u**2 + v**2))

# -----------------------------------------------------------------------------
# initialize conservative variables
def ic():
    # defined at cell-centers (including phatom cells)
    q = np.ones((lmax, mx+1, my+1))
    if (ibcin == 1):
        q[0,:,:] = 1.
        q[1,:,:] = .5
        q[2,:,:] = 0.
        q[3,:,:] = 1.
    if (ibcin == 2):
        q[0,:,:] = 1.
        q[1,:,:] = 1.5
        q[2,:,:] = 0.
        q[3,:,:] = 2.
    return q

# -----------------------------------------------------------------------------
# evulation of the flux vectors
def flux():
    # initialize flux vectors
    ei = np.zeros((lmax, mx, my))
    ej = np.zeros((lmax, mx, my))
    fi = np.zeros((lmax, mx, my))
    fj = np.zeros((lmax, mx, my))

    # compute flux vectors along constant-xi boudnaries
    # defined at boundaries
    for i in range(0, mx):
        for j in range(1, my):
            rho = .5 * (q[0,i,  j] + q[0,i+1,j])
            u   = .5 * (q[1,i,  j] / q[0,i,  j] + \
                        q[1,i+1,j] / q[0,i+1,j])
            v   = .5 * (q[2,i,  j] / q[0,i,  j] + \
                        q[2,i+1,j] / q[0,i+1,j])
            et  = .5 * (q[3,i,  j] + q[3,i+1,j])
            p   = pressure(rho, u, v, et)
            ei[0,i,j] = rho * u
            ei[1,i,j] = rho * u**2 + p
            ei[2,i,j] = rho * u * v
            ei[3,i,j] = (et + p) * u
            fi[0,i,j] = rho * v
            fi[1,i,j] = rho * u * v
            fi[2,i,j] = rho * v**2 + p
            fi[3,i,j] = (et + p) * v

    # compute flux vectors along constant-eta boudnaries
    # defined at boundaries
    # interior cells
    for i in range(1, mx):
        for j in range(0, my):
            rho = .5 * (q[0,i,  j] + q[0,i,j+1])
            u   = .5 * (q[1,i,  j] / q[0,i,  j] + \
                        q[1,i,j+1] / q[0,i,j+1])
            v   = .5 * (q[2,i,  j] / q[0,i,  j] + \
                        q[2,i,j+1] / q[0,i,j+1])
            et  = .5 * (q[3,i,  j] + q[3,i,j+1])
            p   = pressure(rho, u, v, et)
            
            ej[0,i,j] = rho * u
            ej[1,i,j] = rho * u**2 + p
            ej[2,i,j] = rho * u * v
            ej[3,i,j] = (et + p) * u
            fj[0,i,j] = rho * v
            fj[1,i,j] = rho * u * v
            fj[2,i,j] = rho * v**2 + p
            fj[3,i,j] = (et + p) * v

    # upper boundary cells
    # zeroth-order extrapolation of pressure
    # j = my-1
    # for i in range(1, mx):
    #     rho = q[0,i,j]
    #     u   = q[1,i,j] / q[0,i,j]
    #     v   = q[2,i,j] / q[0,i,j]
    #     et  = q[3,i,j]
    #     p   = pressure(rho, u, v, et)

    #     ej[0,i,j] = 0.
    #     ej[1,i,j] = p
    #     ej[2,i,j] = 0.
    #     ej[3,i,j] = 0.
    #     fj[0,i,j] = 0.
    #     fj[1,i,j] = 0.
    #     fj[2,i,j] = p
    #     fj[3,i,j] = 0.

    return (ei, ej, fi, fj)

# -----------------------------------------------------------------------------
# evulation of source term
def source():
    # defined at cell-centers
    h = np.zeros((lmax, mx, my))
    for i in range(1, mx):
        for j in range(1, my):
            yloc = .25 * (y[i,  j] + y[i-1,  j] \
                        + y[i,j-1] + y[i-1,j-1])
            rho  = q[0,i,j]
            u    = q[1,i,j] / q[0,i,j]
            v    = q[2,i,j] / q[0,i,j]
            et   = q[3,i,j]
            p    = pressure(rho, u, v, et)

            h[0,i,j] = - rho*v / yloc
            h[1,i,j] = - rho*u*v / yloc
            h[2,i,j] = - rho*v*v / yloc
            h[3,i,j] = - (et+p)*v / yloc
    return h

# -----------------------------------------------------------------------------
# evulation of the dissipation vectors
def dissp():
    kappa2 = visc/4.
    kappa4 = visc/256.

    # compute flow solutions at cell-centers
    pr = np.zeros((mx+1, my+1))
    for i in range(0, mx+1):
        for j in range(0, my):
            rho = q[0,i,j]
            u   = q[1,i,j] / q[0,i,j]
            v   = q[2,i,j] / q[0,i,j]
            et  = q[3,i,j]
            pr[i,j] = pressure(rho, u, v, et)
    
    # compute dxi at constant-xi boundaries
    dwx = np.zeros((lmax, mx, my))
    for j in range(1, my):
        # compute switch function nu_xi at cell-centers
        dp = np.zeros(mx)
        for i in range(1, mx):
            dp[i] = abs(pr[i+1,j] - 2*pr[i,j] + pr[i-1,j]) / \
                    abs(pr[i+1,j] + 2*pr[i,j] + pr[i-1,j])

        # compute dxi*dU/dxi at boundaries
        d1q = np.zeros((lmax, mx))
        for i in range(1, mx-1):
            d1q[:,i] = q[:,i+1,j] - q[:,i,j]

        # compute dxi3*d3U/dxi3 at boundaries
        d3q = np.zeros((lmax, mx))
        for i in range(1, mx-1):
            d3q[:,i] = d1q[:,i+1] - 2*d1q[:,i] + d1q[:,i-1]

        # compute dxi
        for i in range(1, mx-1):
            dnx  = dni_x[i,j]
            dny  = dni_y[i,j]
            rho  = .5 * (q[0,i,  j] + q[0,i+1,j])
            u    = .5 * (q[1,i,  j] / q[0,i,  j] + \
                         q[1,i+1,j] / q[0,i+1,j])
            v    = .5 * (q[2,i,  j] / q[0,i,  j] + \
                         q[2,i+1,j] / q[0,i+1,j])
            p    = .5 * (pr[i,j] + pr[i+1,j])
            c    = np.sqrt(gamma * p / rho)
            lam  = abs((u*dnx + v*dny) / np.sqrt(dnx**2 + dny**2)) + c
            eps2 = kappa2 * max(dp[i], dp[i+1])
            eps4 = max(0., kappa4 - eps2)

            dwx[:,i,j] = lam*(eps2 * d1q[:,i] - eps4 * d3q[:,i])
        dwx[:,0,   j] = 0.
        dwx[:,mx-1,j] = 0.
    
    # compute deta at constant-xi boundaries
    dwy = np.zeros((lmax, mx, my))
    for i in range(1, mx):
        # compute switch function nu_eta at cell-centers
        dp = np.zeros(my)
        for j in range(1, my-1):
            dp[j] = abs(pr[i,j+1] - 2*pr[i,j] + pr[i,j-1]) / \
                    abs(pr[i,j+1] + 2*pr[i,j] + pr[i,j-1])
        dp[0]    = -dp[1]
        dp[my-1] = 2*dp[my-2] - dp[my-3]
        # HERE!

        # compute deta*dU/deta at boundaries
        d1q = np.zeros((lmax, my))
        for j in range(1, my-1):
            d1q[:,j] = q[:,i,j+1] - q[:,i,j]

        # compute deta3*d3U/deta3 at boundaries
        d3q = np.zeros((lmax, my))
        for j in range(1, my-1):
            d3q[:,j] = d1q[:,j+1] - 2*d1q[:,j] + d1q[:,j-1]

        # compute deta
        for j in range(1, my-1):
            dnx  = dnj_x[i,j]
            dny  = dnj_y[i,j]
            rho  = .5 * (q[0,i,  j] + q[0,i,j+1])
            u    = .5 * (q[1,i,  j] / q[0,i,  j] + \
                         q[1,i,j+1] / q[0,i,j+1])
            v    = .5 * (q[2,i,  j] / q[0,i,  j] + \
                         q[2,i,j+1] / q[0,i,j+1])
            p    = .5 * (pr[i,j] + pr[i,j+1])
            c    = np.sqrt(gamma * p / rho)
            lam  = abs((u*dnx + v*dny) / np.sqrt(dnx**2 + dny**2)) + c
            eps2 = kappa2 * max(dp[j], dp[j+1])
            eps4 = max(0., kappa4 - eps2)

            dwy[:,i,j] = lam*(eps2 * d1q[:,j] - eps4 * d3q[:,j])
        dwy[:,i,my-1] = 0.


    return (dwx, dwy, pr)

# -----------------------------------------------------------------------------
# evulation of residual vectors
def residual():
    res = np.zeros((lmax, mx, my))
    for i in range(1, mx):
        for j in range(1, my):
            # physical flux
            flux_ab = ej[:,i,j-1] * dnj_x[i,j-1] \
                    + fj[:,i,j-1] * dnj_y[i,j-1]
            flux_bc = ei[:,i,  j] * dni_x[i,  j] \
                    + fi[:,i,  j] * dni_y[i,  j]
            flux_cd = ej[:,i,  j] * dnj_x[i,  j] \
                    + fj[:,i,  j] * dnj_y[i,  j]
            flux_da = ei[:,i-1,j] * dni_x[i-1,j] \
                    + fi[:,i-1,j] * dni_y[i-1,j]
            flux_phys = - flux_ab + flux_bc + flux_cd - flux_da

            # AV flux
            flux_ab = dwy[:,i,j-1] * dnj[i,j-1]
            flux_bc = dwx[:,i,  j] * dni[i,  j]
            flux_cd = dwy[:,i,  j] * dnj[i,  j]
            flux_da = dwx[:,i-1,j] * dni[i-1,j]
            flux_av = - flux_ab + flux_bc + flux_cd - flux_da

            # compute residual vector
            res[:,i,j] = - (flux_phys - flux_av)/area[i,j] \
                         + alpha*h[:,i,j]
    return res

# -----------------------------------------------------------------------------
# update q in phantom cells along solid boundary
def bc_wall():
    global q
    j = my
    for i in range(1, mx):
        u         = q[1,i,j-1] / q[0,i,j-1]
        v         = q[2,i,j-1] / q[0,i,j-1]
        vn        = (u*dnj_x[i,j-1] + v*dnj_y[i,j-1]) / dnj[i,j-1]
        u_ghost   = u - 2*vn*dnj_x[i,j-1] / dnj[i,j-1]
        v_ghost   = v - 2*vn*dnj_y[i,j-1] / dnj[i,j-1]
        rho_ghost = q[0,i,j-1]
        p_ghost   = pr[i,j-1]
        et_ghost  = p_ghost/(gamma-1) + .5*rho_ghost* \
                    (u_ghost**2 + v_ghost**2)

        q[0,i,j]  = rho_ghost
        q[1,i,j]  = rho_ghost * u_ghost
        q[2,i,j]  = rho_ghost * v_ghost
        q[3,i,j]  = et_ghost
    return

# -----------------------------------------------------------------------------
# update q in phantom cells along symmetric boundary
def bc_symmetric():
    global q
    j = 0
    for i in range(1, mx):
        q[0,i,j] =  q[0,i,j+1]
        q[1,i,j] =  q[1,i,j+1]
        q[2,i,j] = -q[2,i,j+1]
        q[3,i,j] =  q[3,i,j+1]
    return

# ----------------------------------------------------------------------------- # update q in phantom cells along inlet boundary
def bc_inflow():
    global q
    i = 0
    
    # subsonic inflow (ibcin = 1)
    # specify following conditions:
    # inlet flow angle = 0
    # inlet stagnation pressure and temperature
    # extrapolate: static pressure
    if (ibcin == 1):
        for j in range(1, my):
            # extrapolate static pressure
            rho  = q[0,i+1,j]
            u    = q[1,i+1,j] / q[0,i+1,j]
            v    = q[2,i+1,j] / q[0,i+1,j]
            et   = q[3,i+1,j]
            p1   = pressure(rho, u, v, et)
            rho  = q[0,i+2,j]
            u    = q[1,i+2,j] / q[0,i+2,j]
            v    = q[2,i+2,j] / q[0,i+2,j]
            et   = q[3,i+2,j]
            p2   = pressure(rho, u, v, et)
            p    = 2*p1 - p2

            m2   = 2/(gamma-1) * (p**((1-gamma)/gamma) - 1)
            t    = 1/(1 + .5*(gamma-1)*m2)
            rho  = p / t
            u    = np.sqrt(m2*gamma*t)
            v    = 0.
            et   = p/(gamma-1) + .5*rho*(u**2 + v**2)

            q[0,i,j] = rho
            q[1,i,j] = rho*u
            q[2,i,j] = rho*v
            q[3,i,j] = et


    # supersonic inflow (ibcin = 2)
    # specify following conditions:
    # inlet Mach number
    # inlet flow angle = 0
    # inlet stagnation pressue = 1
    # inlet stagnation temperature = 1
    if (ibcin == 2):
        m = 1.5
        for j in range(1, my):
            term = 1 / (1 + .5*(gamma-1) * m**2)
            p    = term**(gamma/(gamma-1))
            rho  = p/term
            c    = np.sqrt(gamma*p/rho)
            u    = m * c
            v    = 0.
            et   = p/(gamma-1) + .5*rho*(u**2 + v**2)

            q[0,i,j] = rho
            q[1,i,j] = rho*u
            q[2,i,j] = rho*v
            q[3,i,j] = et
    return

# ----------------------------------------------------------------------------- # update q in phantom cells along inlet boundary
def bc_outflow():
    global q
    i = mx

    # subsonic outflow (ibcout = 1)
    # specify following condition:
    # static pressure pback
    # extrapolate other variables
    if (ibcout == 1):
        pback = .8

        for j in range(1,my):
            p     = pback
            # extraploate rho, u, v
            rho1  = q[0,i-1,j]
            u1    = q[1,i-1,j] / q[0,i-1,j]
            v1    = q[2,i-1,j] / q[0,i-1,j]
            rho2  = q[0,i-2,j]
            u2    = q[1,i-2,j] / q[0,i-2,j]
            v2    = q[2,i-2,j] / q[0,i-2,j]
            rho   = 2*rho1 - rho2
            u     = 2*u1 - u2
            v     = 2*v1 - v2

            et = p/(gamma-1) + .5*rho*(u**2 + v**2)
            q[0,i,j] = rho
            q[1,i,j] = rho*u
            q[2,i,j] = rho*v
            q[3,i,j] = et

    # supersonic outflow (ibcout = 2)
    if (ibcout == 2):
        for j in range(1, my):
            q[:,i,j] = 2*q[:,i-1,j] - q[:,i-2,j]
            #q[:,i,j] = q[:,i-1,j]
    return

# -----------------------------------------------------------------------------
# compute time step
def step():
    # defined at cell-centers
    dt = np.zeros((mx,my))
    for i in range(1, mx):
        for j in range(1, my):
            xi = .5*(x[i,j] - x[i-1,j] + x[i,j-1] - x[i-1,j-1])
            xj = .5*(x[i,j] - x[i,j-1] + x[i-1,j] - x[i-1,j-1])
            yi = .5*(y[i,j] - y[i-1,j] + y[i,j-1] - y[i-1,j-1])
            yj = .5*(y[i,j] - y[i,j-1] + y[i-1,j] - y[i-1,j-1])
            
            rho = q[0,i,j]
            u   = q[1,i,j] / q[0,i,j]
            v   = q[2,i,j] / q[0,i,j]
            et  = q[3,i,j]
            p   = pressure(rho, u, v, et)
            c   = np.sqrt(gamma*p/rho)

            qi  =  yj*u - xj*v
            qj  = -yi*u + xi*v
            ci  = c*np.sqrt(xj**2 + yj**2)
            cj  = c*np.sqrt(xi**2 + yi**2)
            dti = area[i,j]/(abs(qi)+abs(ci))
            dtj = area[i,j]/(abs(qj)+abs(cj))

            dt[i,j]  = cfl * (dti*dtj) / (dti+dtj)
    return dt

# -----------------------------------------------------------------------------
# convergence moniter
def monitor():
    global reslist, stop, line1, line2, line3, line4, cb
    ijmax        = abs(res[0,:,:]).argmax()
    (imax, jmax) = np.unravel_index(ijmax, res[0,:,:].shape)
    resmax       = abs(res[0,imax,jmax])
    resavg       = np.mean(abs(res[0,:,:]))

    if (t % 10 == 0):
        print ' step', '%10s' % 'avg.res', '%10s' % 'max.res', \
                       '%11s' % 'x',       '%11s' % 'y'

    print '%5d' % t, '%.4e' % resavg,        '%.4e' % resmax, \
                     '%+.4e' % x[imax,jmax], '%+.4e' % y[imax,jmax]
    reslist.append(resmax)

    if (t % plot_interval == 0):
        if t==0:
            # mid-height
            j = int(mach.shape[-1]/2)
            line1, = ax1.plot(xc[1:,j],mach[1:,j],'-o')
            # symmetric line
            j = 1
            line2, = ax1.plot(xc[1:,j],mach[1:,j],'-o')
            # solid boundary
            j = -1
            line3, = ax1.plot(xc[1:,j],mach[1:,j],'-o')
            ax1.set_ylim(1,3)

            # contour plot of mach number
            con = ax2.contourf(xc[1:,1:],yc[1:,1:],mach[1:,1:], \
                    10,cmap=plt.cm.rainbow)
            #plt.clabel(con, fontsize=12,colors='k')
            ax2.set_xlim(-1,2)
            ax2.set_ylim(0,1)

            # residual plot
            line4, = ax3.plot(range(0, len(reslist)), reslist)
            ax3.set_yscale('log')
            ax3.set_xlim(0,100)
            ax3.set_ylim(10**(int(np.log10(np.amin(reslist)))-1), \
                         10**(int(np.log10(np.amax(reslist)))+1))
            fig.show()
        else:
            j = int(mach.shape[-1]/2)
            line1.set_ydata(mach[1:,j])
            j = 1
            line2.set_ydata(mach[1:,j])
            j = -1
            line3.set_ydata(mach[1:,j])

            ax2.cla()
            con = ax2.contourf(xc[1:,1:],yc[1:,1:],mach[1:,1:], \
                    10,cmap=plt.cm.rainbow)
            #plt.clabel(con, fontsize=12,colors='k')
            ax2.set_xlim(-1,2)
            ax2.set_ylim(0,1)

            line4.set_xdata(range(0, len(reslist)))
            line4.set_ydata(reslist)
            ax3.set_xlim(0, (t/100+1)*100)
            ax3.set_ylim(10**(int(np.log10(np.amin(reslist)))-1), \
                         10**(int(np.log10(np.amax(reslist)))+1))
            fig.canvas.draw()

    if (resmax < eps): stop = True
    return

# -----------------------------------------------------------------------------
# four-stage Runge-Kutta scheme
def rk4():
    global q, ei, ej, fi, fj, dwx, dwy, pr, h, res

    rk   = [1./4, 1./3, 1./2, 1.]
    
    dt = np.ones((mx, my))*.02
    
    q_old = np.copy(q)

    # time stepping
    for m in range(0,4):
        # compute flux vectors
        (ei, ej, fi, fj) = flux()

        # compute artificial viscosity dissipation vectors
        (dwx, dwy, pr) = dissp()

        # compute source vector
        if (alpha == 1): h = source()

        # compute residual vector
        res = residual()

        # time step
        dt = step()

        # update solution
        for i in range(1, mx):
            for j in range(1, my):
                q[:,i,j] = q_old[:,i,j] + rk[m]*dt[i,j]*res[:,i,j]

        # boundary conditions
        bc_wall()
        bc_symmetric()
        bc_inflow()
        bc_outflow()

    # convergence monitor
    calc_mach()

    monitor()
    return

# -----------------------------------------------------------------------------
def calc_mach():
    global xc, yc, mach
    mach = np.zeros((mx,my))
    xc   = np.zeros((mx,my))
    yc   = np.zeros((mx,my))
    for i in range(1,mx):
        for j in range(1,my):
            rho = q[0,i,j]
            u   = q[1,i,j] / q[0,i,j]
            v   = q[2,i,j] / q[0,i,j]
            et  = q[3,i,j]
            p   = pressure(rho, u, v, et)
            c   = np.sqrt(gamma*p/rho)
            mach[i,j] = u/c

            xloc = .25 * (x[i,  j] + x[i-1,  j] \
                        + x[i,j-1] + x[i-1,j-1])
            yloc = .25 * (y[i,  j] + y[i-1,  j] \
                        + y[i,j-1] + y[i-1,j-1])
            xc[i,j] = xloc
            yc[i,j] = yloc

    #xc   = xc[1:,1:]
    #yc   = yc[1:,1:]
    #mach = mach[1:,1:]
    return

# -----------------------------------------------------------------------------
def write_data():
    filename = 'data.txt'
    f = open(filename, 'w')
    for i in range(1, mx):
        for j in range(1, my):
            f.write(str(xc[i,j]) + ' ' + \
                    str(yc[i,j]) + ' ' + \
                    str(q[0,i,j]) + ' ' + \
                    str(q[1,i,j]) + ' ' + \
                    str(q[2,i,j]) + ' ' + \
                    str(q[3,i,j]) + ' ' + \
                    str(mach[i,j]) + '\n')
    f.close()
    return

# -----------------------------------------------------------------------------
# main program

# grid points
mx = 65; my = 17 
# mx = 129; my = 33

# parameters
lmax = 4
gamma = 1.4
alpha = 1       # axisymmetric flow
plot_interval = 1

cfl  = 1.5
visc = 3
eps  = 1e-6
tmax = 20000

# boundary conditions
ibcin  = 2
ibcout = 2

# read mesh file
filename = 'ft03.dat'
(x, y) = read_mesh(filename)
# plot_mesh()

(dni_x, dni_y, dni, dnj_x, dnj_y, dnj) = calc_normal()
area = calc_area()

# intial condtions
q = ic()

fig = plt.figure(figsize=(6,9))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312, aspect='equal')
ax3 = fig.add_subplot(313)

reslist = []
for t in range(0,tmax):
    stop = False
    rk4()
    if (stop): break

# write data to file
write_data()
