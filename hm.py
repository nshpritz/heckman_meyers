import math
import pandas as pd
import numpy as np
import os
import sys
import cmath

def runall():
    
    if len(sys.argv) != 3:
        print("oops")
        exit()
    file = sys.argv[1]
    varb = float(sys.argv[2])
    varb = varb if varb > 1e-7 else 1e-7
    varb = varb if varb < 1 else 1-1e-7

    df_lines = pd.read_excel(file, sheet_name="Lines")
    df_distributions = pd.read_excel(file, sheet_name="Distributions")
    df_output = pd.read_excel(file, sheet_name="OutputPoints")

    # Test that all lines and distributions can work together
    #print(df_lines)
    #print(df_distributions)
    #print(df_output)

    lines_lines = set(df_lines.Line.to_list())
    lines_dist = set(df_distributions.Line.to_list())

    if len(lines_lines) != len(lines_dist):
        print("Lines contains a different number of lines of business than the lines in distributions")
        exit()
    
    for line in lines_lines:
        if line not in lines_dist:
            print("Line in list of lines but not present as a severity distribution")
            exit(0)
    
    for line in lines_dist:
        if line not in lines_lines:
            print("Line has a severity distribution but but not present in the main list")
            exit(0)

    # Set up the vectors and matrices
    longest = 0
    for line in lines_dist:
        i = int(df_distributions[df_distributions.Line == line]["Line"].count())
        if i > longest:
            longest = i
    
    # Beginning of all of the off by 1 issues = these should all be 1 indexed arrays
    # However the array dimensions for gaussian quadrature are NOT 1 indexed
    longest += 2   # don't like this at all
    numlines = len(lines_dist) + 1 # same with this
   
    amt = np.zeros((longest, numlines+2), dtype=np.float64)
    pk = np.zeros((longest, numlines+2), dtype=np.float64)
    xlam = np.zeros(numlines+2, dtype=np.float64)
    sigp = np.zeros(numlines+2, dtype=np.float64)
    npts = np.zeros(numlines+2 , dtype=np.float64)
    a = np.zeros(257, dtype=np.float64)
    t = np.zeros((256, 5), dtype=np.float64)
    f = np.ones((256, 5), dtype=np.float64)
    g = np.zeros((256, 5), dtype=np.float64)

    # Begin computing the severity distribution
    # Initialize Variables

    sigsqa = 0.0
    xmua = 0.0
    
    # now start collecting the data for the program itself
    name = df_lines.Name.to_list()
    temp = df_lines.Contagion.to_list()  # had been [0] + np.maximum(1e7, df_lines.Contagion).to_list()
    varc = [0] + [1e-7 if (temp[i] >=0 and temp[i] < 1e-7) else temp[i] for i in range(len(temp))]
    exloss = [0] + df_lines.ExpectedLoss.to_list()              # had been [0] + df_lines.ExpectedLoss.to_list()
    for i in lines_lines:
        npts[i] = df_distributions[df_distributions.Line == i]["Line"].count() + 1
        #print(i, npts[i])
    
    # Compute the discrete probability mass function
    for line in range(1, numlines):        
        x1 = 0.0
        x2 = 0.0
        npt = int(npts[line])
        df_line = df_distributions[df_distributions.Line == line]
        cumprobability = [0] + df_line['Cumprob'].to_list()
        levels = [0] + df_line['Value'].to_list()
        amt[0][line] = 0
            
        for i in range(1,len(levels)):
            prob = cumprobability[i] - cumprobability[i-1]
            amt[i][line] = levels[i]
            pk[i-1][line] = prob / (amt[i][line] - amt[i-1][line]) if amt[i][line]-amt[i-1][line] >= 1 else 10e7    # note index is one less than on amt - this is a between point probability not sure about the new indexes here
            x1 += prob * (amt[i][line] + amt[i-1][line])/2.0
            x2 += prob * (amt[i][line]**2 + amt[i-1][line]**2 + amt[i][line] * amt[i-1][line]) / 3.0
            #print(i-1, pk[i-1][line], prob, amt[i-1][line], amt[i][line], amt[i][line]-amt[i-1][line])
        k = len(levels)    
        prob = 1 - cumprobability[i]
        x1 += prob * amt[k-1][line]       # index was npt-1
        x2 += prob * amt[k-1][line]**2  # index was npt-1
        pk[k-1][line] = prob
        print("check for x2", npt-1, pk[npt-1][line], amt[npt-2][line], amt[npt-1][line])
        sigs = x2 - x1**2
        xlam[line] = exloss[line] / x1
        sigp[line] = math.sqrt(xlam[line] + varc[line] * xlam[line]**2)
        sigsqa += xlam[line] * sigs + (x1 * sigp[line])**2
        sigs = math.sqrt(sigs)
        xmua += x1*xlam[line]
        print(line, x1, sigs)

    sigsqa = varb * xmua**2 + sigsqa*(1+varb)
    print(varb, xmua, sigsqa)
    siga = math.sqrt(sigsqa)
    print('siga = ', siga)

    # load the output points and compute the step size

    if len(df_output.Value.dropna()) != 0:
        df_output.EntryRatio = df_output.Value / xmua
    else:
        df_output.Value = df_output.EntryRatio * xmua
    
    xmax = int(df_output.Value.max())
    h = 2 * math.pi * siga / xmax
    print(h, siga ,  xmua)

    # fill the step array
    a[1] = 0
    for i in range(1,6):
        a[i+1] = h / 2**(5-i)
        gauss5(i, a, t, f, g, npts, amt, pk, xlam, varc, siga, numlines)  # could be numlines-1 - will have to check
    
    # start the computations
    for i in range(6, 256):
        a[i+1] = a[i] + h
        gauss5(i, a, t, f, g, npts, amt, pk, xlam, varc, siga, numlines)  # could be numlines-1 - will have to check
        e = 0.0
        for j in range(5):
            e = max(e, f[i][j]/t[i][j])
            #print(a[i], f[i][j], t[i][j])
        if e < 2e-5:
            break
    print("h = ", h, "i = ", i)
    nint = i

    for i in range(1, numlines):
        iex = math.floor(exloss[i]+0.5)
        print(f'{i}  {iex}   {name[i-1]}   {varc[i]}    {xlam[i]}    {sigp[i]}')

    values = [int(x) for x in df_output.Value.to_list()]
    for value in values:
        pct, epp = pctepp(value, varb, xmua, siga, a, t, f, g, nint)
        print(value, value/xmua, pct, epp, epp/xmua)

    

    return

def gauss5(i, a, t, f, g, npts, amt, pk, xlam, varc, siga, nl):
    s = [-0.90617985, -0.53846931, 0.0, 0.53846931, 0.90617985]
    for j in range(5):
        t[i][j] = 0.5 *(a[i+1]-a[i]) * s[j] + 0.5 * (a[i+1] + a[i])
        ts = t[i][j] / siga

        for k in range(1,nl):
            fl, gl = aggchr(npts, amt, pk, k, xlam, varc, ts)
            f[i][j] *= fl
            g[i][j] += gl

def aggchr(npts, amt, pk, k, xlam, varc, t):
    xh, xk = sevchr(npts, amt, pk, k, t)
    a  = 1 - varc[k] * xlam[k] * xh
    b = -varc[k] * xlam[k] * xk
    #f = math.exp(-0.5 * math.log(a*a + b*b)/varc[k])    # was this
    #f = math.exp(-math.log(a*a + b*b)/varc[k])
    #f = np.sqrt(a*a + b*b)
    #f = math.exp(a)
    #f = math.exp(-math.sqrt(1-2*xlam[k]*xh + xlam[k]**2 *  varc[k]**2 *(xh*xh +xk*xk))/varc[k])
    #c = math.atan2(b,a)
    #f, g = cmath.polar(-a/varc[k] - b/varc[k]j)    
    im = cmath.sqrt(-1)
    z1 = (a + b*im ) 
    #z2 = -cmath.log(z1) / varc[k]
    f = abs(z1) **(-1/varc[k])
    g = -cmath.phase(z1) / varc[k]
    
    ''''''
    if a==0:
        if b>0:
            c=np.pi / 2
        else:
            c=-np.pi/2
    else:
        c = np.atan(b/a)
        if a<0 and b>=0:
            c+=np.pi
        elif a<0 and b<=0:
            c-=np.pi
    ''''''
    #g =  -c / varc[k]

    return f, g

def sevchr(npts, amt, pk, k, t):
    s2 = 0.0
    c2 = 1.0
    th = 0.0
    tk = 0.0

    npt = int(npts[k])
    for l in range(1, npt):
        a = amt[l][k]*t
        s1 = s2
        c1 = c2
        s2 = math.sin(a)
        c2 = math.cos(a)
        th += pk[l-1][k] * (s2-s1)
        tk += pk[l-1][k] * (c1-c2)
    
    xh = th/t - 1 + pk[npt][k] * c2
    xk = tk/t + pk[npt][k] * s2

    return xh, xk

def pctepp(x, varb, xmua, siga, a, t, f, g, nint):
    w = [0.23692689, 0.4786867, 0.568889, 0.47862867, 0.23692689]

    epp = 0.0
    pct = 0.0
    r = 1 + 1/varb
    for i in range(1, nint):
        p1 = 0.0
        p2 = 0.0
        for j in range(5):
            xp1 = 1 + (x * t[i][j]/(siga * r))**2
            aa = siga * r
            b = x*t[i][j]
            atx = math.atan2(b, aa)
            p1 += w[j] * f[i][j]*xp1**(0.5*(-1-r)) * math.sin((1+r)*atx-g[i][j])/t[i][j]
            p2 += w[j] * f[i][j]*(math.cos(g[i][j])-xp1**(-0.5*r)*math.cos(r*atx-g[i][j]))/t[i][j]**2

        pct += (a[i+1]-a[i]) * p1/2
        epp += (a[i+1]-a[i]) * p2/2

    pct = 0.5 + pct/math.pi
    epp = xmua - x/2 + epp * siga/math.pi

    return pct, epp

if __name__ == '__main__':
    runall()






