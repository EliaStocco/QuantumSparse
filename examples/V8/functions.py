import json

def get_couplings(S,folder,prefix):
    #global S
    J  = json.load(open("{:s}/{:s}.J.json".format(folder,prefix),"r"))
    DM = json.load(open("{:s}/{:s}.DM.json".format(folder,prefix),"r"))
    J2 = json.load(open("{:s}/{:s}.collinear.json".format(folder,prefix),"r"))
    factor = (S**2)/S*(S-0.5)
    #nn2 = 1#J2["J2"]/J2["J1"]
    
    couplings = {"Jt":J['tangential']['popt']['J'],\
                 "Jr":J['radial']    ['popt']['J'],\
                 "Jz":J['collinear'] ['popt']['J'],\
                 "Jt2":J2["J2"],\
                 "Jr2":J2["J2"],\
                 "Jz2":J2["J2"],\
                 "dt":DM['popt'][0],\
                 "dr":DM['popt'][1],\
                 "dz":DM['popt'][2],\
                 "D" :factor*J['collinear'] ['popt']['D'],\
                 "E" :factor*J['E-value']}
    return couplings
