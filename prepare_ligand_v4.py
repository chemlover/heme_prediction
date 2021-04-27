import os
import sys
import numpy as np
from math import * 
def vector(p1, p2):
    return [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]]

def vabs(a):
    return sqrt(pow(a[0],2)+pow(a[1],2)+pow(a[2],2))

def sigma_sq(sep):
        return pow((1+sep),0.15)*pow((1+sep), 0.15)

def calc_dist(p1, p2):
        v=vector(p1,p2)
        return vabs(v)

def vproduct(a, b):
    if type(a)==type([]) and type(b)==type([]):
        return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
    elif type(b)==type([]):
        return [a*b[0], a*b[1], a*b[2]]
    elif type(a)==type([]):
        return [a[0]*b, a[1]*b, a[2]*b]
    return a*b

def vcross_product(a, b):
    cx = a[1]*b[2]-a[2]*b[1]
    cy = a[2]*b[0]-a[0]*b[2]
    cz = a[0]*b[1]-a[1]*b[0]
    return [cx, cy, cz];

def vangle(a, b):
    return acos(vproduct(a, b)/(vabs(a)*vabs(b)))

def dihedral_angle(v1, v2, v3):
    n1 = vcross_product(v1, v2)
    n2 = vcross_product(v2, v3)
    y = vproduct( vproduct(vabs(v2), v1), n2 )
    x = vproduct( n1, n2 )
    return atan2(y, x)

def read_ligand(pdbfile):
    natom = 0
    ligand = []
    with open(pdbfile,"r") as fopen:
         for line in fopen.readlines():
             if len(line) > 66:
                if line[0:4] == "ATOM":
                   natom += 1
                elif line[0:6] == "HETATM":
                   x=float(line[30:38])
                   y=float(line[38:46])
                   z=float(line[46:54])
                   atom = [x,y,z]
                   ligand.append(atom) 
    return ligand,natom

def get_bond(ligand,natom):
    bond=[]
    nlatom=len(ligand)
    for i in range(nlatom):
        for j in range(i+1,nlatom):
            dist = calc_dist(ligand[i],ligand[j])     
            if dist < 2.2:
               bond.append([i,j,dist])
    data = "Atom1,Atom2,r\n"
    for i in range(len(bond)):
        data += str(bond[i][0]+natom)+","+str(bond[i][1]+natom)+","+ str(bond[i][2]/10.0)+"\n" 
    with open("bond.csv","w") as fwrite:
         fwrite.writelines(data)
    return bond

def get_angle(ligand,natom,bond):
    angle = []
    n_bond = len(bond)
    for i in range(n_bond):
        for j in range(i+1,n_bond):
            a1=bond[i][0]
            a2=bond[i][1]
            b1=bond[j][0] 
            b2=bond[j][1]
            if a1==b1 or a1==b2 or a2==b1 or a2==b2:
               #print bond[i],bond[j]
               if a1==b1:
                  c1=b2
                  c2=a2
                  c3=a1
               elif a1==b2:
                  c1=b1
                  c2=a2
                  c3=a1
               elif a2==b1:
                  c1=b2
                  c2=a1
                  c3=a2
               else:
                  c1=b1
                  c2=a1
                  c3=a2
               p1 = vector(ligand[c3],ligand[c2])
               p2 = vector(ligand[c3],ligand[c1])
               radangle = vangle(p1,p2) 
               angle.append([c2,c3,c1,radangle]) 
    data = "Atom1,Atom2,Atom3,theta\n"
    for i in range(len(angle)):
        data += str(angle[i][0]+natom)+","+str(angle[i][1]+natom)+","+ str(angle[i][2]+natom)+","+str(angle[i][3])+"\n"
    with open("angle.csv","w") as fwrite:
         fwrite.writelines(data)
    return angle    
def get_multpile(ligand):
    nlatom=len(ligand)
    n_mult = np.zeros(nlatom)
    for i in range(nlatom):
        for j in range(nlatom):
            dist = calc_dist(ligand[i],ligand[j])
            if dist < 2.1:
               n_mult[i]+=1
    for i in range(nlatom):
        n_mult[i] =n_mult[i]-1
    print n_mult
    return n_mult  
def get_dihedral(ligand,natom,bond,n_mult):
    dihedral_angle = []
    n_bond = len(bond)
    for i in range(n_bond):
         for j in range(i+1,n_bond):
            a1=bond[i][0]
            a2=bond[i][1]
            b1=bond[j][0]
            b2=bond[j][1]
            if a1!=b1 and a1!=b2 and a2!=b1 and a2!=b2:
               dist1 = calc_dist(ligand[a1],ligand[b1])
               dist2 = calc_dist(ligand[a2],ligand[b1])
               dist3 = calc_dist(ligand[a1],ligand[b2])
               dist4 = calc_dist(ligand[a2],ligand[b2])
               if dist1 < 2.1 or dist2 < 2.1 or dist3 < 2.1 or dist4 < 2.1:
                  if dist1 < 2.1:
                     c1=a2
                     c2=a1
                     c3=b1
                     c4=b2
                  elif dist2 < 2.1:
                     c1=a1
                     c2=a2
                     c3=b1
                     c4=b2
                  elif dist3 < 2.1:
                     c1=a2
                     c2=a1
                     c3=b2
                     c4=b1
                  elif dist4 < 2.1:
                     c1=a1
                     c2=a2
                     c3=b2
                     c4=b1
                  v1 = vector(ligand[c1],ligand[c2])
                  v2 = vector(ligand[c2],ligand[c3])
                  v3 = vector(ligand[c3],ligand[c4])
                  #print v1,v2,v3
                  #print c1,c2,c3,c4
                  #print a1,a2,b1,b2
                  n1 =  vcross_product(v1, v2)
                  n2 = vcross_product(v2, v3)
                  y = vproduct( vproduct(vabs(v2), v1), n2 )
                  x = vproduct( n1, n2 )
                  raddihedral = atan2(y, x)
                  #raddihedral = vangle(v1,v2)
                  #print n1,n2,y,x
                  #raddihedral = dihedral_angle(v1,v2,v3) 
                  mult = n_mult[c1] +  n_mult[c2] + n_mult[c3] + n_mult[c4] 
                  if mult % 2 == 0:
                     sigm = 0
                  else:
                     sigm = 3.14159/2
                  dihedral_angle.append([c1,c2,c3,c4,raddihedral,mult,sigm])
    data = "Atom1,Atom2,Atom3,Atom4,theta2d,mult,sigm\n"
    for i in range(len(dihedral_angle)):
        data += str( dihedral_angle[i][0]+natom)+","+str( dihedral_angle[i][1]+natom)+","+ str( dihedral_angle[i][2]+natom)+","+str( dihedral_angle[i][3]+natom)+","+str( dihedral_angle[i][4])+","+str( dihedral_angle[i][5])+","+str( dihedral_angle[i][6])+"\n"
    with open("dihedral_angle.csv","w") as fwrite:
         fwrite.writelines(data)
    return dihedral_angle

def vdw_params(pdbfile,natom):
    re=[]
    with open(pdbfile,"r") as fopen:
         for line in fopen.readlines():
             if len(line) > 66:
                if line[0:6] == "HETATM" and (line[17:20] == "HEM" or line[17:20] == "HEC") :
                   atom = line[12:16]
                   elem = atom.split()[0][0:1]
                   if elem == "C":
                      re.append([3.6,0.06])
                   elif elem == "O":
                      re.append([3.2,0.20]) 
                   elif elem == "N":
                      re.append([3.7,0.12])
                   elif elem == "F":
                      re.append([2.6,0.2])
                   else:
                      print "error" 
    data = "Atom1,r,e\n"
    for i in range(len(re)):
        data +=  str(i+natom)+","+str(re[i][0])+","+ str(re[i][1])+ "\n"
    with open("vdw.csv","w") as fwrite:
         fwrite.writelines(data)
    return re
import imp
import subprocess
import glob
import re
def charge_params(pdbfile,natom):
    charge=[]
    #code = {"-0.12" :"CAA","-0.16" :"CAB", "-0.16" :"CAC",}
    #inv_code_map = {v: k for k, v in code.items()}
    inv_code_map = {"CAA" :"-0.12" ,"CAB":"-0.16" ,"CAC": "-0.16","CAD": "-0.12" , "NA":
            "-0.76" , "CBA": "-0.12" , "CBB": "0.00" , "CBC": "0.00",  "CBD":
            "-0.12" , "NB": "-0.76" , "CGA": "0.72" , "CGD": "0.72" , "ND":
            "-0.76" , "CHA": "-0.19" , "CHB": "-0.19" , "CHC": "-0.19" , "CHD":
            "-0.19" , "CMA": "-0.1" , "CMB": "-0.1" , "CMC": "-0.1" , "CMD":
            "-0.1" , "C1A": "0.32" , "C1B": "0.32" , "C1C": "0.32" , "C1D":
            "0.32" , "O1A": "-0.74" , "O1D": "-0.74" , "C2A": "0.07" , "C2B":
            "0.07" , "C2C": "0.07" , "C2D": "0.07" , "O2A": "-0.74" , "O2D":
            "-0.74", "C3A": "0.07" , "C3B": "0.07" , "C3C": "0.07" , "C3D":
            "0.07" , "C4A": "0.32" , "C4B": "0.32" , "C4C": "0.32" , "C4D":
            "0.32" , "NC": "-0.76" , "FE": "1.20","CA": "2.0","FE2": "2.0","KK": "0.0","CYN":"0.0"}
    print inv_code_map
    with open(pdbfile,"r") as fopen:
         for line in fopen.readlines():
             if len(line) > 66:
                if line[0:6] == "HETATM" and ( line[17:20] == "HEM" or line[17:20] == "HEC") :
                   atom = line[12:16].split()[0]
                   charge.append(inv_code_map[atom])
    data = "Atom1,q1\n"
    for i in range(len(charge)):
        data +=  str(i+natom)+","+str(float(charge[i]))+ "\n"
    with open("charge.csv","w") as fwrite:
         fwrite.writelines(data)
    return charge

def main():
    pdbfile = sys.argv[1]
    ligand,natom = read_ligand(pdbfile)
    bond = get_bond(ligand,natom)             
    angle = get_angle(ligand,natom,bond)
    n_mult = get_multpile(ligand)
    get_dihedral(ligand,natom,bond,n_mult)
    re = vdw_params(pdbfile,natom)
    charge = charge_params(pdbfile,natom)
if __name__ == '__main__':
    main()
 #code = {"-0.12" : "CAA", "-0.16" : "CAB", "-0.16" : "CAC", "-0.12" : "CAD",
 #           "-0.76" : "NA", "-0.12" : "CBA", "0.00" : "CBB", "0.00" : "CBC",
 #           "-0.12" : "CBD", "-0.76" : "NB", "0.72" : "CGA", "0.72" : "CGD",
    #        "-0.76" : "ND", "-0.19" : "CHA", "-0.19" : "CHB", "-0.19" : "CHC",
    #        "-0.19" : "CHD", "-0.1" : "CMA", "-0.1" : "CMB", "-0.1" : "CMC",
    #        "-0.1" : "CMD", "0.32" : "C1A", "0.32" : "C1B", "0.32" : "C1C",
    #        "0.32" : "C1D", "-0.74" : "O1A", "-0.74" : "O1D", "0.07" : "C2A",
    #        "0.07" : "C2B", "0.07" : "C2C", "0.07" : "C2D", "-0.74" : "O2A",
    #        "-0.74": "O2D", "0.07" : "C3A", "0.07" : "C3B", "0.07" : "C3C",
    #        "0.07" : "C3D", "0.32" : "C4A", "0.32" : "C4B", "0.32" : "C4C",
    #        "0.32" : "C4D", "-0.76" : "NC", "1.20" : "FE"}
      
