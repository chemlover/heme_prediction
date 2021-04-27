import os
import sys
from simtk.unit import angstrom
from simtk.unit import kilocalorie_per_mole
import pandas as pd
try:
    OPENAWSEM_LOCATION = os.environ["OPENAWSEM_LOCATION"]
    sys.path.append(OPENAWSEM_LOCATION)
    # print(OPENAWSEM_LOCATION)
except KeyError:
    print("Please set the environment variable name OPENAWSEM_LOCATION.\n Example: export OPENAWSEM_LOCATION='YOUR_OPENAWSEM_LOCATION'")
    exit()

from openmmawsem import *
from helperFunctions.myFunctions import *

def group_constraint_by_distance(oa, d0=0*angstrom, group1=1, group2=2, forceGroup=2, k=10*kilocalorie_per_mole):
    # CustomCentroidBondForce only work with CUDA not OpenCL.
    # only CA, CB, O has mass. so the group have to include those.
    nres = oa.nres
    print(nres, oa.chain_starts)
    chainABCD = [oa.ca[i] for i in range(nres-1)]
    #chainE = [oa.ca[i] for i in range(nres-174, nres)]
    startid,endid = get_ligand_atoms("./start-openmmawsem.pdb")
    ligand = [i for i in range(startid-1,endid)]
    print (startid,endid)
    group1 = chainABCD
    group2 = ligand
    r_cutoff = 3
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    k_constraint = k * oa.k_awsem
    d0 = d0.value_in_unit(nanometer)   # convert to nm
    constraint = CustomCentroidBondForce(2, f"0.5*{k_constraint}*(1-step({r_cutoff}-distance(g1,g2)))*(distance(g1,g2)-{d0})^2")
    # example group set up group1=[oa.ca[7], oa.cb[7]] use the ca and cb of residue 8.
    constraint.addGroup(group1)    # group use particle index.
    constraint.addGroup(group2)
    constraint.addBond([0, 1])
    constraint.setForceGroup(forceGroup)
    return constraint


def small_molecular_bond_term(oa, k=462750.4, bond_info_csv="bond.csv", forceGroup=4):
    #k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    smallMolecularBond = HarmonicBondForce()
    data = pd.read_csv(bond_info_csv)
    for i, line in data.iterrows():
        atom1 = int(line["Atom1"])
        atom2 = int(line["Atom2"])
        r = float(line["r"])
        smallMolecularBond.addBond(atom1, atom2, r, k)
    # for res in oa.ligand_res_list:
    #     for bond in res.bonds():
    #         bond_type = f"{bond.atom1.element.symbol}-{bond.atom2.element.symbol}"
    #         bond_length_dic = {"C-C":0.154, "C-O":0.143}
    #         bond_length = bond_length_dic[bond_type]
    #         # print(bond, bond_type, bond_length)
    #         smallMolecularBond.addBond(bond.atom1.index, bond.atom2.index, bond_length, k)
    smallMolecularBond.setForceGroup(forceGroup)   # start with 11, so that first 10 leave for user defined force.
    return smallMolecularBond

def small_molecular_angle_term(oa, k=836.8, angle_info_csv="angle.csv", forceGroup=4):
    smallMolecularAngle = HarmonicAngleForce()
    data = pd.read_csv(angle_info_csv)
    for i, line in data.iterrows():
        atom1 = int(line["Atom1"])
        atom2 = int(line["Atom2"])
        atom3 = int(line["Atom3"])
        theta = float(line["theta"])
        smallMolecularAngle.addAngle(atom1, atom2,atom3, theta, k)
    smallMolecularAngle.setForceGroup(forceGroup)   # start with 11, so that first 10 leave for user defined force.
    return smallMolecularAngle

def small_molecular_dihedral_angle_term(oa, k=4.6024/200.0/4.814, dihedral_angle_info_csv="dihedral_angle.csv", forceGroup=4): 
    data = pd.read_csv(dihedral_angle_info_csv)
    n_atom_ligand = 0
    for i, line in data.iterrows():
        n_atom_ligand += 1
    #dihedral_angle_function = ''.join(["0.5*(1+cos(min(abs(psi%d-dihe(0,%d)),6.28-abs(psi%d-dihe(0,%d)))))+" \
    #                            % (i, i,i,i) for i in range(n_atom_ligand)])[:-1]
    dihedral_angle_function = ''.join(["0.5*(1+ cos((2*psi%d)-3.1415926))+" \
                                % (i) for i in range(n_atom_ligand)])[:-1]

    dihedral_angle_function = f'{k}*(' + dihedral_angle_function + ");"
    dihedral_angle_parameters = ''.join([f"psi{i}=dihedral(p1, p2, p3, p4);"\
                            for i in range(n_atom_ligand)])
    dihedral_angle_string = dihedral_angle_function+dihedral_angle_parameters
    dihedral_angleSS = CustomCompoundBondForce(4, dihedral_angle_string)
    #print (dihedral_angle_string)
    #for i, line in data.iterrows():
    #    n_mult = int(line['mult'])
    #    theta = float(line['sigm'])
    #    print (i)
       # dihedral_angleSS.addGlobalParameter(f"multi{i}", n_mult)
       # dihedral_angleSS.addGlobalParameter(f"theta{i}", theta)
    for i, line in data.iterrows():
        atom1 = int(line["Atom1"])
        atom2 = int(line["Atom2"])
        atom3 = int(line["Atom3"])
        atom4 = int(line["Atom4"])
        dihedral_angleSS.addBond([atom1,atom2,atom3,atom4])
    #ne = np.loadtxt("./dihe")
    #dihedral_angleSS.addTabulatedFunction("dihe", Discrete2DFunction(1, n_atom_ligand, ne.flatten()))
    dihedral_angleSS.setForceGroup(forceGroup)
    return dihedral_angleSS 

def small_molecular_vdw_term(oa, k=4.184, vdw_info_csv="vdw.csv", periodic=False, forceGroup=5):
    data = pd.read_csv(vdw_info_csv)
    smallMolecular_vdw = CustomNonbondedForce(f"4*1.2*(1/800)*{k}*step(({0.32}/r)^12-({0.32}/r)^6)")
    for ii,line in data.iterrows():
        num = int(line['Atom1']) + 1
    #for i in range(num):
    #     smallMolecular_vdw.addParticle()
    for i in range(oa.natoms):
        smallMolecular_vdw.addParticle()
    for ii,line in data.iterrows(): 
        i = int(line['Atom1'])
        for  jj,line in data.iterrows():
             j = int(line['Atom1'])
             if j > i:
                smallMolecular_vdw.addInteractionGroup([i],[j])
    smallMolecular_vdw.setCutoffDistance(1.00)
    #if periodic:
    #    smallMolecular_vdw.setNonbondedMethod(smallMolecular_vdw.CutoffPeriodic)
    #else:
    #    smallMolecular_vdw.setNonbondedMethod(smallMolecular_vdw.CutoffNonPeriodic)
    smallMolecular_vdw.createExclusionsFromBonds(oa.bonds, 1)
    smallMolecular_vdw.setForceGroup(forceGroup) 
    return smallMolecular_vdw

def small_molecular_debye_huckel_term(self, k_dh=4.15*4.184, forceGroup=5, screening_length=1.0, charge_info_csv="charge.csv"):
        # screening_length (in the unit of nanometers)
        print("Debye Huckel term is ON")
        k_dh *= self.k_awsem*0.1
        k_screening = 1.0
        # screening_length = 1.0  # (in the unit of nanometers)
        min_seq_sep = 1
        smallMolecular_dh = CustomBondForce(f"{k_dh}*charge_i*charge_j/r*exp(-{k_screening}*r/{screening_length})")
        smallMolecular_dh.addPerBondParameter("charge_i")
        smallMolecular_dh.addPerBondParameter("charge_j")
        structure_interactions_dh = []
        data = pd.read_csv(charge_info_csv)
        for ii,line in data.iterrows():
               i = int(line['Atom1'])
               charge_i = float(line['q1'])
               for  jj,line in data.iterrows():
                    j = int(line['Atom1'])
                    charge_j = float(line['q1'])
                    if j > i and charge_i*charge_j!=0.0:
                        structure_interactions_dh.append([i, j, [charge_i, charge_j]])
        for structure_interaction_dh in structure_interactions_dh:
            smallMolecular_dh.addBond(*structure_interaction_dh)
        smallMolecular_dh.setForceGroup(forceGroup)
        return smallMolecular_dh

def small_molecular_exclude_volume_term(oa, k=5 * kilocalorie_per_mole, periodic=False, forceGroup=5):
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    r_excl = 0.30
    smallMolecular_excludeVolume = CustomNonbondedForce(f"{k}*step({r_excl}-r)*(r-{r_excl})^2")
    for i in range(oa.natoms):
        smallMolecular_excludeVolume.addParticle()
    # print(oa.ca)
    # print(oa.bonds)
    # print(oa.cb)
    ligand_atoms_index_list = [atom.index for atom in oa.ligand_atom_list]
    protein_atoms_index_list = [atom.index for atom in oa.protein_atom_list]
    smallMolecular_excludeVolume.addInteractionGroup(ligand_atoms_index_list, protein_atoms_index_list)

    smallMolecular_excludeVolume.setCutoffDistance(1.00)
    if periodic:
        smallMolecular_excludeVolume.setNonbondedMethod(smallMolecular_excludeVolume.CutoffPeriodic)
    else:
        smallMolecular_excludeVolume.setNonbondedMethod(smallMolecular_excludeVolume.CutoffNonPeriodic)
    smallMolecular_excludeVolume.createExclusionsFromBonds(oa.bonds, 1)
    smallMolecular_excludeVolume.setForceGroup(forceGroup)   # start with 11, so that first 10 leave for user defined force.

    return smallMolecular_excludeVolume


def protein_small_molecular_debye_huckel_term(self, k_dh=4.15*4.184, forceGroup=6, screening_length=1.0, charge_info_csv="charge.csv"):
        # screening_length (in the unit of nanometers)
        print("Debye Huckel term is ON")
        k_dh *= self.k_awsem*0.1
        k_screening = 1.0
        # screening_length = 1.0  # (in the unit of nanometers)
        min_seq_sep = 1
        smallMolecular_dh = CustomBondForce(f"{k_dh}*charge_i*charge_j/r*exp(-{k_screening}*r/{screening_length})*0.5*(1+tanh((r-0.6)*20))")
        smallMolecular_dh.addPerBondParameter("charge_i")
        smallMolecular_dh.addPerBondParameter("charge_j")
        structure_interactions_dh = []
        data = pd.read_csv(charge_info_csv)
        for ii,line in data.iterrows():
            i = int(line['Atom1'])
            charge_i = float(line['q1'])
            for j in range(self.nres):
                charge_j = 0.0
                if self.seq[j] == "R" or self.seq[j]=="K":
                        charge_j = 1.0
                if self.seq[j] == "D" or self.seq[j]=="E":
                        charge_j = -1.0
                if charge_i*charge_j!=0.0:
                   structure_interactions_dh.append([i, self.cb[j], [charge_i, charge_j]])
        for structure_interaction_dh in structure_interactions_dh:
            smallMolecular_dh.addBond(*structure_interaction_dh)
        smallMolecular_dh.setForceGroup(forceGroup)
        return smallMolecular_dh

def get_ligand_O_atoms(pdbfile):
    n_line = 0
    id_Oatoms = []
    with open(pdbfile,"r") as fopen:
         for line in fopen.readlines():
             n_line += 1
             if line[13:14] == "O" and line[0:6] == "HETATM":
                id_Oatoms.append(n_line)
    return id_Oatoms

def get_ligand_atoms(pdbfile):
    n_line = 0
    startid = 0
    with open(pdbfile,"r") as fopen:
         for line in fopen.readlines():
             n_line += 1
             if line[0:6]== "HETATM":
                startid = n_line
                break
    n_line = 0
    endid = 0
    with open(pdbfile,"r") as fopen:
          for line in fopen.readlines():
             n_line += 1
             if line[0:6]== "HETATM":
                endid = n_line
    return startid,endid

def get_ligand_N_atoms(pdbfile):
    n_line = 0
    id_Natoms = []
    with open(pdbfile,"r") as fopen:
         for line in fopen.readlines():
             n_line += 1
             if line[13:14] == "N" and line[0:6] == "HETATM":
                id_Natoms.append(n_line)
    return id_Natoms

def get_ligand_Fe_atoms(pdbfile):
    n_line = 0
    with open(pdbfile,"r") as fopen:
         for line in fopen.readlines():
             n_line += 1
             if line[12:14] == "FE" and line[0:6] == "HETATM":
                Fe_atomid = n_line
                break
    return n_line

def get_ligand_S_atoms(pdbfile):
    n_line = 0
    id_Satoms = []
    with open(pdbfile,"r") as fopen:
         for line in fopen.readlines():
             n_line += 1
             if (line[13:16] == "CAC" or line[13:16] == "CAB" )and line[0:6] == "HETATM":
                id_Satoms.append(n_line)
    return id_Satoms

def small_molecular_interaction_to_proteins_HEC_CS(oa, k=23 * kilocalorie_per_mole,forceGroup=20):
    Satomid = get_ligand_S_atoms("./start-openmmawsem.pdb")
    nres = oa.nres
    print ("!!!!!")
    print (Satomid)
    print (len(oa.h),len(oa.n))
    S1r = Satomid[0] - 1
    S2r = Satomid[1] - 1
    k = k.value_in_unit(kilojoule_per_mole)/2
    LP_function = ''.join(["exp(-(r1%d-0.28)*(r1%d-0.28)*4)*0.7*(1-tanh((r1%d-0.28)*3))*(1+tanh((r1%d-0.28)*32))+ exp(-(r2%d-0.28)*(r2%d-0.28)*4)*0.7*(1-tanh((r2%d-0.28)*3))*(1+tanh((r2%d-0.28)*32)) +" \
                            % (i,i,i, i, i, i, i,i) for i in range(1)])[:-1]
    LP_function = f'-{k}*(' + LP_function + ");"
    LP_parameters = ''.join([f"r1{i}=distance(p1, p3);\
                                 r2{i}=distance(p2, p4);"\
                            for i in range(1)])
    LP_string = LP_function+LP_parameters
    LPSS = CustomCompoundBondForce(4, LP_string)
    #LPSS.addPerBondParameter("kSS")
    for i in range(oa.nres-3):
        if i not in oa.chain_starts and i not in oa.chain_ends and not oa.res_type[i] == "IGL" and not oa.res_type == "IPR":
           if oa.h[i] > 0 and oa.n[i] > 0:
              if oa.seq[i] == "C" and oa.seq[i+3] == "C":
                 print ("C C+3 !")
                 LPSS.addBond([oa.cb[i],oa.cb[i+3],S1r,S2r])
    LPSS.setForceGroup(forceGroup)
    return LPSS

def small_molecular_interaction_to_proteins_hb(oa, k=1.9 * kilocalorie_per_mole,forceGroup=7):
    Oatomid = get_ligand_O_atoms("./start-openmmawsem.pdb")
    nres = oa.nres
    print ("!!!!!")
    print (Oatomid)
    print (len(oa.h),len(oa.n))
    O1r = Oatomid[0] - 1
    O2r = Oatomid[1] - 1
    O3r = Oatomid[2] - 1
    O4r = Oatomid[3] - 1
    k = k.value_in_unit(kilojoule_per_mole)/2
    LP_function = ''.join(["exp(-(r1%d-0.16)*(r1%d-0.16)*4)*(tanh(4*(O1%d-1.57)-6)+1)/2.0/0.65 + exp(-(r2%d-0.16)*(r2%d-0.16)*4)*(tanh(4*(O2%d-1.57)-6)+1)/2.0/0.65 + exp(-(r3%d-0.16)*(r3%d-0.16)*4)*(tanh(4*(O3%d-1.57)-6)+1)/2.0/0.65 + exp(-(r4%d-0.16)*(r4%d-0.16)*4)*(tanh(4*(O4%d-1.57)-6)+1)/2.0/0.65+" \
                            % (i,i,i, i, i, i, i , i, i,i, i,i) for i in range(1)])[:-1]
    LP_function = f'-{k}*(' + LP_function + ");"
    LP_parameters = ''.join([f"r1{i}=distance(p1, p2);\
                                 r2{i}=distance(p1, p3);\
                                 r3{i}=distance(p1, p4);\
                                 r4{i}=distance(p1, p5);\
                                 O1{i}=angle(p2, p1, p6);\
                                 O2{i}=angle(p3, p1, p6);\
                                 O3{i}=angle(p4, p1, p6);\
                                 O4{i}=angle(p5, p1, p6);"\
                            for i in range(1)])
    LP_string = LP_function+LP_parameters
    LPSS = CustomCompoundBondForce(6, LP_string)
    #LPSS.addPerBondParameter("kSS")
    for i in range(oa.nres):
        if i not in oa.chain_starts and i not in oa.chain_ends and not oa.res_type[i] == "IGL" and not oa.res_type == "IPR":
           if oa.h[i] > 0 and oa.n[i] > 0:
              LPSS.addBond([oa.h[i],O1r,O2r,O3r,O4r,oa.n[i]])
    LPSS.setForceGroup(forceGroup)
    return LPSS



def small_molecular_interaction_to_proteins_hb2(oa, k=1.9 * kilocalorie_per_mole,forceGroup=8):
    Oatomid = get_ligand_O_atoms("./start-openmmawsem.pdb")
    nres = oa.nres
    print ("!!!!!")
    print (Oatomid)
    print (len(oa.h),len(oa.n))
    O1r = Oatomid[0] - 1
    O2r = Oatomid[1] - 1
    O3r = Oatomid[2] - 1
    O4r = Oatomid[3] - 1
    k = k.value_in_unit(kilojoule_per_mole)/4/2
    LP_function = ''.join(["kresID*(exp(-(r1%d-rresID)*(r1%d-rresID)*4)*0.5*(1-tanh((r1%d-rresID+0.1)*32))*0.5*(1+tanh((r1%d-rresID-0.1)*32)) + exp(-(r2%d-rresID)*(r2%d-rresID)*4)*0.5*(1-tanh((r2%d-rresID+0.1)*32))*0.5*(1+tanh((r1%d-rresID-0.1)*32)) + exp(-(r3%d-rresID)*(r3%d-rresID+0.1)*4)*0.5*(1-tanh((r3%d-rresID+0.1)*32))*0.5*(1+tanh((r1%d-rresID-0.1)*32)) + exp(-(r4%d-rresID)*(r4%d-rresID)*4)*0.5*(1-tanh((r4%d-rresID+0.1)*32)))*0.5*(1+tanh((r1%d-rresID-0.1)*32))+" \
                            % (i,i,i, i, i, i, i , i,i,i,i,i,i,i,i,i) for i in range(1)])[:-1]
    LP_function = f'-{k}*(' + LP_function + ");"
    LP_parameters = ''.join([f"r1{i}=distance(p1, p2);\
                                 r2{i}=distance(p1, p3);\
                                 r3{i}=distance(p1, p4);\
                                 r4{i}=distance(p1, p5);"\
                            for i in range(1)])
    LP_string = LP_function+LP_parameters
    LPSS = CustomCompoundBondForce(5, LP_string)
    LPSS.addPerBondParameter("kresID")
    LPSS.addPerBondParameter("rresID")
    for i in range(oa.nres):
        if i not in oa.chain_starts and i not in oa.chain_ends and not oa.res_type[i] == "IGL" and not oa.res_type == "IPR":
           if oa.seq[i] == "Y":
              LPSS.addBond([oa.cb[i],O1r,O2r,O3r,O4r],[1.2,0.96])
           elif oa.seq[i] == "W":
              LPSS.addBond([oa.cb[i],O1r,O2r,O3r,O4r],[0.4,0.56])
           elif oa.seq[i] == "K":
              LPSS.addBond([oa.cb[i],O1r,O2r,O3r,O4r],[1.0,0.96])
           elif oa.seq[i] == "R":
              LPSS.addBond([oa.cb[i],O1r,O2r,O3r,O4r],[0.8,1.06])
           elif oa.seq[i] == "H":
              LPSS.addBond([oa.cb[i],O1r,O2r,O3r,O4r],[0.8,0.76])
           elif oa.seq[i] == "D":
              LPSS.addBond([oa.cb[i],O1r,O2r,O3r,O4r],[1.4,0.76])
           elif oa.seq[i] == "E":
              LPSS.addBond([oa.cb[i],O1r,O2r,O3r,O4r],[1.4,0.76])
           elif oa.seq[i] == "N":
              LPSS.addBond([oa.cb[i],O1r,O2r,O3r,O4r],[1.2,0.76])
           elif oa.seq[i] == "Q":
              LPSS.addBond([oa.cb[i],O1r,O2r,O3r,O4r],[1.2,0.96])
           elif oa.seq[i] == "C":
              LPSS.addBond([oa.cb[i],O1r,O2r,O3r,O4r],[0.6,0.46])
           elif oa.seq[i] == "S":
              LPSS.addBond([oa.cb[i],O1r,O2r,O3r,O4r],[1.2,0.46])
           elif oa.seq[i] == "T":
              LPSS.addBond([oa.cb[i],O1r,O2r,O3r,O4r],[1.2,0.46])
    LPSS.setForceGroup(forceGroup)
    return LPSS

def small_molecular_interaction_to_proteins_term(oa, k=10 * kilocalorie_per_mole, interaction_csv="het_protein_bonds.csv", forceGroup=10):
    k = k.value_in_unit(kilojoule_per_mole)   # convert to kilojoule_per_mole, openMM default uses kilojoule_per_mole as energy.
    smallMolecularInteraction = HarmonicBondForce()
    # interaction_csv = "het_protein_bonds.csv"
    data = pd.read_csv(interaction_csv)
    for i, line in data.iterrows():
        atom1 = int(line["Ligand_atom_index"])
        atom2 = int(line["Protein_atom_index"])
        r = float(line["r"])
        smallMolecularInteraction.addBond(atom1, atom2, r, k)
    
    smallMolecularInteraction.setForceGroup(forceGroup)   # start with 11, so that first 10 leave for user defined force.
    return smallMolecularInteraction


def small_molecular_interaction_to_proteins_term1(self, k=0.1 * kilocalorie_per_mole,k_screening = 1.0, screening_length=1.0, forceGroup=10):
    Featomid = get_ligand_Fe_atoms("./start-openmmawsem.pdb")
    print (Featomid)
    Featomidr = Featomid - 1
    print ("protein ligand potential on")
    k = k.value_in_unit(kilojoule_per_mole)
    small_molecular_interaction_to_proteins1 = CustomBondForce(f"{k}*residue_strength*0.5*(1-tanh((0.9-(r-0.2)/2.0)*6.0))")
    small_molecular_interaction_to_proteins1.addPerBondParameter("residue_strength")
    structure_interactions_lps = []
   
    for i in range(self.nres): 
        residue_strength = 0.0
        if self.seq[i] == "M" or self.seq[i]=="Y":
           residue_strength = 1.0
        if self.seq[i] == "C":
           residue_strength = 3.0
        if self.seq[i] == "H":
           residue_strength = 10.0
        if residue_strength !=0.0:
           print (residue_strength)
           structure_interactions_lps.append([self.cb[i], Featomidr,[residue_strength]])
    for structure_interaction_lp in structure_interactions_lps:
            small_molecular_interaction_to_proteins1.addBond(*structure_interaction_lp)
    small_molecular_interaction_to_proteins1.setForceGroup(forceGroup)
    print (structure_interactions_lps)
    return small_molecular_interaction_to_proteins1

def small_molecular_interaction_to_proteins_term2(oa, k=1.4 * kilocalorie_per_mole,k_screening = 1.0, screening_length=1.0,forceGroup=9):
    Featomid = get_ligand_Fe_atoms("./start-openmmawsem.pdb")
    Natomid = get_ligand_N_atoms("./start-openmmawsem.pdb")
    print ("shortrange")
    print (Featomid,Natomid)
    Featomidr = Featomid - 1
    nres = oa.nres
    N1r = Natomid[0] - 1
    N2r = Natomid[1] - 1
    N3r = Natomid[2] - 1
    N4r = Natomid[3] - 1
    k = k.value_in_unit(kilojoule_per_mole)
    LP_function = ''.join(["kSSresId*0.5*(1+tanh((0.9-r%d)*6))*exp(-2*(r%d-0.57)*(r%d-0.57))*((tanh(2*(sin(N1%d)-0.5))+1)/2.0+(tanh(2*(sin(N2%d)-0.5))+1)/2.0+(tanh(2*(sin(N3%d)-0.5))+1)/2.0+(tanh(2*(sin(N4%d)-0.5))+1)/2.0)/3.6*(tanh(16*(sin(N1%d)-0.86))+1)/2.0*(tanh(16*(sin(N2%d)-0.86))+1)/2.0*(tanh(16*(sin(N3%d)-0.86))+1)/2.0*(tanh(16*(sin(N4%d)-0.86))+1)/2.0*0.5*(1-tanh((r%d-0.7)*100)) +" \
                            % (i,i,i, i, i, i, i , i, i, i,i,i) for i in range(1)])[:-1]
    LP_function = f'-{k}*(' + LP_function + ");"
    LP_parameters = ''.join([f"r{i}=distance(p1, p2);\
                                 N1{i}=angle(p1, p2, p3);\
                                 N2{i}=angle(p1, p2, p4);\
                                 N3{i}=angle(p1, p2, p5);\
                                 N4{i}=angle(p1, p2, p6);"\
                            for i in range(1)])
    LP_string = LP_function+LP_parameters
    LPSS = CustomCompoundBondForce(6, LP_string)
    LPSS.addPerBondParameter("kSSresId")
    for i in range(oa.nres):
        if i not in oa.chain_starts and i not in oa.chain_ends and not oa.res_type[i] == "IGL" and not oa.res_type == "IPR":
           if oa.seq[i] == "M" or oa.seq[i]=="Y":
              LPSS.addBond([oa.cb[i],Featomidr,N1r,N2r,N3r,N4r],[1.0])
           if oa.seq[i] == "C":
              LPSS.addBond([oa.cb[i],Featomidr,N1r,N2r,N3r,N4r],[3.0])
           if oa.seq[i] == "H":
              LPSS.addBond([oa.cb[i],Featomidr,N1r,N2r,N3r,N4r],[10.0])
    LPSS.setForceGroup(forceGroup)
    return LPSS



def small_molecular_interaction_to_proteins_term3(oa, k=0.1 * kilocalorie_per_mole,k_screening = 1.0, screening_length=1.0,forceGroup=10):
    Featomid = get_ligand_Fe_atoms("./start-openmmawsem.pdb")
    Natomid = get_ligand_N_atoms("./start-openmmawsem.pdb")
    print ("longrange")
    print (Featomid,Natomid)
    Featomidr = Featomid - 1
    nres = oa.nres
    N1r = Natomid[0] - 1
    N2r = Natomid[1] - 1
    N3r = Natomid[2] - 1
    N4r = Natomid[3] - 1
    k = k.value_in_unit(kilojoule_per_mole)
    LP_function = ''.join(["kSSresId*0.5*(1+tanh((0.9-(r%d+0.5)/2.0)*6))*exp(-0.5*(r%d-0.57)*(r%d-0.57))*((tanh(4*(sin(N1%d)-0.5))+1)/2.0+(tanh(4*(sin(N2%d)-0.5))+1)/2.0+(tanh(4*(sin(N3%d)-0.5))+1)/2.0+(tanh(4*(sin(N4%d)-0.5))+1)/2.0)/4*(tanh(16*(sin(N1%d)-0.86))+1)/2.0*(tanh(16*(sin(N2%d)-0.86))+1)/2.0*(tanh(16*(sin(N3%d)-0.86))+1)/2.0*(tanh(16*(sin(N4%d)-0.86))+1)/2.0 +" \
                            % (i,i,i, i, i, i, i , i, i, i,i) for i in range(1)])[:-1]
    LP_function = f'-{k}*(' + LP_function + ");"
    LP_parameters = ''.join([f"r{i}=distance(p1, p2);\
                                 N1{i}=angle(p1, p2, p3);\
                                 N2{i}=angle(p1, p2, p4);\
                                 N3{i}=angle(p1, p2, p5);\
                                 N4{i}=angle(p1, p2, p6);"\
                            for i in range(1)])
    LP_string = LP_function+LP_parameters
    LPSS = CustomCompoundBondForce(6, LP_string)
    LPSS.addPerBondParameter("kSSresId")
    for i in range(oa.nres):
        if i not in oa.chain_starts and i not in oa.chain_ends and not oa.res_type[i] == "IGL" and not oa.res_type == "IPR":
           if oa.seq[i] == "M" or oa.seq[i]=="Y":
              LPSS.addBond([oa.cb[i],Featomidr,N1r,N2r,N3r,N4r],[1.0])
           if oa.seq[i] == "C":
              LPSS.addBond([oa.cb[i],Featomidr,N1r,N2r,N3r,N4r],[3.0])
           if oa.seq[i] == "H":
              LPSS.addBond([oa.cb[i],Featomidr,N1r,N2r,N3r,N4r],[10.0])
    LPSS.setForceGroup(forceGroup)
    return LPSS

def burial_term_ligand(oa,k=1.9 * kilocalorie_per_mole,forceGroup=21):
    nres = oa.nres
    k = k.value_in_unit(kilojoule_per_mole)/nres
    Oatomid = get_ligand_O_atoms("./start-openmmawsem.pdb")
    O1r = Oatomid[0] - 1
    O2r = Oatomid[1] - 1
    O3r = Oatomid[2] - 1
    O4r = Oatomid[3] - 1
    LP_function = ''.join(["0.5*(1-tanh((r1%d-0.6)*8))+0.5*(1-tanh((r2%d-0.6)*8))+0.5*(1-tanh((r3%d-0.6)*8))+0.5*(1-tanh((r4%d-0.6)*8))+" \
                            % (i,i,i, i) for i in range(1)])[:-1]
    LP_function = f'{k}*(' + LP_function + ");"
    LP_parameters = ''.join([f"r1{i}=distance(p1, p2);\
                                 r2{i}=distance(p1, p3);\
                                 r3{i}=distance(p1, p4);\
                                 r4{i}=distance(p1, p5);"\
                            for i in range(1)])
    LP_string = LP_function+LP_parameters
    LPSS = CustomCompoundBondForce(5, LP_string)
    for i in range(oa.nres):
        if i not in oa.chain_starts and i not in oa.chain_ends and not oa.res_type[i] == "IGL" and not oa.res_type == "IPR":
              LPSS.addBond([oa.cb[i],O1r,O2r,O3r,O4r])
    LPSS.setForceGroup(forceGroup)
    return LPSS

def set_up_forces(oa, computeQ=True, submode=0, contactParameterLocation=".", membrane_center=-0*angstrom):
    # apply forces
    forces = [
        con_term(oa, forceGroup=11),
        chain_term(oa, forceGroup=12),
        chi_term(oa, forceGroup=13),
        excl_term(oa, periodic=False, forceGroup=14),
        rama_term(oa, forceGroup=15),
        rama_proline_term(oa, forceGroup=15),
        rama_ssweight_term(oa, k_rama_ssweight=2*8.368, forceGroup=15),
        contact_term(oa),
        # for membrane protein simulation use contact_term below.
        # contact_term(oa, z_dependent=True, inMembrane=True, membrane_center=membrane_center, k_relative_mem=3),
        beta_term_1(oa),
        beta_term_2(oa),
        beta_term_3(oa),
        pap_term_1(oa),
        pap_term_2(oa),
        # er_term(oa),
        # membrane_term(oa, k=1*kilocalorie_per_mole, membrane_center=membrane_center),
        # membrane_preassigned_term(oa, k=1*kilocalorie_per_mole, membrane_center=membrane_center, zimFile="PredictedZim"),
        fragment_memory_term(oa, frag_file_list_file="./frags.mem", npy_frag_table="./frags.npy", UseSavedFragTable=True),
        # fragment_memory_term(oa, frag_file_list_file="./single_frags.mem", npy_frag_table="./single_frags.npy", UseSavedFragTable=True),
        #fragment_memory_term(oa, frag_file_list_file="./fragsLAMW.mem", npy_frag_table="./frags.npy", UseSavedFragTable=True),
        # debye_huckel_term(oa, chargeFile="charge.txt"),
        # debye_huckel_term(oa)
    ]
    if computeQ:
        forces.append(rg_term(oa))
        forces.append(q_value(oa, "crystal_structure-cleaned.pdb", forceGroup=1))
        # forces.append(qc_value(oa, "crystal_structure-cleaned.pdb"))
    if submode == 0:
        additional_forces = [
            # contact_term(oa),
            small_molecular_bond_term(oa),
            small_molecular_angle_term(oa),
            small_molecular_dihedral_angle_term(oa),
            small_molecular_vdw_term(oa),
            small_molecular_debye_huckel_term(oa),
            small_molecular_interaction_to_proteins_term2(oa),
            small_molecular_interaction_to_proteins_term1(oa),
            protein_small_molecular_debye_huckel_term(oa),
            small_molecular_interaction_to_proteins_hb(oa),
            small_molecular_interaction_to_proteins_hb2(oa),
            small_molecular_interaction_to_proteins_HEC_CS(oa, k=175 * kilocalorie_per_mole,forceGroup=3),
            small_molecular_exclude_volume_term(oa),
            group_constraint_by_distance(oa),
            burial_term_ligand(oa)
        ]
        forces += additional_forces
    return forces
