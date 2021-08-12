# Version 1.9

import math, copy
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import minres





def site_prob(beta, site_ene, pre):
    Pa = np.zeros(len(site_ene))
    Pa = np.exp(-beta*site_ene) / pre
    part_func = np.sum(Pa)

    Pa /= part_func
    Pa_sqrt = np.sqrt(Pa)
    return Pa, Pa_sqrt



def rates(beta, o_o_ind_list, o_o_elist, site_ene, pre, preT, Pa, Pa_sqrt):
    wab = np.zeros((len(site_ene),len(site_ene)))
    Wab = np.zeros((len(site_ene),len(site_ene)))
    
    for j in range(len(o_o_ind_list)):
        for k in o_o_ind_list[j]:
            wab[j][k] =  pre[j] * np.exp(-beta*(o_o_elist[j][k]-site_ene[j])) / preT[j][k]
            Wab[j][k] = (Pa_sqrt[j]) * wab[j][k] * (1/Pa_sqrt[k])
            
        wab[j][j] = -np.sum(wab[j])
        Wab[j][j] = wab[j][j]
    return wab, Wab



def get_bvec(o_o_ind_list, Pa_sqrt, wab, transport_vec):
    bvec = np.zeros((len(o_o_ind_list),3))
    
    for j in range(len(o_o_ind_list)):
        for k in o_o_ind_list[j]:
            bvec[j] += Pa_sqrt[j] * wab[j][k] * transport_vec[j][k]    
    return bvec



def get_eta(o_o_ind_list, Wab, bvec):
    eta = np.zeros((3,len(o_o_ind_list)))
    
    Ilist, Jlist, Vlist  = [], [], []
    for j, x in enumerate(o_o_ind_list):
        for k in x:
            Ilist.append(j)
            Jlist.append(k)
            Vlist.append(Wab[j][k])
        Ilist.append(j)
        Jlist.append(j)
        Vlist.append(Wab[j][j])
    
    W = sparse.csr_matrix((Vlist,(Ilist,Jlist)), shape=(len(o_o_ind_list),len(o_o_ind_list)))
    for j in range(3):
        eta[j], info = minres(W, bvec[:,j], x0=np.random.rand(W.shape[0]), tol=1e-8)
    return eta



def get_bias(bvec, eta):
    bias = np.zeros((3,3))
    
    for row in range(3):
        for col in range(3):
            for b,e in zip(bvec[:,row],eta[col]):
                bias[row][col] += b*e
    return bias



def get_bare(o_o_ind_list, Pa, wab, transport_vec):
    Do= np.zeros((3,3))
    
    for j in range(len(o_o_ind_list)):
        for k in o_o_ind_list[j]:
            Do += 0.5 * Pa[j] * wab[j][k] * np.outer(transport_vec[j][k],transport_vec[j][k])    
    return Do



def diffuser(Tlist, beta_list, site_ene, o_o_ind_list, o_o_elist, pre, preT, transport_vec):
    D_T_list = []
    
    for i,T in enumerate(Tlist):
        Pa, Pa_sqrt = site_prob(beta_list[i], site_ene, pre)
        wab, Wab = rates(beta_list[i], o_o_ind_list, o_o_elist, site_ene, pre, preT, Pa, Pa_sqrt)
        bvec = get_bvec(o_o_ind_list, Pa_sqrt, wab, transport_vec)
        eta = get_eta(o_o_ind_list, Wab, bvec)
        bias = get_bias(bvec, eta)
        Do = get_bare(o_o_ind_list, Pa, wab, transport_vec)

        D = np.zeros((3,3))
        D = (Do + bias)*1e-8
        D_T_list.append(np.diag(D))
    return D_T_list




















##########
def posreader(PosName='POSCAR'):
#    """
#    Read the atomic configuration from POSCAR
#
#    Args:
#        PosName (str): the name of the POSCAR File, (default: 'POSCAR')
#    """
    POS = {} #B #Initialize the dictionary for POSCAR information
    Fid = open(PosName,'r')
   
    Line = Fid.readline() 
    POS['CellName'] = Line.split('\n')[0] #B #Comment line                      #Comment line

    Line = Fid.readline()
    Sline = Line.split()
    POS['LattConst'] = float(Sline[0]) #B #Lattice constant                     #Universal scaling factor (lattice constant)
    
    POS['Base'] = [[0.0]*3 for i in range(3)] #B #Initilize the base list
    for i in range(3):                                                          #Three lattice vectors
        Line = Fid.readline()
        Sline = Line.split()
        #@POS['Base'][i] = [float(Sline[i]) for i in range(3)];
        POS['Base'][i] = [float(Sline[i])*POS['LattConst'] for i in range(3)] #!


    Line = Fid.readline()
    Sline = Line.split()
    POS['EleName'] = Sline #B #The name of each element                         #Name of atomic species
    POS['EleNum']= len(POS['EleName']) #B #number of elements involved          #EleNum = 3 (Sr, Ti, O)
       
    Line = Fid.readline()
    Sline = Line.split()
    POS['AtomNum'] = [0]*POS['EleNum']
    POS['AtomSum'] = 0
    for ind, Num in enumerate(Sline):
       POS['AtomNum'][ind] = int(Num)                                           #AtomNum = [32, 32, 96], number of atoms per atomic species
       POS['AtomSum'] += int(Num)                                               #AtomSum = 160, total number of atoms


    Line = Fid.readline()
    Sline = Line.split()
    FL = Sline[0][0] #B #Check the first letter
    if (FL=='S'):                                                               #Selective dynamics
        POS['IsSel'] = 1
        POS['SelMat'] = [['X']*3 for i in range(POS['AtomSum'])]
        Line = Fid.readline()
        Sline = Line.split()
        FL = Sline[0][0] #B #Check the first letter for coord
    else:
        POS['IsSel'] = 0

    
    #B # Set up the lattice type            
    if (FL=='D') | (FL=='d'):                                                   #Direct coordinates
        POS['LatType'] = 'Direct'
    elif (FL=='C') | (FL=='c'):                                                 #Cartesian coordinates
        POS['LatType'] = 'Cartesian'
    else:
        print("Please check the POSCAR file, the lattice type is not direct or cartesian")
 
    POS['LattPnt'] = [[0.0]*3 for i in range(POS['AtomSum'])] #B #Initialize lattice points

    if (POS['LatType']=='Direct'): #!
        for i in range(POS['AtomSum']): #!
            Line = Fid.readline() #!
            Sline = Line.split() #!
            POS['LattPnt'][i] = [float(Sline[i]) for i in range(3)] #!
            if(POS['IsSel']): #!
                POS['SelMat'][i] = [Sline[i+3] for i in range(3)] #!
                
    elif (POS['LatType']=='Cartesian'): #!
        BaseInv = np.linalg.inv(POS['Base']) #!
        for i in range(POS['AtomSum']): #!
            Line = Fid.readline() #!
            Sline = Line.split() #!
            POS['LattPnt'][i] = [float(Sline[i]) for i in range(3)] #!
            POS['LattPnt'][i] = list(np.dot(BaseInv, POS['LattPnt'][i])) #!
            if(POS['IsSel']): #!
                POS['SelMat'][i] = [Sline[i+3] for i in range(3)] #!

    else: #!
        print("Please check the POSCAR file, the lattice type is not direct or cartesian") #!
        
#@    for i in range(POS['AtomSum']):
#@        Line = Fid.readline()
#@        Sline = Line.split()
#@        POS['LattPnt'][i] = [float(Sline[i]) for i in range(3)]                 #LattPnt = [0.25, 0.0, 0.125], Direct coordinates
#@        if(POS['IsSel']):
#@            POS['SelMat'][i] = [Sline[i+3] for i in range(3)]

    Fid.close()
    #B #The current version does not support reading the POSCAR with velocity information!!!!!!!!!!!!!!!!
    return POS
##########





##########
def poswriter(PosName,POS):
#    """
#    Write out the POS into a POSCAR file
#
#    Args:
#        PosName: the name of the POSCAR file
#        POS: the POS dictionary
#    """
    Fid = open(PosName,'w')
    Fid.write('%s ' %POS['CellName'])
    Fid.write('\n')
    
    Fid.write('%f \n' %POS['LattConst'])    
    for i in range(3):
        Fid.write('%f %f %f \n' %(POS['Base'][i][0], POS['Base'][i][1], POS['Base'][i][2]))

    for i in range(POS['EleNum']):
        Fid.write('%s ' %POS['EleName'][i])
    Fid.write('\n')

    for i in range(POS['EleNum']):
        Fid.write('%i ' %POS['AtomNum'][i])
    Fid.write('\n')
    
    if (POS['IsSel']):
        Fid.write('Selective Dynamics \n')

    Fid.write('%s \n' %POS['LatType'])
    for i in range(POS['AtomSum']):
        Fid.write('%f %f %f ' %(POS['LattPnt'][i][0], POS['LattPnt'][i][1], POS['LattPnt'][i][2]))
        if (POS['IsSel']):
            Fid.write('%s %s %s ' %(POS['SelMat'][i][0], POS['SelMat'][i][1], POS['SelMat'][i][2]))
        Fid.write('\n')
    
    Fid.close()
##########





##########
def dismatcreate(POS):
#    """
#    Create the distance matrix for a given POS
#
#    Args:
#        POS: the POS dictionary
#    """
    POS['dismat'] = [[0.0]*POS['AtomSum'] for i in range(POS['AtomSum'])]
    
    for AtomInd1, Pnt1 in enumerate(POS['LattPnt']):
        for AtomInd2, Pnt2 in enumerate(POS['LattPnt']):
            Pnt1=np.array(Pnt1)
            Pnt2=np.array(Pnt2)
            PntDis = Pnt1 - Pnt2
            
            for i in range(3):
                if (PntDis[i]>0.5):
                    PntDis[i] = 1 - PntDis[i]
                elif (PntDis[i]<-0.5):
                    PntDis[i] = PntDis[i] + 1
                elif (PntDis[i]>=-0.5) & (PntDis[i]<=0.5):
                    PntDis[i] = abs(PntDis[i])
                else:
                    print("Something is wrong when calculating dist matrix")
                    
            PntDis = np.dot(PntDis, POS['Base'])
            POS['dismat'][AtomInd1][AtomInd2] = math.sqrt(PntDis[0]**2 + PntDis[1]**2 + PntDis[2]**2)
               
    return POS
##########




















##########
def findGrpLst(Lst):
#    """
#    Group the lst
#    """

    #Lst=[1,1] -> GrpLst=[[0,1]]
    #Lst=[1,2] -> GrpLst=[[0],[1]]
    #Lst=[1,1,2] -> GrpLst=[[0,1],[2]]
    #Lst=[1,2,2] -> GrpLst=[[0],[1,2]]
    #Lst=[1,2,1] -> GrpLst=[[0],[1],[2]]
    
    GrpLst = [[0]]
    Loc = 0
    NLst = len(Lst)
    for Ind in range(1,NLst):
        #B #print Lst[Ind], Lst[Ind - 1];
        if Lst[Ind] == Lst[Ind - 1]:    #Lst의 숫자가 그 전 숫자와 같으면
            GrpLst[Loc].append(Ind)     #같은 []에 숫자를 추가. [0,1,..]
        else:                           
            Loc += 1 
            GrpLst.append([])           #다르면
            GrpLst[Loc].append(Ind)     #다음 []에 숫자를 추가. [0],[1],..
    return GrpLst
##########
    




##########
def grporderPermute(Lst, GrpLst):
#    '''
#    Find the permution of group list
#
#    Args:
#        Lst: The original list, somthing like [1,100,1000]
#        GrpLst: the list of group of Lst, something like [[0,1],2]
#    '''
    
    #Lst=[1,1] -> GrpLst=[[0,1]] -> PermuteGrpLst=[[0]]
    #Lst=[1,2] -> GrpLst=[[0],[1]] -> PermuteGrpLst=[]
    #Lst=[1,1,2] -> GrpLst=[[0,1],[2]] -> PermuteGrpLst=[[0],[1,2]]
    #Lst=[1,2,2] -> GrpLst=[[0],[1,2]] -> PermuteGrpLst=[[0,1],[2]]
    #Lst=[1,2,1] -> GrpLst=[[0],[1],[2]] -> PermuteGrpLst=[]
    
    LstLen = len(Lst)                                       #Lst=[1,1,2] 일 때,
    GrpLen = len(GrpLst)                                    #GrpLst=[[0,1],[2]]
    PermuteGrp = []                                         #LstLen = 3
    flag = 0                                                #GrpLen = 2
    
    for GrpInd1, Grp1 in enumerate(GrpLst):                 #GrpLst=[[0,1],[2]]
        for GrpInd2 in range(GrpInd1,GrpLen):
            Grp2 = GrpLst[GrpInd2]
            PermuteGrp.append([])
            for i in Grp1:                                  # i:[0,1] / j:[0,1],[2] -> (0,0),(0,1),(1,0),(1,1) / (0,2),(1,2)
                for j in Grp2:                              # j:  [2] / j:      [2] -> (2,2)
                    PermuteGrp[flag].append((i,j))          
            if len(PermuteGrp[flag]) == 1: #B #Should have at least 1 item
                PermuteGrp.remove(PermuteGrp[flag])         # 이 중, (2,2)과 같이 한 개 뿐인 것 제거함
            else:
                flag += 1                                   #PermuteGrp = [ [(0,0),(0,1),(1,0),(1,1)], [(0,2),(1,2)]]
    
    PermuteGrpLst = [[] for i in range(len(PermuteGrp))]
    PermuteLst = [[] for i in range(len(PermuteGrp))]
    
    Count = 0
    for i in range(LstLen):                                 #PermuteGrp에서 (i,j)가 몇 번째에 있는지
        for j in range(i+1,LstLen):                         #(i,j) = (0,1) -> (0,2) -> (1,2) -> (2,2)
            for PInd,Permute in enumerate(PermuteGrp):      #Count:     0        1        2        x
                if (i,j) in Permute:                        #PermuteGrp:0        1        1
                    PermuteGrpLst[PInd].append(Count)       #PermuteGrpLst = [ [   0 ], [   1 ,   2 ] ]
                    PermuteLst[PInd].append((i,j))          #PermuteLst =    [ [(0,1)], [(0,2),(1,2)] ]
                    break
            Count += 1
    
    return PermuteGrpLst
##########
    




##########
def grpSort(Lst, GrpLst):   
#    """
#    Sort the Lst with certain group constaint
#    """
    Lst0 = list(Lst)
    for Grp in GrpLst:
        if len(Grp) <= 1:                               #Grp=[0]
            continue                                    #Grp가 한개면 그냥 넘어감
        
        else:                                           #Grp=[0,1]
            SubLst = sorted([Lst0[i] for i in Grp])     #Grp가 두개 이상이면, 여기에 해당하는 Lst를 작은 순서부터 정렬
            for i, ind in enumerate(Grp):               #Lst=[20,10,30] -> [10,20,30]
                Lst0[ind] = SubLst[i]
    return Lst0
##########
    




##########
def lstGrpAdd(Lst, MaxVal, GrpLst):                 
#    """
#    Add one into the lst, this operation is used to when we iterate all the
#    combinations with candidate in Lst variable. In this case, the function
#    would call the next combination, the list would also be ordered in this
#    case, which means [1,2,3] is treated the same as [2,3,1] and [3,2,1].
#
#    Args:
#        Lst: The Lst to be operated, should consist only integers, the minimum
#             of each candidate is 0, while the maximum is MaxVal
#        MaxVal: Maximum value of integers expected in each Lst element
#    """

    #MaxVal=[7,7,91], GrpLst=[[0,1],[2]] 일 때,
    #Lst=[0,0,0] -> [1,0,0] -> ... -> [7,0,0]
    #    [1,1,0] -> [2,1,0] -> ... -> [7,1,0]
    #    [2,2,0] -> [3,2,0] -> ... -> [7,2,0] -> ... [7,7,91]
    #이런 식으로 중복되는 것 없이 리스트를 만듬

    LstLen = len(Lst)
    if (Lst[0] < MaxVal[0]):                        #Lst의 첫번째 수
        Lst[0]+=1                                       #Lst=[0,0,0] -> [1,0,0] -> ... -> [7,0,0]                 

        
    elif (Lst[0] == MaxVal[0]):                     #Lst의 첫번째 수가 MaxVal에 도달했으면
        NotFullInd = LstLen - 1                         

        for i in range(1,LstLen):                   #Lst 중 MaxVal에 도달하지 못한 첫번째 index 찾아서 1을 더함
            if (Lst[i] != MaxVal[i]):
                NotFullInd = i                      
                break                                   #NotFullInd=1
        Lst[NotFullInd] += 1                            #Lst=[7,0,0] -> [7,1,0]
  
        #B #Find the belonging group of NotFullInd
        for GrpInd, Grp in enumerate(GrpLst):       
            if NotFullInd in Grp:                   
                ChgInd=GrpInd                       
                break
                
        for i in range(NotFullInd):
            if i in GrpLst[ChgInd]:                 #NotFullInd=1가 Grp=[0,1]에 있다면, Lst의 앞 숫자에 뒤 숫자를 넣음
                Lst[i] = Lst[NotFullInd]                #Lst=[7,1,0] -> [1,1,0]
            else:                                   #NotFullInd=2가 Grp=[2]에 있다면, Lst의 앞 숫자에 0을 넣음
                Lst[i] = 0                              #Lst=[7,7,1] -> [0,0,1]
    return Lst
##########





##########
def lstOrderAdd(Lst, MaxVal):
#    """
#    Add one into the lst, this operation is used to when we iterate all the
#    combinations with candidate in Lst variable. In this case, the function
#    would call the next combination, the list would also be ordered in this
#    case, which means [1,2,3] is treated the same as [2,3,1] and [3,2,1].
#
#    Args:
#        Lst: The Lst to be operated, should consist only integers, the minimum
#             of each candidate is 0, while the maximum is MaxVal
#        MaxVal: Maximum value of integers expected in each Lst element
#    """
    
    LstLen = len(Lst)
    
    if (Lst[0] < MaxVal[0]):
        Lst[0]+=1
        
    elif (Lst[0] == MaxVal[0]):
        NotFullInd = LstLen - 1
        
        for i in range(1,LstLen):
            if (Lst[i] != MaxVal[i]):
                NotFullInd = i
                break            
        Lst[NotFullInd] += 1
        
        for i in range(NotFullInd):
            Lst[i] = Lst[NotFullInd]

    return Lst
##########





##########
def listPermute(Lst):
#    '''
#    Create permution with the consideration of degree of freedom
#    '''
    
    #[2,3] -> [2],[3]
    #[2,3],[2,3] -> [2,2],[3,2],[3,3]
    #[2,3],[4,5] -> [2,4],[3,4],[2,5],[3,5]

    PermuteLst = []
    Lstlen = len(Lst)
    LstItemInd = [0]*Lstlen
    MaxItemInd = [0]*Lstlen
    
    for Ind, LstItem in enumerate(Lst):
        MaxItemInd[Ind] = len(LstItem) - 1

    GrpLst = findGrpLst(Lst)
    while (LstItemInd[-1]<=MaxItemInd[-1]):
        ItemLst = []
        for i, Ind in enumerate(LstItemInd):
            ItemLst.append(Lst[i][Ind])
        PermuteLst.append(ItemLst)
        LstItemInd = lstGrpAdd(LstItemInd, MaxItemInd, GrpLst)
    
    return PermuteLst
##########




















##########
#@def ceFind(SubLatt, POSRef, NCut=3, Isprint=0, DCut='default'):
def ceFind(SubLatt, POSRef, NCut, DCut, Isprint=0): #!
#    '''
#    Method to find the clusters with a given reference lattice
#
#    Args:
#        SubLatt: the projection of solid solution into reference lattice
#                 something like [[0,1],[1,2],[3,4]];
#        POSRef: POSCAR dictionary for reference lattice
#        NCut: Cutoff size of clusters (default: 3)
#        DCut: Cutoff length of each dimension of the cluster 
#              (default: Half of the box size)
#    '''
    
    #@print('#############################################################')
    if DCut == 'default':
        DCut = 100.0
        TS = 0.3
        for i in range(3):
            DMax = max(POSRef['Base'][i])/2.0 + TS
            if DMax < DCut:
                DCut = DMax
    #@print('Cutoff cluster length is %f A' %DCut)
    
    NSubLatt = POSRef['EleNum']                                                 #NSubLatt=3, (A,B,O three sublattices)
    ClusterDesLst = []
    PrimDistLst = []
    AllPrimLattLst = []
    ClusterNum = []
    IndMax = max(SubLatt[-1])                                                   #SubLatt=[[0],[1,2],[3,4]] -> IndMax=4

    FreeSubLatt = []
    FreePrim = []
    for i in range(NSubLatt):                                                   #SubLatt =     [ [0], [1,2], [3,4]]
        if len(SubLatt[i]) == 1: #!
            FreeSubLatt.append(SubLatt[i][0:-1]) #!
        if len(SubLatt[i]) > 1:                                                 #                 0    1      2
            FreePrim.append(i)                                                  #FreePrim =    [       1   ,  2   ]
            FreeSubLatt.append(SubLatt[i]) #!
        #@FreeSubLatt.append(SubLatt[i][0:-1]) #B #Get rid of last one            #FreeSubLatt = [  [], [1]  , [3]  ]
        
    NFreePrim = len(FreePrim)
    NFreeSubLatt = len(FreeSubLatt)
    FreePrim = np.array(FreePrim)                                               #NFreePrim=2
    FreeSubLatt = np.array(FreeSubLatt)                                         #NFreeSubLatt=3
    #B #print(NFreePrim,NFreeSubLatt,FreePrim,FreeSubLatt);

    for N in range(2,NCut+1):                                                   #2개 이상의 원자로 이루어진 cluster
        PrimIndLst = [0]*N                                                      
        PrimDistLst.append([])
        AllPrimLattLst.append([])
        while (PrimIndLst[-1]<=NFreePrim-1):                                    #PrimIndLst=[0,0]->[1,0]->[1,1]
            #B #print(PrimIndLst);
            PrimLattLst = list(FreePrim[PrimIndLst])                            #PrimLattLst=[1,1]->[2,1]->[2,2]
            AllPrimLattLst[N-2].append(PrimLattLst)                             #AllPrimLattLst=[[1,1],[2,1],[2,2]]
                                                                                #              =[[1,1,1],[2,1,1],[2,2,1],[2,2,2]]
            DistLst = findCluster(POSRef,PrimLattLst,DCut)
            
            PrimDistLst[N-2].append(DistLst)                                    #PrimDistLst=[   [1,1]거리,  [2,1]거리,  [2,2]거리            ]
            PrimIndLst=lstOrderAdd(PrimIndLst,[NFreePrim-1]*N)          #           =[ [1,1,1]거리,[2,1,1]거리,[2,2,1]거리,[2,2,2]거리 ]
            
    #@print('The Distance list of primary lattice is '+str(PrimDistLst))
    #@print('The cluster made from primary lattice is '+str(AllPrimLattLst))



    ClusterDesLst.append([])
    ClusterNum.append(0)
    for SubLatt in FreeSubLatt:                                                 #1개의 원자로 이루어진 cluster
        if SubLatt:
            #@print(SubLatt)
            for Latt in SubLatt:
                #@ClusterDesLst[0].append([Latt])                               #ClusterDesLst[0]=[[1],[3]]
                ClusterDesLst[0].append([[Latt]]) #!
                ClusterNum[0] += 1                                              #ClusterNum[0]=2
    #B #print(ClusterDesLst);


    
    for N in range(2,NCut+1):                                                   #2개 이상의 원자로 이루어진 cluster
        IndLst = [0]*N
        ClusterDesLst.append([])
        ClusterNum.append(0)
        
        while (IndLst[-1]<=NFreeSubLatt-1):
            LattLst = list(FreeSubLatt[IndLst])                                 #LattLst=[ [], []]->[[1], []]->[[3], []]->
                                                                                #       =[[1],[1]]->[[3],[1]]->[[3],[3]]->
            if not [] in LattLst:                                               #       = ... -> [[1],[1],[1]] -> ...
                #B #print('LattLst = '+str(LattLst));
                PrimLattLst = [0]*N
                for LattInd, Latt in enumerate(LattLst):
                    if Latt in list(FreeSubLatt): 
                        SubInd = list(FreeSubLatt).index(Latt)
                        PrimLattLst[LattInd] = SubInd                           #LattLst=[[1],[1]] -> PrimLattLst=[1,1]
                    else:
                        print('Cannot Latt in FreeSubLatt!!!')
                #@print('PrimLattLst = '+str(PrimLattLst))
                
                if PrimLattLst in AllPrimLattLst[N-2]:
                    PrimInd = AllPrimLattLst[N-2].index(PrimLattLst)
                    DistLst = PrimDistLst[N-2][PrimInd]                         #DistLst: LattLst에 해당하는 거리 정보
                else:
                    print('Cannot find the relevant PrimLattLst!!!')
                    break
                
                for Dist in DistLst:                                            #LattLst, DisLst를 모아서 Cluster 만듬
                    PermuteLattLst = listPermute(LattLst)
                    for PermuteLst in PermuteLattLst:
                        Dist = [round(elem,2) for elem in Dist] #!
                        Cluster = [PermuteLst,Dist]
                        if not Cluster in ClusterDesLst[N-2]:
                            ClusterDesLst[N-1].append(Cluster)
                            ClusterNum[N-1] += 1
                            
            IndLst = lstOrderAdd(IndLst,[NFreeSubLatt-1]*N) #B #Next one
    ClusterSum = sum(ClusterNum)
    #@print('#############################################################')

    if (Isprint):
        #@print('#############################################################')
        print('%i indepedent Clusters have been found in this structure' %(ClusterSum))
        for N in range(1,NCut+1):
            print('%i Clusters with %i atoms is given below:' %(ClusterNum[N-1], N))
            ClusterStr = ''
            for Cluster in ClusterDesLst[N-1]:
                ClusterStr+=str(Cluster)
                #@ClusterStr+='\t'
                ClusterStr+='\n' #!
            print(ClusterStr)
        #@print('#############################################################')

    return ClusterSum, ClusterNum, ClusterDesLst
##########





##########
def findCluster(POSRef, LattLst, DCut):
#    '''
#    Find the Distance Lst for a given PrimAtomLst
#
#    Args:
#        POSRef: dictionary of POSRef
#        PrimAtomLst: atom list in PrimAtomLst
#        DCut: Cutoff distance of
#
#    '''
    
    NLst = len(LattLst)                                                         #LattLst = [2,1], [O,B] 일 때
    IndLst = [0]*NLst                                                           #IndLst = [0,0] -> [1,0] -> ... -> [95,0] ->
    GIndLst = [0]*NLst                                                          #                  [1,1] -> ... -> [95,1] ->
    #@TS = 0.05*NLst
    TS = 0.1 #!
    DistLst = []
    IndLstMax = []
    
    for i in range(NLst):
        IndLstMax.append(POSRef['AtomNum'][LattLst[i]]-1)                       #IndLstMax = [95,31], LattLst에 해당하는 원자 개수 -1

    while (IndLst[-1]<=IndLstMax[-1]):
        for i, Ind in enumerate(IndLst):
            #@Indtmp = LattLst[i] - 1
            Indtmp = LattLst[i] #!
            GIndLst[i] = Ind + sum(POSRef['AtomNum'][0:Indtmp])                 #GIndLst = [64,32] -> [65,32] -> ...
            
        Dist = []
        #@GrpLst = MathKit.findGrpLst(LattLst)
        for i in range(NLst):
            for j in range(i+1,NLst):
                Distmp = POSRef['dismat'][GIndLst[i]][GIndLst[j]]
                Dist.append(Distmp)
        
        GrpLst = findGrpLst(LattLst) #!
        PermuteGrpLst = grporderPermute(LattLst,GrpLst)
        Dist = grpSort(Dist,PermuteGrpLst)
        
        flag = 1
        for Disttmp in DistLst:
            Distmp = grpSort(Disttmp,PermuteGrpLst)
            #@DistDiff = sum(abs(np.array(Dist)-np.array(Disttmp)))
            DistDiff = sum(abs(np.array(Dist)-np.array(Distmp))) #!
            if (DistDiff < TS):
                flag = 0
        if (min(Dist) > 0) & (max(Dist) < DCut) & flag:
            DistLst.append(Dist)
            
        #@GrpLst = MathKit.findGrpLst(LattLst)
        IndLst = lstGrpAdd(IndLst,IndLstMax,GrpLst)

    return DistLst
##########





##########
#@def clustercount1(Clusterdes, POS, TS=0.2):
def clustercount1(Clusterdes, POS, TS=0.1):
#    '''
#    enumerate and count clusters in a given lattce, this version is cleaner
#    and more robust than the method below: clustercount
#
#    Args:
#        Clusterdes: Cluster description, in the format of list, somthing
#                    like [[[0,1,2],[2.6,2.7,2.8]],[[1,1],[2.5]],[[2]]]
#        POS: Dictionary containing the position information, in the format of POSCAR
#        TS: Allowed variation of cluster bond length
#        Outputs: ClusterLst, which is a list with all the description of indentified
#                 clusters as specified in Clusterdes
#    '''

                                                                                # Sr Fe O Ti
                                                                                # ClusterDes = [ [[1]], [[1,2],[2.0]], [[1,1,2],[3.9,2.1,1.9]] ]
    
    ClusterNum = len(Clusterdes)
    ClusterLst = [[] for i in range(ClusterNum)]
    SumLst = [sum(POS['AtomNum'][0:i]) for i in range(POS['EleNum'])]           #SumLst = [0,32,40,132] (각 원자의 시작 번호)

    for CInd, Cluster in enumerate(Clusterdes):
        CSize = len(Cluster[0]) #B #Cluster Size                                          

        if CSize == 1:                                                          #clustercount랑 같음
            for i in range(POS['AtomNum'][Cluster[0][0]]):
                GInd = i + sum(POS['AtomNum'][0:Cluster[0][0]])
                ClusterLst[CInd].append([GInd])
                
        else:                                                                   #여기부터 clustercount보다 개선됨
            IndLst = [0]*CSize                                                  #Cluster = [[1,1,2],[3.9,2.1,1.9]] 일 때
            IndLstMax = []
            GIndLst = [0]*CSize
            GrpLst = findGrpLst(Cluster[0])                             #GrpLst = [[0,1],[2]]
            PermuteLst = grporderPermute(Cluster[0],GrpLst)             #PermuteLst = [[0],[1,2]]
            DistRef = grpSort(Cluster[1],PermuteLst)                    #DistRef = [3.9,1.9,2.1]
            
            for Ele in Cluster[0]:                                              
                IndLstMax.append(POS['AtomNum'][Ele] - 1)                       #IndLstMax = [7,7,91] (cluster를 구성하는 각 원자의 개수 - 1)
            
            if -1 in IndLstMax: #!
                continue #!
                
            while (IndLst[-1] <= IndLstMax[-1]):
                
                for i, Ind in enumerate(IndLst):                                
                    GIndLst[i] = Ind + SumLst[Cluster[0][i]]                    #GIndLst = [32,32,40] -> [33,32,40] -> ...

                Dist = []
                for i in range(CSize):                                          #GIndLst에 해당하는 원자 사이의 거리
                    for j in range(i+1,CSize):
                        Dist.append(POS['dismat'][GIndLst[i]][GIndLst[j]])
                        
                flag = 1                                                        
                Dist = grpSort(Dist,PermuteLst)
                for Dind,D in enumerate(Dist):
                    if abs(D - DistRef[Dind]) > TS:                             #|원자 사이 거리 - 설정| > 0.2
                        flag = 0
                        break
                if flag:
                    ClusterLst[CInd].append(list(GIndLst))
                    
                lstGrpAdd(IndLst,IndLstMax,GrpLst)                      #IndLst = [0,0,0] -> [1,0,0] -> ...

    return ClusterLst
##########





##########
def countCluster(ClusterLst):
    ClusterCount = []
    for i in range(len(ClusterLst)):
        #B #print('ClusterLst='+str(ClusterLst[i]))
        ClusterCount.append(len(ClusterLst[i]))     #ClusterLst의 각 cluster 수를 셈
    return ClusterCount
##########
    




##########
def clusterE(ClusterLst, ClusterCoef):
#    '''
#    Calculate total energy
#
#    Args:
#        ClusterLst: List of indentified clusters
#        ClusterCoef: ECI of each cluster
#    '''
    
    #B #ClusterCount = [];
    #B #for i in range(len(ClusterLst)):
    #B #    ClusterCount.append(len(ClusterLst[i]));
    ClusterCount = countCluster(ClusterLst);
    #B #print ClusterCount,ClusterCoef, len(ClusterCount), len(ClusterCoef);
    
    ECE = 0.0;
    ECE = ECE + ClusterCoef[0];                         #ClusterCoef[0]: constant
    #B #print ECE;
    for i in range(len(ClusterCount)):
        ECE = ECE + ClusterCount[i]*ClusterCoef[i+1];
        #B #print ECE
    return ECE
##########





##########
def dismatswap(dismat, Ind1, Ind2):                 #dismat: 전체 원자 사이의 거리 정보, Ind1/Ind2: 바꿀 두 원자
#    '''
#    Update the distance matrix
#
#    Args:
#        dismat: distance matrix
#        Ind1, Ind2: the indexes of two atoms that swap positions
#    '''
    
    lendismat = len(dismat[1])                      #lendismat: 전체 원자 개수

    tmp = dismat[Ind1][:]                           #Ind1, Ind2 두 원자 사이의 거리 정보 바꿈
    dismat[Ind1][:] = dismat[Ind2][:]               #(Ind1, 다른원자) -> (Ind2, 다른원자)
    dismat[Ind2][:] = tmp                           #(Ind1, Ind2) -> (Ind2, Ind2), 뒤에서 더 처리함

    for i in range(len(dismat[1])):                 #Ind1, Ind2 두 원자가 아닌 다른 원자들의 거리 정보 바꿈
        if (i!=Ind1)&(i!=Ind2):                     #(다른원자, Ind1) -> (다른원자, Ind2)
            dismat[i][Ind1] = dismat[Ind1][i]
            dismat[i][Ind2] = dismat[Ind2][i]
            
    tmp = dismat[Ind1][Ind2]                        #(Ind1, Ind2) -> (Ind2, Ind2) -> (Ind2, Ind1), 여기서 완성됨
    dismat[Ind1][Ind2] = dismat[Ind1][Ind1]
    dismat[Ind1][Ind1] = tmp
    
    tmp = dismat[Ind2][Ind1]
    dismat[Ind2][Ind1] = dismat[Ind2][Ind2]
    dismat[Ind2][Ind2] = tmp

    return dismat
##########





##########
#@def clusterswap1(ClusterDes, POS, ClusterLst, Atom1, Atom2, Ind1, Ind2, TS=0.2):
def clusterswap1(ClusterDes, POS, ClusterLst, Atom1, Atom2, Ind1, Ind2, TS=0.1):
#    '''
#    Update the cluster information after swapping atoms
#    This is a cleaner and more robust version of clusterswap method below
#
#    Args:
#        ClusterDes: Description about clusters
#        POS: POSCAR dictionary
#        ClusterLst: Cluster information
#        Atom1, Atom2: Atom sublattice
#        Ind1, Ind2: Atom indices
#    '''
    
    ClusterNum = len(ClusterLst)
    SumLst = [sum(POS['AtomNum'][0:i]) for i in range(POS['EleNum'])]

    ClusterLst_cp = copy.deepcopy(ClusterLst) #!
    for LstInd, Lst in enumerate(ClusterLst):
        for Ind, AtomInd in enumerate(Lst):
            if (Ind1 in AtomInd) | (Ind2 in AtomInd):
                #@ClusterLst[LstInd].remove(AtomInd)
                ClusterLst_cp[LstInd].remove(AtomInd) #!
    ClusterLst = copy.deepcopy(ClusterLst_cp) #!

    for CInd, Cluster in enumerate(ClusterDes):
        CSize = len(Cluster[0])
        
        if (CSize==1) & (Atom1==Cluster[0][0]):
            #@ClusterLst[ClusterInd].append([Ind1])
            ClusterLst[CInd].append([Ind1]) #!
            
        elif (CSize==1) & (Atom2==Cluster[0][0]):
            #@ClusterLst[ClusterInd].append([Ind2])
            ClusterLst[CInd].append([Ind2]) #!
            
        else:
            for AtomI, Atom in enumerate([Atom1, Atom2]):                       #O <-> VO, Atom1=3(O), Atom2=4(VO)
                if Atom in Cluster[0]:                                          #Cluster = [2,3,3]인 경우, [Fe,O,O]
                    
                    AtomInd = [Ind1,Ind2][AtomI]                                #AtomInd: 바뀐 O 원자 번호
                    AtomLoc = Cluster[0].index(Atom)                            #AtomLoc=1, Atom1=3이 Cluster=[2,3,3]중 몇 번째인지
                    IndLst = [0]*(CSize-1)
                    IndLstMax = []
                    GIndLst = [0]*(CSize-1)
                    
                    GrpLst = findGrpLst(Cluster[0])
                    PermuteLst = grporderPermute(Cluster[0],GrpLst)
                    DistRef = grpSort(Cluster[1],PermuteLst)
                    
                    ClusterTmp = copy.deepcopy(Cluster[0])                      #Cluster=[2,3,3]에서 Atom1=3을 뺀다
                    ClusterTmp.remove(Atom)                                     #ClusterTmp = [2,3]
                    GrpLst_tmp = findGrpLst(ClusterTmp) #!
                    
                    for Ele in ClusterTmp:
                        IndLstMax.append(POS['AtomNum'][Ele] - 1)               #IndLstMax = [7,91]
                    
                    while (IndLst[-1] <= IndLstMax[-1]):
                        for i, Ind in enumerate(IndLst):
                            #@GIndLst[i] = Ind + SumLst[Cluster[0][i]]            
                            GIndLst[i] = Ind + SumLst[ClusterTmp[i]] #!         #GIndLst=[56,64]
                        GIndLst.insert(AtomLoc,AtomInd)                         #GIndLst=[56,바뀐 O 원자,64]
                        GIndLst = grpSort(GIndLst,GrpLst)

                        Dist = []
                        for i in range(CSize):
                            for j in range(i+1,CSize):
                                Dist.append(POS['dismat'][GIndLst[i]][GIndLst[j]])

                        flag = 1
                        Dist = grpSort(Dist,PermuteLst)
                        for Dind, D in enumerate(Dist):
                            #@if abs (D - DistRef[ind]) > TS:
                            if abs (D - DistRef[Dind]) > TS: #!
                                flag = 0
                                break
                        if flag:
                            #@ClusterLst[CInd].append(list(GIndLst))
                            if GIndLst not in ClusterLst[CInd]: #!
                                ClusterLst[CInd].append(GIndLst) #!
                        
                        GIndLst = [0]*(CSize-1) #!
                        #@MathKit.lstGrpAdd(IndLst,IndLstMax,GrpLst)
                        lstGrpAdd(IndLst,IndLstMax,GrpLst_tmp) #!

    return ClusterLst
##########